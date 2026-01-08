import os
import gc
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
)

from ...utils.type_converter import get_module_device
from .model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

# Local paths for internal configs/tokenizers
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
QWEN_PATH = os.path.join(_base_dir, "models_configs", "Qwen3-8B")
CLIP_PATH = os.path.join(_base_dir, "models_configs", "clip-vit-large-patch14")

class ScaledFP8Linear(nn.Module):
    """
    Handles block-wise FP8 scaling during forward pass.
    Keeps weights in FP8 to save memory.
    """
    def __init__(self, weight, scale, bias=None):
        super().__init__()
        self.register_buffer("weight", weight.detach())
        self.register_buffer("scale", scale.detach())
        if bias is not None:
            self.register_buffer("bias", bias.detach())
        else:
            self.bias = None

    def forward(self, x):
        # 1. Cast weight to match x dtype for matmul
        w = self.weight.to(x.dtype)
        
        # 2. Apply block-wise scaling (calculate block size dynamically)
        bh = w.shape[0] // self.scale.shape[0]
        bw = w.shape[1] // self.scale.shape[1]
        s = self.scale.repeat_interleave(bh, dim=0).repeat_interleave(bw, dim=1)
        
        # Match weight shape exactly in case of non-multiple dimensions
        if s.shape != w.shape:
            s = torch.nn.functional.pad(s, (0, w.shape[1]-s.shape[1], 0, w.shape[0]-s.shape[0]), mode='replicate')[:w.shape[0], :w.shape[1]]
            
        w = w * s.to(w.dtype)
        
        return torch.nn.functional.linear(x, w, self.bias)

class HYTextModel(nn.Module):
    def __init__(
        self,
        llm_type: Optional[str] = "qwen3",
        max_length_llm: int = 512,
        active_llm_sd: Optional[Dict] = None,
        active_clip_sd: Optional[Dict] = None,
        dequantizer: Optional[any] = None,
        ggml_ops: Optional[any] = None,
        ggml_tensor: Optional[any] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.enable_llm_padding = kwargs.get("enable_llm_padding", True)
        self._orig_max_length_llm = max_length_llm
        self.crop_start = 0
        self.ggml_ops = ggml_ops
        self.ggml_tensor = ggml_tensor

        # Initialize Tokenizers
        self.sentence_emb_tokenizer = CLIPTokenizer.from_pretrained(CLIP_PATH, local_files_only=True)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)

        # Build Model Structures (Empty weights to save RAM)
        from accelerate import init_empty_weights
        with init_empty_weights():
            clip_config = AutoConfig.from_pretrained(CLIP_PATH, local_files_only=True)
            if hasattr(clip_config, "text_config"): clip_config = clip_config.text_config
            self.sentence_emb_text_encoder = CLIPTextModel(clip_config)
            
            llm_config = AutoConfig.from_pretrained(QWEN_PATH, local_files_only=True)
            self.llm_text_encoder = AutoModelForCausalLM.from_config(llm_config)

        # Patch models to use GGML layers if ops are provided
        if self.ggml_ops:
            self._patch_model_to_ggml(self.sentence_emb_text_encoder, self.ggml_ops)
            self._patch_model_to_ggml(self.llm_text_encoder, self.ggml_ops)

        # Load Weights
        if active_clip_sd:
            self._load_weights_sequentially(self.sentence_emb_text_encoder, active_clip_sd, dequantizer)
        if active_llm_sd:
            self._load_weights_sequentially(self.llm_text_encoder, active_llm_sd, dequantizer)

        self.sentence_emb_text_encoder.eval().requires_grad_(False)
        self.llm_text_encoder.eval().requires_grad_(False)
        
        self.vtxt_dim = self.sentence_emb_text_encoder.config.hidden_size
        self.ctxt_dim = self.llm_text_encoder.config.hidden_size
        self.crop_start = self._compute_crop_start()
        self.max_length_llm = self._orig_max_length_llm + self.crop_start
        print(f"[HYTextModel] CLIP initialized: hidden_size={self.vtxt_dim}")
        print(f"[HYTextModel] LLM initialized: hidden_size={self.ctxt_dim}, crop_start={self.crop_start}")

    def _patch_model_to_ggml(self, model, ggml_ops):
        """Replaces Linear and Embedding layers with GGUF-aware equivalents."""
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                new_layer = None
                if isinstance(module, torch.nn.Linear):
                    new_layer = ggml_ops.Linear(module.in_features, module.out_features, bias=module.bias is not None)
                elif isinstance(module, torch.nn.Embedding):
                    new_layer = ggml_ops.Embedding(module.num_embeddings, module.embedding_dim)
                
                if new_layer:
                    parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, new_layer)

    def _get_module_and_param_name(self, model, key):
        if "." not in key: return model, key
        path, param_name = key.rsplit(".", 1)
        mod = model
        for part in path.split("."):
            mod = getattr(mod, part, None)
            if mod is None: break
        return mod, param_name

    def _load_weights_sequentially(self, model, state_dict, dequantizer=None):
        from accelerate.utils import set_module_tensor_to_device
        test_param = next(model.parameters(), None)
        target_device = test_param.device if (test_param is not None and test_param.device.type != "meta") else torch.device("cpu")
        
        model_name = model.__class__.__name__
        # Filter out obvious non-text-encoder keys (vision, projection, etc.)
        filtered_sd = {k: v for k, v in state_dict.items() if not any(x in k for x in ["vision_model", "visual", "logit_scale", "text_projection"])}
        total_keys = len(filtered_sd)
        print(f"[HYTextModel] Loading {total_keys} weights for {model_name}...")
        
        # Pre-scan for scaling factors (weight_scale_inv, etc.)
        scales = {}
        for k in list(filtered_sd.keys()):
            if any(k.endswith(x) for x in [".weight_scale_inv", ".scale_weight", ".weight_scale"]):
                # Map the base key to its scale
                base_k = k.rsplit(".", 1)[0]
                scales[base_k] = filtered_sd.pop(k)

        loaded_count = 0
        missing_keys = []
        
        for key in list(filtered_sd.keys()):
            try:
                weight = filtered_sd.pop(key)
                target_key = None
                
                # Super-aggressive universal matching logic
                clean_key = key
                if clean_key.startswith("text_model."): clean_key = clean_key[11:]
                elif clean_key.startswith("model."): clean_key = clean_key[6:]
                elif clean_key.startswith("transformer."): clean_key = clean_key[12:]
                
                # Exhaustive prefix search: check if attribute path exists in model
                for p in ["", "text_model.", "model.", "transformer.", "encoder."]:
                    pk = f"{p}{clean_key}"
                    mod, param_name = self._get_module_and_param_name(model, pk)
                    if mod is not None and (hasattr(mod, param_name) or hasattr(mod, "ggml_load_from_state_dict")):
                        target_key = pk
                        break
                
                if target_key is None:
                    missing_keys.append(key)
                    continue

                mod, param_name = self._get_module_and_param_name(model, target_key)
                is_ggml_layer = hasattr(mod, "ggml_load_from_state_dict") or "GGMLLayer" in mod.__class__.__name__
                
                # Dequantize GGUF weights (has tensor_type attribute)
                if dequantizer and hasattr(weight, "tensor_type") and not is_ggml_layer:
                    weight = dequantizer(weight)
                
                # Resolve the target parameter/member and its shape
                param = getattr(mod, param_name, None)
                expected_shape = param.shape if param is not None else getattr(mod, "tensor_shape", None)
                
                # Convert FP8 safetensor weights using ScaledFP8Linear if scales exist
                is_fp8 = hasattr(weight, "dtype") and str(weight.dtype) == "torch.float8_e4m3fn"
                if is_fp8:
                    base_path = key.rsplit(".", 1)[0]
                    scale = scales.get(base_path, None)
                    
                    if scale is not None:
                        # Patch standard Linear with ScaledFP8Linear
                        if isinstance(mod, nn.Linear):
                            mod_path = target_key.rsplit(".", 1)[0]
                            if "." in mod_path:
                                parent_path, child_name = mod_path.rsplit(".", 1)
                                parent = model.get_submodule(parent_path)
                            else:
                                parent = model
                                child_name = mod_path
                            
                            new_mod = ScaledFP8Linear(weight, scale, bias=mod.bias)
                            setattr(parent, child_name, new_mod)
                            loaded_count += 2 # Count weight + scale
                            continue
                        else:
                            # Fallback if not linear (e.g. embedding with unexpected scale)
                            weight = weight.to(torch.bfloat16) * scale.to(torch.bfloat16)
                            loaded_count += 1
                
                # Refined backup shape detection for GGMLLayers
                if expected_shape is None:
                    if hasattr(mod, "out_features"):
                        if param_name == "bias": expected_shape = (mod.out_features,)
                        else: expected_shape = (mod.out_features, mod.in_features)
                    elif hasattr(mod, "num_embeddings"):
                        expected_shape = (mod.num_embeddings, mod.embedding_dim)

                if expected_shape is not None:
                    # Generic shape correction (squeeze/unsqueeze)
                    if expected_shape != weight.shape:
                        if len(expected_shape) == 1 and len(weight.shape) == 2 and weight.shape[1] == 1:
                            weight = weight.squeeze(1)
                    
                    if expected_shape == weight.shape:
                        if is_ggml_layer:
                            setattr(mod, param_name, torch.nn.Parameter(weight, requires_grad=False))
                        else:
                            set_module_tensor_to_device(model, target_key, target_device, value=weight)
                        loaded_count += 1
                    else:
                        missing_keys.append(f"{key} (shape mismatch: {weight.shape} vs {expected_shape})")
                else:
                    missing_keys.append(f"{key} (could not determine target shape)")
            except Exception as e:
                missing_keys.append(f"{key} (error: {str(e)})")
                continue
        
        # Safety Fix: Ensure NO module is left with weight=None or on meta device
        for name, mod in model.named_modules():
            is_ggml_layer = hasattr(mod, "ggml_load_from_state_dict") or "GGMLLayer" in mod.__class__.__name__
            
            # Check for parameters on meta device
            for p_name, param in mod.named_parameters(recurse=False):
                if param is not None and param.device.type == "meta":
                    print(f"[HYTextModel] WARNING: Materializing missing parameter '{name}.{p_name}' from meta tensor")
                    # Use to_empty to materialize on the target device
                    mod.to_empty(device=target_device, recurse=False)
                    break # to_empty handles all params in this module
            
            # Check for buffers on meta device
            for b_name, buffer in mod.named_buffers(recurse=False):
                if buffer is not None and buffer.device.type == "meta":
                    print(f"[HYTextModel] WARNING: Materializing missing buffer '{name}.{b_name}' from meta tensor")
                    mod.to_empty(device=target_device, recurse=False)
                    break

            if is_ggml_layer and hasattr(mod, "weight") and (mod.weight is None or mod.weight.device.type == "meta"):
                shape = getattr(mod, "tensor_shape", None)
                if shape is None:
                    if hasattr(mod, "out_features"): shape = (mod.out_features, mod.in_features)
                    elif hasattr(mod, "num_embeddings"): shape = (mod.num_embeddings, mod.embedding_dim)
                if shape:
                    mod.weight = torch.nn.Parameter(torch.zeros(shape, device=target_device), requires_grad=False)
        
        self.enforce_proper_initialization()
        print(f"[HYTextModel] {model_name} loading complete. Success: {loaded_count}/{total_keys}")
        if missing_keys:
            print(f"[HYTextModel] Missing/Skipped {len(missing_keys)} keys. Top 10: {missing_keys[:10]}")

    def enforce_proper_initialization(self):
        """Fixes position_ids and logit_scale for CLIP."""
        from accelerate.utils import set_module_tensor_to_device
        if self.sentence_emb_text_encoder:
            for path in ["text_model.embeddings", "embeddings"]:
                obj = self.sentence_emb_text_encoder
                for p in path.split("."): 
                    obj = getattr(obj, p, None)
                    if obj is None: break
                if obj and hasattr(obj, "position_ids"):
                    device = obj.position_ids.device if obj.position_ids.device.type != "meta" else torch.device("cpu")
                    max_pos = obj.position_ids.shape[1]
                    pos_ids = torch.arange(max_pos, device=device).unsqueeze(0)
                    set_module_tensor_to_device(self.sentence_emb_text_encoder, f"{path}.position_ids", device, value=pos_ids)
                    print(f"[HYTextModel] Re-initialized CLIP position_ids buffer ({max_pos} tokens)")
            
            if not hasattr(self.sentence_emb_text_encoder, "logit_scale"):
                self.sentence_emb_text_encoder.register_buffer("logit_scale", torch.tensor(4.6052))
                print("[HYTextModel] Initialized missing CLIP logit_scale to 4.6052")
            elif self.sentence_emb_text_encoder.logit_scale.device.type == "meta":
                device = get_module_device(self.sentence_emb_text_encoder)
                if device.type == "meta": device = torch.device("cpu")
                set_module_tensor_to_device(self.sentence_emb_text_encoder, "logit_scale", device, value=torch.tensor(4.6052))
                print("[HYTextModel] Materialized CLIP logit_scale from meta-tensor")

    def encode_llm(self, text: List[str]) -> Tuple[Tensor, Tensor]:
        """Encodes text using the LLM text encoder."""
        device = get_module_device(self.llm_text_encoder)
        target_device = get_module_device(self)
        
        print(f"[HYTextModel] LLM Encoding {len(text)} prompt(s)...")
        try:
            llm_text = [self.llm_tokenizer.apply_chat_template(
                [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=False
            ) for t in text]
        except Exception:
            llm_text = [f"<|im_start|>system\n{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in text]
        
        enc = self.llm_tokenizer(
            llm_text, truncation=True, padding="max_length" if self.enable_llm_padding else False,
            max_length=self.max_length_llm, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = self.llm_text_encoder(**enc, output_hidden_states=True, return_dict=True)
        
        ctxt_raw = out.hidden_states[-1] if hasattr(out, "hidden_states") else out.last_hidden_state
        if ctxt_raw.device != target_device: ctxt_raw = ctxt_raw.to(target_device)

        ctxt_raw = ctxt_raw[:, self.crop_start:self.max_length_llm].contiguous()
        ctxt_length = (enc["attention_mask"].sum(dim=-1).to(target_device) - self.crop_start).clamp(min=0, max=self._orig_max_length_llm)
        
        # Signal check (safely convert to float first)
        c_mean = ctxt_raw.mean().item()
        c_std = ctxt_raw.std().item()
        c_rms = ctxt_raw.pow(2).mean().sqrt().item()
        print(f"[HYTextModel] LLM stats: mean={c_mean:.4f}, std={c_std:.4f}, rms={c_rms:.4f}")
        return ctxt_raw, ctxt_length

    def encode_sentence_emb(self, text: List[str]) -> Tensor:
        device = get_module_device(self.sentence_emb_text_encoder)
        enc = self.sentence_emb_tokenizer(text, truncation=True, padding=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.sentence_emb_text_encoder(**enc)
        v_emb = out.pooler_output.unsqueeze(1) if hasattr(out, "pooler_output") and out.pooler_output is not None else self._encode_pooling(enc["attention_mask"], out.last_hidden_state)
        
        # Signal check
        v_mean = v_emb.mean().item()
        v_std = v_emb.std().item()
        v_rms = v_emb.pow(2).mean().sqrt().item()
        print(f"[HYTextModel] CLIP stats: mean={v_mean:.4f}, std={v_std:.4f}, rms={v_rms:.4f}")
        return v_emb

    def _encode_pooling(self, mask: Tensor, tokens: Tensor) -> Tensor:
        mask_exp = mask.unsqueeze(-1).expand(tokens.size()).float()
        emb = torch.sum(tokens * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        return nn.functional.normalize(emb, p=2, dim=1).unsqueeze(1)

    def encode(self, text: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        ctxt, ctxt_len = self.encode_llm(text)
        return self.encode_sentence_emb(text), ctxt, ctxt_len

    def _compute_crop_start(self) -> int:
        marker = "<BOC>"
        try:
            messages = [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {"role": "user", "content": marker}]
            s = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            s = f"<|im_start|>system\n{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}<|im_end|>\n<|im_start|>user\n{marker}<|im_end|>\n<|im_start|>assistant\n"
            
        full_ids = self.llm_tokenizer(s, return_tensors="pt")["input_ids"][0].tolist()
        marker_ids_full = self.llm_tokenizer(marker, add_special_tokens=False)["input_ids"]
        if hasattr(marker_ids_full, "tolist"): marker_ids = marker_ids_full.tolist()
        else: marker_ids = marker_ids_full
        if isinstance(marker_ids[0], list): marker_ids = marker_ids[0]
        
        for i in range(len(full_ids) - len(marker_ids) + 1):
            if full_ids[i:i+len(marker_ids)] == marker_ids: return i
        return 0
