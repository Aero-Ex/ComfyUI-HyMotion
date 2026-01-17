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

try:
    from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op, register_layout_class
except ImportError:
    from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op
    register_layout_class = None
import comfy.ops

from dataclasses import dataclass

class BlockScaledFP8Layout(QuantizedLayout):
    """
    Handles block-wise FP8 scaling (e.g. Qwen3 models).
    Storage format:
    - qdata: FP8 tensor
    - scale: Block-wise scale tensor (2D)
    - orig_dtype: Original dtype for dequantization
    """
    
    @dataclass
    class Params:
        scale: torch.Tensor
        orig_dtype: torch.dtype
        orig_shape: tuple
        
        def clone(self):
            """Clone the params - required by comfy_kitchen QuantizedTensor"""
            return BlockScaledFP8Layout.Params(
                scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                orig_dtype=self.orig_dtype,
                orig_shape=self.orig_shape
            )

        def to_device(self, device):
            """Move params to device - required by comfy_kitchen QuantizedTensor"""
            return BlockScaledFP8Layout.Params(
                scale=self.scale.to(device) if isinstance(self.scale, torch.Tensor) else self.scale,
                orig_dtype=self.orig_dtype,
                orig_shape=self.orig_shape
            )

        def to_dtype(self, dtype):
            """Change params dtype - required by comfy_kitchen QuantizedTensor"""
            return BlockScaledFP8Layout.Params(
                scale=self.scale.to(dtype) if isinstance(self.scale, torch.Tensor) else self.scale,
                orig_dtype=dtype,
                orig_shape=self.orig_shape
            )
    
    @classmethod
    def quantize(cls, tensor, scale=None, dtype=torch.float8_e4m3fn, **kwargs):
        raise NotImplementedError("BlockScaledFP8Layout only supports loading pre-quantized weights.")

    @staticmethod
    def dequantize(qdata, scale, orig_dtype=None, **kwargs):
        # Handle both direct args and Params object
        if hasattr(scale, 'scale'):
            params = scale
            scale = params.scale
            if orig_dtype is None:
                orig_dtype = params.orig_dtype
        
        if orig_dtype is None:
            orig_dtype = torch.float32
            
        # Efficient block-wise dequantization using broadcasting
        bh = qdata.shape[0] // scale.shape[0]
        bw = qdata.shape[1] // scale.shape[1]
        
        # Reshape qdata to [num_blocks_h, bh, num_blocks_w, bw]
        q_reshaped = qdata.view(scale.shape[0], bh, scale.shape[1], bw)
        
        # Multiply by scale [num_blocks_h, 1, num_blocks_w, 1]
        # Cast to orig_dtype for computation
        dequantized = q_reshaped.to(orig_dtype) * scale.unsqueeze(1).unsqueeze(3).to(orig_dtype)
        
        return dequantized.reshape(qdata.shape)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return qtensor._qdata, qtensor._layout_params.scale

# Register linear op for BlockScaledFP8Layout
@register_layout_op(torch.ops.aten.linear.default, "BlockScaledFP8Layout")
def block_fp8_linear(func, args, kwargs):
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Dequantize to match input_tensor.dtype for math compatibility
    if isinstance(weight, QuantizedTensor):
        qdata, scale = weight.layout_cls.get_plain_tensors(weight)
        weight = weight.layout_cls.dequantize(qdata, scale, orig_dtype=input_tensor.dtype)
    
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    # Ensure bias matches input dtype if present
    if bias is not None and bias.dtype != input_tensor.dtype:
        bias = bias.to(input_tensor.dtype)

    return torch.nn.functional.linear(input_tensor, weight, bias)

# Register the layout
if register_layout_class is not None:
    register_layout_class("BlockScaledFP8Layout", BlockScaledFP8Layout)
else:
    try:
        from comfy.quant_ops import LAYOUTS
        LAYOUTS["BlockScaledFP8Layout"] = BlockScaledFP8Layout
    except ImportError:
        print("[HYTextModel] WARNING: Could not find LAYOUTS or register_layout_class in comfy.quant_ops")

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
        
        # Use ComfyUI ops for patching
        self.ops = comfy.ops.disable_weight_init

        # Initialize Tokenizers
        print(f"[HYTextModel] Initializing tokenizers from {CLIP_PATH} and {QWEN_PATH}...")
        self.sentence_emb_tokenizer = CLIPTokenizer.from_pretrained(CLIP_PATH, local_files_only=True)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, local_files_only=True)

        # Build Model Structures (Empty weights to save RAM)
        print("[HYTextModel] Building model structures with accelerate.init_empty_weights()...")
        from accelerate import init_empty_weights
        with init_empty_weights():
            clip_config = AutoConfig.from_pretrained(CLIP_PATH, local_files_only=True)
            if hasattr(clip_config, "text_config"): clip_config = clip_config.text_config
            self.sentence_emb_text_encoder = CLIPTextModel(clip_config)
            
            llm_config = AutoConfig.from_pretrained(QWEN_PATH, local_files_only=True)
            self.llm_text_encoder = AutoModelForCausalLM.from_config(llm_config)

            # Patch models to use ComfyUI ops or GGML ops
            print(f"[HYTextModel] Patching models with {'GGML' if self.ggml_ops else 'ComfyUI'} ops...")
            patch_ops = self.ggml_ops if self.ggml_ops else self.ops
            self._patch_model_to_ops(self.sentence_emb_text_encoder, patch_ops)
            self._patch_model_to_ops(self.llm_text_encoder, patch_ops)

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
        
        # Caching mechanism
        self._cache = {}
        self._cache_max_size = kwargs.get("cache_max_size", 16)
        
        print(f"[HYTextModel] CLIP initialized: hidden_size={self.vtxt_dim}")
        print(f"[HYTextModel] LLM initialized: hidden_size={self.ctxt_dim}, crop_start={self.crop_start}")

    def _patch_model_to_ops(self, model, ops):
        """Replaces layers with ComfyUI-aware equivalents."""
        to_replace = []
        for name, module in model.named_modules():
            new_layer = None
            if isinstance(module, torch.nn.Linear):
                if hasattr(ops, "Linear"):
                    new_layer = ops.Linear(module.in_features, module.out_features, bias=module.bias is not None)
            elif isinstance(module, torch.nn.Embedding):
                if hasattr(ops, "Embedding"):
                    new_layer = ops.Embedding(module.num_embeddings, module.embedding_dim)
            elif isinstance(module, torch.nn.LayerNorm):
                if hasattr(ops, "LayerNorm"):
                    new_layer = ops.LayerNorm(module.normalized_shape, eps=module.eps, elementwise_affine=module.elementwise_affine)
            elif isinstance(module, torch.nn.GroupNorm):
                if hasattr(ops, "GroupNorm"):
                    new_layer = ops.GroupNorm(module.num_groups, module.num_channels, eps=module.eps, affine=module.affine)
            elif "RMSNorm" in module.__class__.__name__:
                # Handle various RMSNorm implementations (transformers, custom, etc.)
                if hasattr(ops, "RMSNorm"):
                    # Safely get hidden size from weight or normalized_shape
                    hidden_size = None
                    if hasattr(module, "weight") and module.weight is not None:
                        hidden_size = module.weight.shape[0]
                    elif hasattr(module, "normalized_shape"):
                        hidden_size = module.normalized_shape[0]
                    
                    if hidden_size is not None:
                        new_layer = ops.RMSNorm(hidden_size, eps=getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
            
            if new_layer:
                to_replace.append((name, new_layer))

        for name, new_layer in to_replace:
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            try:
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, new_layer)
            except Exception as e:
                print(f"[HYTextModel] Failed to patch {name}: {e}")

    def _get_module_and_param_name(self, model, key):
        if "." not in key: return model, key
        path, param_name = key.rsplit(".", 1)
        mod = model
        for part in path.split("."):
            if part.isdigit():
                # Handle ModuleList/ModuleDict indexing
                try:
                    idx = int(part)
                    if isinstance(mod, (nn.ModuleList, nn.Sequential)):
                        mod = mod[idx]
                    else:
                        mod = getattr(mod, part, None)
                except (ValueError, IndexError, KeyError):
                    mod = None
            else:
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
        print(f"[HYTextModel] Found {len(scales)} scale tensors.")

        loaded_count = 0
        missing_keys = []
        
        for key in list(filtered_sd.keys()):
            try:
                weight = filtered_sd.pop(key)
                target_key = None
                
                # Try keys in order of priority
                # 1. Original key as-is
                # 2. Various prefix combinations
                keys_to_try = [key]
                
                # Generate variations
                clean_key = key
                if clean_key.startswith("text_model."):
                    clean_key = clean_key[11:]
                    keys_to_try.append(clean_key)
                elif clean_key.startswith("transformer."):
                    clean_key = clean_key[12:]
                    keys_to_try.append(clean_key)
                # NOTE: Do NOT strip "model." prefix - Qwen3 uses model.layers structure
                
                # Add prefix variations
                for p in ["text_model.", "model.", "transformer.", "encoder."]:
                    for k in [key, clean_key]:
                        if not k.startswith(p):
                            keys_to_try.append(f"{p}{k}")
                
                # Exhaustive search for matching key
                for pk in keys_to_try:
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
                    # Force float32 for dequantized weights on CPU to avoid SDPA dtype mismatches
                    target_dtype = torch.float32 if target_device.type == "cpu" else None
                    weight = dequantizer(weight, dtype=target_dtype)
                
                # Resolve the target parameter/member and its shape
                param = getattr(mod, param_name, None)
                expected_shape = param.shape if param is not None else getattr(mod, "tensor_shape", None)
                
                # Convert FP8 weights to QuantizedTensor for native FP8 storage
                is_fp8 = hasattr(weight, "dtype") and str(weight.dtype) == "torch.float8_e4m3fn"
                if is_fp8:
                    base_path = key.rsplit(".", 1)[0]
                    scale = scales.get(base_path, None)
                    
                    if scale is not None:
                        # Wrap FP8 weight in QuantizedTensor with BlockScaledFP8Layout
                        # This keeps weights in FP8 format (~8GB vs ~32GB dequantized)
                        # Dequantization happens at inference time in the linear op
                        # Determine appropriate orig_dtype based on device
                        # CPU usually needs float32 for compatibility/performance
                        orig_dtype = torch.float32 if target_device.type == "cpu" else torch.bfloat16
                        
                        params = BlockScaledFP8Layout.Params(
                            scale=scale,
                            orig_dtype=orig_dtype,
                            orig_shape=tuple(weight.shape)
                        )
                        weight = QuantizedTensor(weight, "BlockScaledFP8Layout", params)
                    else:
                        # FP8 without scale - fallback to device-appropriate dtype
                        target_dtype = torch.float32 if target_device.type == "cpu" else torch.bfloat16
                        print(f"[HYTextModel] WARNING: FP8 weight without scale: {key}")
                        weight = weight.to(target_dtype)
                
                # Refined backup shape detection for GGMLLayers
                if expected_shape is None:
                    if hasattr(mod, "out_features"):
                        if param_name == "bias": expected_shape = (mod.out_features,)
                        else: expected_shape = (mod.out_features, mod.in_features)
                    elif hasattr(mod, "num_embeddings"):
                        expected_shape = (mod.num_embeddings, mod.embedding_dim)

                if expected_shape is not None:
                    # Get actual shape - handle QuantizedTensor
                    if isinstance(weight, QuantizedTensor):
                        weight_shape = weight.params.orig_shape
                    else:
                        weight_shape = tuple(weight.shape)
                    
                    # Generic shape correction (squeeze/unsqueeze) - only for non-quantized
                    if not isinstance(weight, QuantizedTensor) and expected_shape != weight_shape:
                        if len(expected_shape) == 1 and len(weight_shape) == 2 and weight_shape[1] == 1:
                            weight = weight.squeeze(1)
                            weight_shape = tuple(weight.shape)
                    
                    # Convert expected_shape to tuple for comparison
                    expected_tuple = tuple(expected_shape)
                    
                    if expected_tuple == weight_shape:
                        if is_ggml_layer:
                            setattr(mod, param_name, torch.nn.Parameter(weight, requires_grad=False))
                        elif isinstance(weight, QuantizedTensor):
                            # For QuantizedTensor, set directly as Parameter
                            # set_module_tensor_to_device doesn't handle quantized tensors
                            setattr(mod, param_name, torch.nn.Parameter(weight, requires_grad=False))
                        else:
                            # For regular tensors on CPU, ensure float32 for norms/embeddings
                            if target_device.type == "cpu" and hasattr(weight, 'dtype') and weight.dtype in [torch.float16, torch.bfloat16]:
                                weight = weight.to(torch.float32)
                            set_module_tensor_to_device(model, target_key, target_device, value=weight)
                        loaded_count += 1
                    else:
                        missing_keys.append(f"{key} (shape mismatch: {weight_shape} vs {expected_tuple})")
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
        import time
        t_start = time.time()
        device = get_module_device(self.llm_text_encoder)
        target_device = get_module_device(self)
        
        # Check cache for single prompt (common case in ComfyUI)
        if len(text) == 1 and f"llm_{text[0]}" in self._cache:
            return self._cache[f"llm_{text[0]}"]

        print(f"[HYTextModel] LLM Encoding {len(text)} prompt(s) on {device}...")
        t_template_start = time.time()
        try:
            llm_text = [self.llm_tokenizer.apply_chat_template(
                [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {"role": "user", "content": t}],
                tokenize=False, add_generation_prompt=False
            ) for t in text]
        except Exception:
            llm_text = [f"<|im_start|>system\n{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n" for t in text]
        t_template_end = time.time()
        
        t_tokenize_start = time.time()
        enc = self.llm_tokenizer(
            llm_text, truncation=True, padding="max_length" if self.enable_llm_padding else "longest",
            max_length=self.max_length_llm, return_tensors="pt"
        ).to(device)
        t_tokenize_end = time.time()

        t_forward_start = time.time()
        with torch.no_grad():
            out = self.llm_text_encoder(**enc, output_hidden_states=True, return_dict=True)
        if device.type == "cuda": torch.cuda.synchronize()
        t_forward_end = time.time()
        
        t_post_start = time.time()
        ctxt_raw = out.hidden_states[-1] if hasattr(out, "hidden_states") else out.last_hidden_state
        if ctxt_raw.device != target_device: ctxt_raw = ctxt_raw.to(target_device)

        ctxt_raw = ctxt_raw[:, self.crop_start:self.max_length_llm].contiguous()
        ctxt_length = (enc["attention_mask"].sum(dim=-1).to(target_device) - self.crop_start).clamp(min=0, max=self._orig_max_length_llm)
        
        # Signal check (safely convert to float first)
        c_mean = ctxt_raw.mean().item()
        c_std = ctxt_raw.std().item()
        c_rms = ctxt_raw.pow(2).mean().sqrt().item()
        t_post_end = time.time()
        
        print(f"[HYTextModel] LLM stats: mean={c_mean:.4f}, std={c_std:.4f}, rms={c_rms:.4f}")
        print(f"[HYTextModel] LLM Timing: Template={t_template_end-t_template_start:.3f}s, Tokenize={t_tokenize_end-t_tokenize_start:.3f}s, Forward={t_forward_end-t_forward_start:.3f}s, Post={t_post_end-t_post_start:.3f}s, Total={time.time()-t_start:.3f}s")
        
        # Cache result if single prompt
        if len(text) == 1:
            if len(self._cache) >= self._cache_max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[f"llm_{text[0]}"] = (ctxt_raw, ctxt_length)
            
        return ctxt_raw, ctxt_length

    def encode_sentence_emb(self, text: List[str]) -> Tensor:
        device = get_module_device(self.sentence_emb_text_encoder)
        
        # Check cache
        if len(text) == 1 and f"clip_{text[0]}" in self._cache:
            return self._cache[f"clip_{text[0]}"]

        enc = self.sentence_emb_tokenizer(text, truncation=True, padding=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.sentence_emb_text_encoder(**enc)
        v_emb = out.pooler_output.unsqueeze(1) if hasattr(out, "pooler_output") and out.pooler_output is not None else self._encode_pooling(enc["attention_mask"], out.last_hidden_state)
        
        # Signal check
        v_mean = v_emb.mean().item()
        v_std = v_emb.std().item()
        v_rms = v_emb.pow(2).mean().sqrt().item()
        print(f"[HYTextModel] CLIP stats: mean={v_mean:.4f}, std={v_std:.4f}, rms={v_rms:.4f}")
        
        # Cache result
        if len(text) == 1:
            self._cache[f"clip_{text[0]}"] = v_emb
            
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
