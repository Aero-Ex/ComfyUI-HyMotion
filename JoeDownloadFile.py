"""
JoeDownloadFile Node - Register a file on the server and
expose a safe download_id + filename to the frontend UI.
The browser then calls a dedicated HTTP route to stream
the file, without ever seeing the raw server path.
"""

from pathlib import Path
from typing import Tuple
import os

import folder_paths
from .nodes_3d_viewer import register_download_path


class JoeDownloadFile:
    """
    Utility node that exposes a *download token* to frontend,
    not the real filesystem path.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "要下载的文件绝对路径（仅在服务端使用，不会直接暴露给前端）。"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "prepare_download"
    OUTPUT_NODE = True
    CATEGORY = "HY-Motion/Tools"

    def prepare_download(
        self,
        file_path: str,
    ) -> Tuple[str]:
        """
        Validate the file path and register it in the download registry.
        Frontend will receive (filename, download_id, exists) and should
        call /hymotion/download_file/{download_id} to actually download.
        """
        try:
            print("[JoeDownloadFile] Preparing file content for download UI...")

            if not file_path or not str(file_path).strip():
                raise ValueError("file_path 为空，请提供要下载的文件路径。")
            else:
                print(f"[JoeDownloadFile] file_path={file_path}")
            raw = str(file_path).strip().replace("\\", "/")
            
            input_base = Path(folder_paths.get_input_directory()).resolve()
            output_base = Path(folder_paths.get_output_directory()).resolve()

            print(f"[JoeDownloadFile] input_base={input_base}, output_base={output_base}")
            # 1) 先按原始路径解析（绝对或相对）
            p = Path(raw).expanduser().resolve()
            if not (p.exists() and p.is_file()):
                # 2) 如果不存在，尝试前面拼接 output_base
                p_output = (output_base / raw).resolve()
                if p_output.exists() and p_output.is_file():
                    p = p_output
                else:
                    # 3) 如果还不存在，再尝试前面拼接 input_base
                    p_input = (input_base / raw).resolve()
                    if p_input.exists() and p_input.is_file():
                        p = p_input

            exists = p.exists() and p.is_file()
            file_name = p.name if exists else p.name

            file_size = None
            if exists:
                try:
                    file_size = p.stat().st_size
                except OSError:
                    file_size = None

            download_id = None
            if exists:
                download_id = register_download_path(str(p))

            print(f"[JoeDownloadFile] file_path={p.absolute()}, exists={exists}, size={file_size}, download_id={download_id}")

            info = (
                "JoeDownloadFile Ready\n"
                f"File name: {file_name}\n"
                f"File path: {p.absolute()}\n"
                f"Exists on server: {exists}\n"
                f"File size (bytes): {file_size}\n"
                f"Download id: {download_id}\n"
            )

            # 只把文件名和 download_id 传给前端，不暴露真实路径
            return {
                "ui": {
                    "file_name": [file_name],
                    "download_id": [download_id],
                    "file_exists": [bool(exists)],
                    "file_size_bytes": [file_size],
                    "file_path": [str(p.absolute())],
                },
                "result": (info,),
            }

        except Exception as e:
            error_msg = f"JoeDownloadFile failed: {str(e)}"
            print(error_msg)
            return {
                "ui": {
                    "file_name": [""],
                    "download_id": [None],
                    "file_exists": [False],
                },
                "result": (error_msg,),
            }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-run when input changes so UI can update
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "JoeDownloadFile": JoeDownloadFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoeDownloadFile": "Joe Download File",
}

