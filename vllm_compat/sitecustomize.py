"""Compatibility shim loaded only by the Phase 3 vLLM service.

The server driver advertises CUDA 12.5 while the installed FlashAttention PTX
was built by a newer CUDA toolchain. Qwen3-VL otherwise auto-selects that
kernel for its vision encoder even when TORCH_SDPA is requested. Returning
False here keeps the vision path on PyTorch SDPA without modifying vLLM or the
shared Python installation.
"""

try:
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as transformers_import_utils

    def _flash_attention_disabled() -> bool:
        return False

    transformers_utils.is_flash_attn_2_available = _flash_attention_disabled
    transformers_import_utils.is_flash_attn_2_available = _flash_attention_disabled
except Exception:
    # vLLM will still report a clear startup/runtime error if transformers is
    # unavailable, so this shim must not hide the original import exception.
    pass
