"""General plugin opt-in; nạp riêng trong mỗi process vLLM."""

def register() -> None:
    import os
    if not os.environ.get('FAST_INFER_VLLM_RESEARCH_CONFIG'):
        return
    from .adapter import install
    install()
