"""Optional runtime dependencies used only for terminal formatting."""

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs):
        return text


__all__ = ["colored"]
