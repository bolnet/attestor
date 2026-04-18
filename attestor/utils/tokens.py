"""Token estimation utilities."""


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string. Approximation: ~1.3 tokens per word."""
    return int(len(text.split()) * 1.3)
