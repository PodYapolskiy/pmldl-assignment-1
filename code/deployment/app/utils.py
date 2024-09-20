def markdown_progress(x: float) -> str:
    """
    Returns a bar from a number between 0 and 100.
    """
    x = round(x * 100)
    return f"""![](https://geps.dev/progress/{x})"""
