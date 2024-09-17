import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def markdown_progress(x: float) -> str:
    """
    Returns a bar from a number between 0 and 100.
    """
    x = round(x * 100)
    return f"""![](https://geps.dev/progress/{x})"""
