def iou(
        a1: int,
        b1: int,
        a2: int,
        b2: int
) -> float:
    """
    Compute the intersection over union of two character range
    """
    intersection = max(0, min(b1, b2) - max(a1, a2))
    union = (b1 - a1) + (b2 - a2) - intersection

    if union > 0:
        return intersection / union
    else:
        return 0.0


# if __name__ == "__main__":
#     print(iou(0, 100, 50, 150))
