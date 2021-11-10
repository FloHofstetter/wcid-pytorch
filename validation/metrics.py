import torch
from typing import Dict


def true_negative(prediction: torch.Tensor, label: torch.Tensor) -> float:
    """
    Return sum of all true negative pixel.

    :param prediction: Predicted tensor.
    :param label: Label tensor.
    :return: True negative count.
    """
    return torch.sum(~prediction & ~label).item()


def true_positive(prediction: torch.Tensor, label: torch.Tensor) -> float:
    """
    Return sum of all true positive pixel.

    :param prediction: Predicted tensor.
    :param label: Label tensor.
    :return: True positive count.
    """
    return torch.sum(prediction & label).item()


def false_negative(prediction: torch.Tensor, label: torch.Tensor) -> float:
    """
    Return sum of all false negative.

    :param prediction: Predicted tensor.
    :param label: Label tensor.
    :return: False negative count
    """
    return torch.sum(~prediction & label).item()


def false_positive(prediction: torch.Tensor, label: torch.Tensor) -> float:
    """
    Return sum of all false positive.

    :param prediction: Predicted tensor.
    :param label: Label tensor.
    :return: False positive count.
    """
    return torch.sum(prediction & ~label).item()


def iou_score(fn: float, fp: float, tp: float) -> float:
    """
    Calculate intersection over union.

    :param fn: False negative count.
    :param fp: False positive count.
    :param tp: True positive count.
    :return: IoU score.
    """
    return 0.0 if fp + tp + fn == 0 else tp / (fp + tp + fn)


def f1_score(fn: float, fp: float, tp: float) -> float:
    """
    Calculate F1 score.

    :param fn: False negative count.
    :param fp: False positive count.
    :param tp: True positive count.
    :return: F1 score.
    """
    return 0.0 if 2 * tp + fp + fn == 0 else (2 * tp) / ((2 * tp) + fp + fn)


def accuracy(tp: float, fn: float, tn: float, fp: float) -> float:
    """
    Calculate accuracy.

    :param tp: True positive count.
    :param fn: False negative count.
    :param tn: True negative count.
    :param fp: False positive count.
    :return: Accuracy.
    """
    return 0.0 if tp + tn + fp + fn == 0 else (tp + tn) / (tp + tn + fp + fn)


def precision(tp: float, fn: float) -> float:
    """
    Calculate precision.

    :param tp: True positive count.
    :param fn: False negative count.
    :return: Precision.
    """
    return 0.0 if tp + fn == 0.0 else tp / (tp + fn)


def recall(fn: float, tp: float) -> float:
    """
    Calculate recall.

    :param fn: False negative count.
    :param tp: True positive count.
    :return: Recall.
    """
    return 0 if tp + fn == 0 else fn / (tp + fn)


def calculate_metrics(
    prediction: torch.Tensor, label: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate  intersection over union, f1 score, accuracy, precision
    and recall. Return values as dict.

    :param prediction: Predicted tensor.
    :param label: Label tensor.
    :return: Dict with iou, f1, accuracy (acc), precision(prc) and recall (rec).
    """
    prediction = prediction.to(torch.bool)
    label = label.to(torch.bool)
    fp = false_positive(prediction, label)
    fn = false_negative(prediction, label)
    tp = true_positive(prediction, label)
    tn = true_negative(prediction, label)
    metrics = {
        "iou": iou_score(fn=fn, fp=fp, tp=tp),
        "f1": f1_score(fn=fn, fp=fp, tp=tp),
        "acc": accuracy(tp=tp, fn=fn, tn=tn, fp=fp),
        "pre": precision(tp=tp, fn=fn),
        "rec": recall(fn=fn, tp=tp),
    }
    return metrics
