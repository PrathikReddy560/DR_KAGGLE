"""Classification, calibration, and selective-prediction reports for both stages."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _as_builtin(value):
    return value.item() if isinstance(value, np.generic) else value


def classification_summary(y_true, y_pred, class_names):
    """Return accuracy, precision, recall, F1, confusion matrix, and per-class accuracy."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(len(class_names)))
    if y_true.size == 0:
        return {
            "available": False,
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_precision": None,
            "weighted_recall": None,
            "weighted_f1": None,
            "per_class": {},
            "confusion_matrix": np.zeros((len(labels), len(labels)), dtype=int).tolist(),
        }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "available": True,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                # For a single-label classifier, per-class accuracy is recall.
                "per_class_accuracy": float(recall[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "sklearn_classification_report": {
            key: {metric: _as_builtin(value) for metric, value in values.items()}
            if isinstance(values, dict) else _as_builtin(values)
            for key, values in classification_report(
                y_true, y_pred, labels=labels, target_names=list(class_names),
                zero_division=0, output_dict=True,
            ).items()
        },
    }


def save_confusion_matrix(matrix, class_names, output_path, title):
    """Write a labelled confusion matrix even when no samples were accepted."""
    matrix = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(max(6, len(class_names) * 1.35),) * 2)
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    labels = list(range(len(class_names)))
    axis.set(
        xticks=labels,
        yticks=labels,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted grade",
        ylabel="True grade",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column, row, str(matrix[row, column]), ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_classification_report(y_true, y_pred, class_names, output_dir, prefix):
    """Save the standard report used by Stage 1 and deterministic evaluations."""
    report = classification_summary(y_true, y_pred, class_names)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    save_confusion_matrix(
        report["confusion_matrix"], class_names,
        os.path.join(output_dir, f"{prefix}_confusion_matrix.png"),
        f"{prefix.replace('_', ' ').title()} confusion matrix",
    )
    return report


def expected_calibration_error(confidence, correct, bins=15):
    """ECE with fixed-width confidence bins, as specified in the report."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence >= lower) & ((confidence < upper) if upper < 1 else (confidence <= upper))
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def _binary_dr_metrics(y_true, predictions, probabilities):
    labels = (np.asarray(y_true) > 0).astype(int)
    binary_predictions = (np.asarray(predictions) > 0).astype(int)
    scores = np.asarray(probabilities)[:, 1:].sum(axis=1)
    result = {
        "sensitivity": None,
        "specificity": None,
        "binary_dr_auc": None,
    }
    if labels.size == 0:
        return result
    matrix = confusion_matrix(labels, binary_predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result["sensitivity"] = float(tp / (tp + fn)) if tp + fn else None
    result["specificity"] = float(tn / (tn + fp)) if tn + fp else None
    if len(np.unique(labels)) == 2:
        result["binary_dr_auc"] = float(roc_auc_score(labels, scores))
    return result


def save_uncertainty_evaluation(
    y_true,
    predictions,
    probabilities,
    predictive_variance,
    accepted,
    class_names,
    output_dir,
    prefix,
    confidence_threshold,
    gate1_pass=None,
):
    """Save the report's Method-A/Method-B comparison metrics.

    Accuracy, precision, recall, F1, and confusion matrix are reported for
    accepted predictions; calibration and binary DR AUC are retained on all
    test samples so rejection cannot inflate those quantities.
    """
    y_true = np.asarray(y_true, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictive_variance = np.asarray(predictive_variance, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    confidence = probabilities.max(axis=1)
    uncertainty = predictive_variance.mean(axis=1)
    all_summary = classification_summary(y_true, predictions, class_names)
    accepted_summary = classification_summary(y_true[accepted], predictions[accepted], class_names)
    report = {
        "confidence_threshold": float(confidence_threshold),
        "coverage": float(accepted.mean()),
        "rejection_rate": float((~accepted).mean()),
        "accepted_predictions": accepted_summary,
        "all_predictions": all_summary,
        "ece_all": expected_calibration_error(confidence, predictions == y_true),
        "ece_accepted": expected_calibration_error(confidence[accepted], (predictions == y_true)[accepted]),
        "predictive_uncertainty_mean": float(uncertainty.mean()),
        "predictive_uncertainty_accepted_mean": (
            float(uncertainty[accepted].mean()) if accepted.any() else None
        ),
        "binary_dr": _binary_dr_metrics(y_true, predictions, probabilities),
    }
    if gate1_pass is not None:
        report["ganomaly_gate1_pass_rate"] = float(np.asarray(gate1_pass, dtype=bool).mean())
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    save_confusion_matrix(
        accepted_summary["confusion_matrix"], class_names,
        os.path.join(output_dir, f"{prefix}_accepted_confusion_matrix.png"),
        f"{prefix.replace('_', ' ').title()} accepted-prediction confusion matrix",
    )
    return report
