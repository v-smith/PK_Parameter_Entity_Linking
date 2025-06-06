from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix
)


def split_matched_and_unmatched(dataset: List[dict]) -> tuple:
    matched = []
    unmatched = []
    for ex in dataset:
        if ex.get("subset_matched", False):
            matched.append(ex)
        else:
            ex_copy = ex.copy()
            ex_copy["llm_pred"] = "Q100"  # Assign NIL
            unmatched.append(ex_copy)
    return matched, unmatched


def evaluate(y_true, y_pred, id_to_label=None, print_results=True):
    """
    Compute micro precision, recall, F1, and accuracy for multiclass predictions.
    """
    if id_to_label:
        y_true = [id_to_label.get(y, y) for y in y_true]
        y_pred = [id_to_label.get(y, y) for y in y_pred]

    metrics = {
        "mic_P": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "mic_R": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "mic_F1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "mac_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    metrics = {k: f"{v * 100:.2f}%" for k, v in metrics.items()}

    if print_results:
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    return metrics


def evaluate_retrieval(retrieval_results, top_predictions, gold_labels, label_mapping=None, k=10, print_results=True):
    """
    Compute MRR and Recall@K for retrieval and evaluate top-1 predictions.
    """
    assert len(retrieval_results) == len(gold_labels)

    total = len(gold_labels)
    reciprocal_ranks = []
    recall_hits = []

    for preds, gold in zip(retrieval_results, gold_labels):
        pred_ids = [x["param_id"] for x in preds]
        if gold in pred_ids:
            rank = pred_ids.index(gold) + 1
            reciprocal_ranks.append(1 / rank)
            recall_hits.append(1)
        else:
            reciprocal_ranks.append(0)
            recall_hits.append(0)

    mrr = sum(reciprocal_ranks) / total
    recall_at_k = sum(recall_hits) / total
    scores_top = evaluate(top_predictions, gold_labels, label_mapping, print_results=False)

    results = {
        "MRR (%)": f"{mrr * 100:.2f}%",
        f"Recall@{k} (%)": f"{recall_at_k * 100:.2f}%",
        "Scores of Top Retrieved": scores_top,
    }

    if print_results:
        print("\nRetrieval Evaluation:")
        for k, v in results.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")

    return results


def print_classification_errors(results_dict, id_to_label=None, max_errors=100, y_true_label="y_true", y_pred_label="y_pred", texts_label="texts"):
    """
    Print and return misclassified examples.
    """
    y_true = results_dict[y_true_label]
    y_pred = results_dict[y_pred_label]
    texts = results_dict[texts_label]
    model_name = results_dict.get("model_name", "Model")

    print(f"\n🔍 Misclassifications from {model_name}:\n")
    errors_summary = []

    for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
        if true != pred:
            true_label = id_to_label.get(true, true) if id_to_label else true
            pred_label = id_to_label.get(pred, pred) if id_to_label else pred

            print(f" TEXT: {text}")
            print(f"✅ True: {true_label}")
            print(f"❌ Pred: {pred_label}\n")

            errors_summary.append({"text": text, "true": true, "pred": pred})
            if len(errors_summary) >= max_errors:
                break

    if not errors_summary:
        print("✅ No misclassifications found!")

    return errors_summary


def plot_confusion_matrix(y_true, y_pred, label_mapping=None, normalize=True, figsize=(10, 8), save_path=None):
    """
    Plot a heatmap-style confusion matrix like your reference image.
    """
    label_mapping["Q100"] = "NIL"

    # Optionally remap class labels
    if label_mapping:
        y_true = [label_mapping.get(y, y) for y in y_true]
        y_pred = [label_mapping.get(y, y) for y in y_pred]

    # Determine label set
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: (x == "NIL", x))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true' if normalize else None)

    # Optional: identify missing columns and mask them (make fully white)
    missing_preds = [i for i, col in enumerate((cm == 0).all(axis=0)) if col]
    missing_actuals = [i for i, row in enumerate((cm == 0).all(axis=1)) if row]
    mask = np.zeros_like(cm, dtype=bool)
    mask[:, missing_preds] = True
    mask[missing_actuals, :] = True

    plt.figure(figsize=figsize)
    sns.set_style(style="white")

    ax = sns.heatmap(
        cm,
        mask=mask,
        cmap="rocket_r",
        square=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        vmin=0, vmax=1  # ensure low values show as sand
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()











