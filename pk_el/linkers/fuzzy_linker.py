from matplotlib import pyplot as plt
from rapidfuzz.fuzz import ratio
from tqdm import tqdm

from pk_el.evaluation import evaluate_retrieval, evaluate, print_classification_errors


def fuzzy_token_set_score(tokens1, tokens2, threshold=80):
    """Fuzzy matching score between tokens."""
    matched = 0
    for t1 in tokens1:
        best = max([ratio(t1, t2) for t2 in tokens2], default=0)
        if best >= threshold:
            matched += 1
    return matched / max(len(tokens1), len(tokens2)) * 100


def fuzzy_string_score(str1, str2):
    """String-level fuzzy match score."""
    return ratio(str1, str2)


def find_top_k_fuzzy_params_tokens(tokens, param_index, k=10, matching_mode="token", threshold=80):
    """
    Finds top-k matches based on fuzzy score between mention and parameter candidates.
    Supports token-level or string-level matching.

    Args:
        tokens (list or str): Token list or string (depending on mode).
        param_index (dict): {frozenset or str: List[param_id]}.
        matching_mode (str): "token" or "string".
        threshold (int): Only consider matches above this threshold.

    Returns:
        List[dict]: Top-k matches as {param_id, score}.
    """
    results = []

    for key, param_ids in param_index.items():
        if matching_mode == "token":
            score = fuzzy_token_set_score(set(tokens), key, threshold=threshold)
        elif matching_mode == "string":
            score = fuzzy_string_score(" ".join(tokens), " ".join(key))  # Both must be lists of tokens
        else:
            raise ValueError("matching_mode must be 'token' or 'string'")

        if score >= threshold:
            for param_id in param_ids:
                results.append({"param_id": param_id, "score": score})

    # Sort and deduplicate
    results.sort(key=lambda x: (-x["score"], x["param_id"]))
    seen = set()
    unique_results = []
    for item in results:
        if item["param_id"] not in seen:
            seen.add(item["param_id"])
            unique_results.append(item)
            if len(unique_results) == k:
                break

    return unique_results


def find_top_k_fuzzy_params(mention, param_index, k=10, threshold=80):
    """
    Finds top-k matches based on fuzzy score between mention and parameter candidates.
    Supports token-level or string-level matching.

    Args:
        mention (list or str): Token list or string (depending on mode).
        param_index (dict): {frozenset or str: List[param_id]}.
        matching_mode (str): "token" or "string".
        threshold (int): Only consider matches above this threshold.

    Returns:
        List[dict]: Top-k matches as {param_id, score}.
    """
    results = []

    for key, param_ids in param_index.items():
        score = fuzzy_token_set_score(set(mention), key, threshold=threshold)
        if score >= threshold:
            for param_id in param_ids:
                results.append({"param_id": param_id, "score": score})

    # Sort and deduplicate
    results.sort(key=lambda x: x["score"], reverse=True)
    seen = set()
    unique_results = []
    for item in results:
        if item["param_id"] not in seen:
            seen.add(item["param_id"])
            unique_results.append(item)
            if len(unique_results) == k:
                break

    return unique_results


def retrieve_entities_fuzzy(
    dataset,
    param_index,
    k=10,
    match_threshold=80,
    matching_mode="token",
    category_constrained=False,
):
    print(f"\nRetrieving using {matching_mode}-level fuzzy matching with threshold {match_threshold}...")
    if category_constrained:
        print("🔒 Category-constrained matching enabled.")

    results_dict = {
        "retrieval_predictions": [],
        "top_predictions": [],
        "true_labels": [],
        "texts": [],
        "matched_top_predictions": [],
        "matched_true_labels": [],
        "matched_texts": [],
        "unmatched_top_predictions": [],
        "unmatched_true_labels": [],
        "not_matched": [],
    }

    matched_count = 0

    for sample in tqdm(dataset):
        span_start = sample["spans"][0]["start"]
        span_end = sample["spans"][0]["end"]
        mention_text = sample["text"][span_start:span_end]
        tokens = sample["tokens"]
        true_label = sample["label"]

        # Constrain param_index for this example
        if category_constrained and "subsetted_concepts" in sample:
            constrained_param_index = {
                key: [pid for pid in pids if pid in sample["subsetted_concepts"]]
                for key, pids in param_index.items()
            }
            constrained_param_index = {k: v for k, v in constrained_param_index.items() if v}
        else:
            constrained_param_index = param_index

        # Match
        if matching_mode in ["token", "string"]:
            top_k = find_top_k_fuzzy_params_tokens(tokens, constrained_param_index, k=k,
                                               matching_mode=matching_mode, threshold=match_threshold)
        else:
            raise ValueError("matching_mode must be 'token', 'string', or 'original_string'")

        top_pred = "Q100"
        is_matched = False

        if top_k:
            top_score = top_k[0]["score"]
            if top_score >= match_threshold:
                top_pred = top_k[0]["param_id"]
                print(f"Top-pred: {mention_text}, pred: {top_pred}, score: {top_score}")
                matched_count += 1
                is_matched = True

        results_dict["retrieval_predictions"].append(top_k)
        results_dict["top_predictions"].append(top_pred)
        results_dict["true_labels"].append(true_label)
        results_dict["texts"].append(mention_text)

        if is_matched:
            results_dict["matched_top_predictions"].append(top_pred)
            results_dict["matched_true_labels"].append(true_label)
            results_dict["matched_texts"].append(mention_text)
        else:
            results_dict["unmatched_top_predictions"].append("Q100")
            results_dict["unmatched_true_labels"].append(true_label)

        if not is_matched:
            results_dict["not_matched"].append(sample)

    total = len(dataset)
    print(f"\n[Retrieval Summary]")
    print(f"Total mentions:             {total}")
    print(f"Matched (above threshold):  {matched_count}")
    print(f"Unmatched (Q100):           {total - matched_count}")

    results_dict["matched_count"] = matched_count
    return results_dict



def plot_f1_vs_matched(results, dataset_name="Dataset", score_key="micro_f1_all", save_path=None):
    """
    Plots micro-F1 vs. percentage of matched examples across thresholds.

    Args:
        results (list[dict]): Output of multiple retrieval runs (one per threshold).
        dataset_name (str): For plot title.
        save_path (str): If provided, saves the figure.
    """
    thresholds = [r["match_threshold"] for r in results]
    micro_f1 = [r[score_key] for r in results]
    matched_pct = [r["matched_percent"] for r in results]

    fig, ax1 = plt.subplots()

    color_f1 = 'tab:blue'
    ax1.set_xlabel("Match Threshold")
    ax1.set_ylabel("Micro-F1 (%)", color=color_f1)
    ax1.plot(thresholds, micro_f1, marker='o', color=color_f1, label="Micro-F1")
    ax1.tick_params(axis='y', labelcolor=color_f1)

    ax2 = ax1.twinx()  # second y-axis
    color_matched = 'tab:red'
    ax2.set_ylabel("Matched (%)", color=color_matched)
    ax2.plot(thresholds, matched_pct, marker='s', linestyle='--', color=color_matched, label="Matched %")
    ax2.tick_params(axis='y', labelcolor=color_matched)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()



def run_fuzzy_retrieval_eval(
    dataset,
    param_index,
    k,
    match_threshold,
    label_mapping,
    matching_mode="string",
    category_constrained: bool = False
):
    results = retrieve_entities_fuzzy(
        dataset,
        param_index,
        k=k,
        match_threshold=match_threshold,
        matching_mode=matching_mode,
        category_constrained=category_constrained
    )

    total = len(dataset)
    if total == 0:
        print("⚠️ Fuzzy retrieval skipped: empty dataset.")
        return {
            "k": k,
            "match_threshold": match_threshold,
            "MRR": 0.0,
            f"recall@k": 0.0,
            "micro_f1_all": 0.0,
            "micro_f1_matched": 0.0,
            "matched_results": {},
            "unmatched_results": {},
            "matched_percent": 0.0,
            "unmatched_examples": [],
            "matched_predictions": []
        }
    matched = results["matched_count"]
    matched_percent = (matched / total) * 100

    print("Combined Scores:")
    eval_scores_all = evaluate_retrieval(
        results["retrieval_predictions"],
        results["top_predictions"],
        results["true_labels"],
        label_mapping=label_mapping,
        k=k
    )

    print("\nMatched Scores:")
    if matched > 0:
        matched_results = evaluate(
            results["matched_true_labels"],
            results["matched_top_predictions"],
            id_to_label=label_mapping
        )
    else:
        matched_results = {}

    print("\nUnmatched Scores:")
    if total - matched > 0:
        unmatched_results = evaluate(
            results["unmatched_true_labels"],
            results["unmatched_top_predictions"],
            id_to_label=label_mapping
        )
    else:
        unmatched_results = {}

    print("\n\nMatched Error Analysis:")
    print_classification_errors(
        results,
        id_to_label=param_index,
        y_true_label="matched_true_labels",
        y_pred_label="matched_top_predictions",
        texts_label="matched_texts"
    )

    unmatched_indices = [i for i, pred in enumerate(results["top_predictions"]) if pred == "Q100"]
    unmatched_examples = [dataset[i] for i in unmatched_indices]

    # New: matched examples with true + predicted labels
    matched_preds = list(zip(results["matched_true_labels"], results["matched_top_predictions"]))

    return {
        "k": k,
        "match_threshold": match_threshold,
        "MRR": float(eval_scores_all["MRR (%)"].replace('%', '')),
        f"recall@k": float(eval_scores_all[f"Recall@{k} (%)"].replace('%', '')),
        "micro_f1_all": float(eval_scores_all["Scores of Top Retrieved"]["mic_F1"].replace('%', '')),
        "micro_f1_matched": float(matched_results.get("mic_F1", "0").replace('%', '')),
        "matched_results": matched_results,
        "unmatched_results": unmatched_results,
        "matched_percent": matched_percent,
        "unmatched_examples": unmatched_examples,
        "matched_predictions": matched_preds  # <-- New field: list of (true, pred)
    }


