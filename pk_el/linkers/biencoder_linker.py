import random
from collections import defaultdict

import numpy as np
from datasets import Dataset
from sentence_transformers import InputExample

from pk_el.evaluation import evaluate_retrieval, plot_confusion_matrix
from pk_el.linkers.representation_linkers import retrieve_entities_embeddings


def generate_input_examples(
    data,
    id_to_label,
    param_to_category,
    mention_key="mention_text",
    label_key="label",
    nil_label="Q100",
    include_random_neg=True,
    num_random_negs=1,
    include_hard_negatives=True,
    num_hard_negs=1,
    return_dataset=True,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    examples = []
    all_param_ids = list(id_to_label.keys())
    category_to_param_ids = defaultdict(list)

    for pid, cid in param_to_category.items():
        category_to_param_ids[cid].append(pid)

    for ex in data:
        mention = ex[mention_key]
        label_id = ex[label_key]

        # Case-insensitive NIL label check
        if str(label_id).strip().lower() == str(nil_label).strip().lower():
            continue  # Skip NILs

        # Positive example
        examples.append(InputExample(texts=[mention, id_to_label[label_id]], label=1.0))

        # Hard negatives: from same category
        if include_hard_negatives:
            category = param_to_category.get(label_id)
            hard_neg_candidates = [pid for pid in category_to_param_ids[category] if pid != label_id]
            sampled_hard_negs = random.sample(hard_neg_candidates, min(len(hard_neg_candidates), num_hard_negs))
            for hard_pid in sampled_hard_negs:
                examples.append(InputExample(texts=[mention, id_to_label[hard_pid]], label=0.0))

        # Random negatives: from full ontology
        if include_random_neg:
            rand_candidates = [pid for pid in all_param_ids if pid != label_id]
            sampled_rand_negs = random.sample(rand_candidates, min(len(rand_candidates), num_random_negs))
            for rand_pid in sampled_rand_negs:
                examples.append(InputExample(texts=[mention, id_to_label[rand_pid]], label=0.0))

    if return_dataset:
        return Dataset.from_dict({
            "query": [ex.texts[0] for ex in examples],
            "response": [ex.texts[1] for ex in examples],
            "label": [ex.label for ex in examples],
        })
    else:
        return examples


def evaluate_biencoder_el_by_source(
    dataset,
    param_to_id,
    model,
    retrieval_feature_name,
    k,
    match_threshold,
    label_mapping,
    use_category_constraint=True,
    print_results=True,
    plot_conf_matrix=True,
):
    """
    Evaluate retrieval performance by source, and plot a single combined confusion matrix.
    """
    # Split dataset by 'source'
    datasets_by_source = defaultdict(list)
    for ex in dataset:
        source = ex.get("source", "unknown")
        datasets_by_source[source].append(ex)

    eval_summaries = {}

    # Collect all predictions and labels for combined plotting
    all_true = []
    all_pred = []

    for source, subset in datasets_by_source.items():
        print(f"Evaluating source: {source}")
        retrieved = retrieve_entities_embeddings(
            dataset=subset,
            param_to_id=param_to_id,
            model=model,
            k=k,
            match_threshold=match_threshold,
            retrieval_feature_name=retrieval_feature_name,
            use_category_constraint=use_category_constraint
        )

        results = evaluate_retrieval(
            retrieved["retrieval_predictions"],
            retrieved["top_predictions"],
            retrieved["true_labels"],
            label_mapping=label_mapping,
            k=k,
            print_results=print_results
        )

        # Accumulate for combined confusion matrix
        all_true.extend(retrieved["true_labels"])
        all_pred.extend(retrieved["top_predictions"])

        # Metric calculations
        mrr = float(results["MRR (%)"].replace('%', ''))
        recall = float(results[f"Recall@{k} (%)"].replace('%', ''))
        micro_f1 = float(results["Scores of Top Retrieved"]["mic_F1"].replace('%', ''))

        nil_count = sum(1 for pred in retrieved["top_predictions"] if pred == "Q100")
        nil_prop = nil_count / len(retrieved["top_predictions"]) if retrieved["top_predictions"] else 0.0

        eval_summaries[source] = {
            "source": source,
            "MRR": mrr,
            f"recall@k": recall,
            f"micro_f1": micro_f1,
            "nil_count": nil_count,
            "nil_prop": round(nil_prop * 100, 2),
            "n": len(subset)
        }

    # Plot combined confusion matrix
    if plot_conf_matrix:
        print("\nCombined confusion matrix:")
        plot_confusion_matrix(
            all_true,
            all_pred,
            label_mapping=label_mapping,
            normalize=True
        )

    return eval_summaries

