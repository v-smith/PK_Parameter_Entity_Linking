from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def find_top_k_embeddings(input_mention, model, ontology_texts, ontology_embeddings, k: int = 10):
    """
    Returns the top-k matches using sentence-level embeddings (e.g., from BERT).

    Args:
        input_mention (str): The mention to match.
        model (SentenceTransformer): Preloaded embedding model.
        ontology_texts (list of str): Ontology entries (raw text).
        ontology_embeddings (ndarray): Precomputed dense vectors for ontology_texts.
        k (int): Number of top matches to return.

    Returns:
        list[dict]: List of top-k matches with score and matched string.
    """
    input_embedding = model.encode([input_mention], normalize_embeddings=True)
    similarities = cosine_similarity(input_embedding, ontology_embeddings).flatten()

    top_k_indices = similarities.argsort()[::-1][:k]

    results = []
    for idx in top_k_indices:
        results.append({
            "matched_mention": ontology_texts[idx],
            "score": similarities[idx],
            "method": "sentence_embedding"
        })

    return results


def retrieve_entities_embeddings(
    dataset,
    param_to_id,
    model,
    retrieval_feature_name: str = 'text_with_tagged_mention',
    k: int = 10,
    match_threshold: float = 0.80,
    use_category_constraint: bool = True,
):
    print("Retrieving...")
    results_dict = {
        "retrieval_predictions": [],
        "top_predictions": [],
        "true_labels": [],
        "not_matched": [],
        "incorrect_predictions": [],
    }

    ontology_text_to_id = {text: param_id for text, param_id in param_to_id.items()}
    ontology_texts = list(ontology_text_to_id.keys())
    ontology_embeddings = model.encode(ontology_texts, normalize_embeddings=True)

    for sample in tqdm(dataset):
        span_start = sample["spans"][0]["start"]
        span_end = sample["spans"][0]["end"]
        mention = sample["text"][span_start:span_end]
        mention_retrieval_text = sample[retrieval_feature_name]

        if mention_retrieval_text is None:
            print(sample["text"])
            continue

        true_label = sample["label"]
        allowed_concepts = set(sample.get("subsetted_concepts", [])) if use_category_constraint else None

        top_k_retrievals = find_top_k_embeddings(
            mention_retrieval_text, model, ontology_texts, ontology_embeddings, k=k
        )

        # Convert retrieved text → ID
        retrieved_ids = [
            {
                "param_id": ontology_text_to_id.get(entry["matched_mention"], "Q100"),
                "score": entry["score"]
            }
            for entry in top_k_retrievals
        ]

        # Filter by score and (optionally) category constraint
        filtered_retrieved_ids = [
            r for r in retrieved_ids
            if r["score"] >= match_threshold and (
                not use_category_constraint or r["param_id"] in allowed_concepts
            )
        ]

        if not filtered_retrieved_ids:
            filtered_retrieved_ids = [{"param_id": "Q100", "score": 0.0}]

        results_dict["retrieval_predictions"].append(filtered_retrieved_ids)
        results_dict["true_labels"].append(true_label)

        top_pred = filtered_retrieved_ids[0]["param_id"]
        results_dict["top_predictions"].append(top_pred)

        if top_pred != true_label:
            if top_pred == "Q100":
                results_dict["not_matched"].append({
                    "mention": mention,
                    "text": sample["text"],
                    "true_label": true_label,
                })
            else:
                results_dict["incorrect_predictions"].append({
                    "mention": mention,
                    "text": sample["text"],
                    "predicted_label": top_pred,
                    "true_label": true_label,
                })

    return results_dict

