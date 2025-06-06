import re
from typing import List

from tqdm import tqdm

from pk_el.evaluation import evaluate, print_classification_errors
from pk_el.tokenizers.basic_tokenizer import basic_tokenizer, nltk_tokenizer, basic_preprocessing
from pk_el.tokenizers.ontology_subset_patterns import COMPILED_PATTERN_TO_CATEGORIES, COMPILED_PATTERNS_TO_PARAMETERS, \
    CATEGORY_PRIORITIES
from pk_el.tokenizers.patterns import RATE_CONSTANT_UNIT_REGEX, ENZYME_CONTEXT_REGEX, \
    T_HALF_BETA_CONTEXT_REGEX, T_HALF_GAMMA_CONTEXT_REGEX, T_HALF_Z_CONTEXT_REGEX, NIL_PATTERNS_COMPILED
from pk_el.tokenizers.pk_tokenizer import pk_tokenizer
from pk_el.tokenizers.scispacy_tokenizer import scispacy_tokenizer
from pk_el.utils import write_jsonl


def load_tokenizer(tokenizer_name: str):
    tokenizer_name = tokenizer_name.lower()

    tokenizers = {
        "basic": basic_tokenizer,
        "pk": pk_tokenizer,
        "scispacy": scispacy_tokenizer,
        "nltk": nltk_tokenizer,
    }

    if tokenizer_name not in tokenizers:
        raise NameError(
            f"Unknown tokenizer name '{tokenizer_name}'.\n"
            "Tokenizer name must be one of: 'basic', 'pk', 'scispacy', or 'biobert'."
        )

    return tokenizers[tokenizer_name]


def tokenize_data(data_list: List[dict], tokenizer_name):
    """
    Tokenizes the 'text' value in each dictionary in the list and stores the result under 'tokens'.

    :param data_list: List of dictionaries, each containing a 'text' key.
    :param tokenizer: Tokenizer function that takes a string and returns a list of tokens.
    :return: The modified list with tokenized text.
    """
    print("Tokenizing mentions...")
    tokenizer = load_tokenizer(tokenizer_name)
    for item in tqdm(data_list):
        span_start = item["spans"][0]["start"]
        span_end = item["spans"][0]["end"]
        mention = item["text"][span_start:span_end]
        item["tokens"] = tokenizer(mention)
    return data_list

def create_tokenized_param_names_and_synonyms_to_ids(ontology_df, tokenizer_name):
    print("Tokenizing parameter variants...")
    tokenizer = load_tokenizer(tokenizer_name)
    tokenized_param_to_ids = {}
    count = 0

    for _, row in ontology_df.iterrows():
        param_id = row["parameter_id"]
        param_name = row["parameter_name"]
        synonyms = row.get("parameter_synonyms", [])

        # Make a list of all variants for this parameter
        variants = [param_name]
        if isinstance(synonyms, list):
            variants.extend([syn for syn in synonyms if syn])

        for name in variants:
            tokenized_key = frozenset(sorted(tokenizer(name)))
            count += 1

            if tokenized_key in tokenized_param_to_ids:
                if param_id not in tokenized_param_to_ids[tokenized_key]:
                    tokenized_param_to_ids[tokenized_key].append(param_id)
                    print(f"⚠️  Ambiguous tokenized key: {tokenized_key}. Adding another param_id: {param_id}")
            else:
                tokenized_param_to_ids[tokenized_key] = [param_id]

    print(f"Processed {count} parameter variants.")
    print(f"Tokenized parameter variants (unique token sets): {len(tokenized_param_to_ids)}")
    return tokenized_param_to_ids

def find_exact_params(tokenized_mention, tokenized_param_to_id: dict, table_mention:bool):
    """
    Matches mention tokens against a dictionary using exact match.
    Now supports multiple param IDs per token set.

    Args:
        tokenized_mention (list): List of tokens from the mention.
        tokenized_param_to_id (dict): Mapping from token frozensets to lists of parameter IDs.
        default_map (dict, optional): Mapping from mention strings or frozenset to default param_id.

    Returns:
        dict or None: Matched parameter(s) with method and optional ambiguity flag.
    """
    mention_set = set(tokenized_mention)
    full_mention_fset = frozenset(mention_set)

    if full_mention_fset in tokenized_param_to_id:
        candidates = tokenized_param_to_id[full_mention_fset]

        if len(candidates) == 1:
            return {"param_id": candidates[0]}

        # Try to resolve with default map (optional)
        if table_mention:
            if "Q57" in candidates:
                return {"param_id": "Q57"}

        # Ambiguous match — return all candidates
        return {
            "param_ids": candidates,
            "ambiguous": True
        }

    return None  # No exact match found


def is_non_pk_mention(text: str) -> bool:
    text = text.lower()
    return any(p.search(text) for p in NIL_PATTERNS_COMPILED)


def link_mentions_exact(tokenized_dataset, tokenized_param_to_id, table_mention: bool, text_key="text_with_tagged_mention"):
    results_dict = {
        "y_true": [],
        "y_pred": [],
        "texts": [],
        "unlinked_samples": [],
        "unlinked_results": {"y_true": [], "y_pred": [], "texts": []},
    }

    returned_multiple = 0
    exclusion_matches = 0
    disambiguation_matches = 0

    print("Linking mentions...")
    for sample in tqdm(tokenized_dataset):
        mention_text = sample["mention"]
        full_text = sample["text"]
        span_start = sample["spans"][0]["start"]
        span_end = sample["spans"][0]["end"]

        # Check for matches to exclusion terms
        if is_non_pk_mention(mention_text) or is_in_invalid_context(full_text, span_start, span_end):
            #print(f"Invalid mention: {sample['text_with_tagged_mention']}")
            results_dict["y_pred"].append("Q100")
            results_dict["y_true"].append(sample["label"])
            results_dict["texts"].append(sample[text_key])
            exclusion_matches += 1
            continue

        # find param matches
        tokens = sample["tokens"]
        result = find_exact_params(tokens, tokenized_param_to_id, table_mention=table_mention)
        if result:
            predicted_label = "Q100"
            # Unambiguous
            if "param_id" in result:
                predicted_label = result["param_id"]

            # Ambiguous → try to disambiguate
            elif "param_ids" in result:
                disamb_result = disambiguate_multiple(sample, result)
                if disamb_result and "param_id" in disamb_result:
                    predicted_label = disamb_result["param_id"]
                    disambiguation_matches += 1
                elif disamb_result and "param_ids" in disamb_result:
                    sample["potential_candidates"] = disamb_result["param_ids"]
                    results_dict["unlinked_samples"].append(sample)
                    returned_multiple += 1
                    continue
                else:
                    # No disambiguation matches
                    sample["potential_candidates"] = result["param_ids"]
                    results_dict["unlinked_samples"].append(sample)
                    results_dict["unlinked_results"]["y_true"].append(sample["label"])
                    results_dict["unlinked_results"]["texts"].append(sample[text_key])
                    results_dict["unlinked_results"]["y_pred"].append("Q100")
                    returned_multiple += 1
                    continue

            results_dict["y_pred"].append(predicted_label)
            results_dict["y_true"].append(sample["label"])
            results_dict["texts"].append(sample[text_key])
        else:
            results_dict["unlinked_samples"].append(sample)
            results_dict["unlinked_results"]["y_true"].append(sample["label"])
            results_dict["unlinked_results"]["texts"].append(sample[text_key])
            results_dict["unlinked_results"]["y_pred"].append("Q100")

    print(f"Number of exclusions identified: {exclusion_matches}, {exclusion_matches/len(tokenized_dataset):.2f}%")
    print(f"Number of disambiguation matches: {disambiguation_matches}, {disambiguation_matches/len(tokenized_dataset):.2f}%")
    print(f"Number with multiple labels: {returned_multiple}, {returned_multiple/len(tokenized_dataset):.2f}%")

    return results_dict


def is_in_invalid_context(full_text, span_start, span_end, window=15):
    """
    Checks if the mention appears in an invalid context by examining surrounding text.
    Only checks a small window around the mention.
    """
    full_text = full_text.lower()
    mention_text = full_text[span_start:span_end]
    context = full_text[max(0, span_start - window): span_end + window].lower()
    mention = re.escape(mention_text.lower())  # ESCAPE HERE

    # Define invalid context patterns
    invalid_patterns = [
        rf"{mention}[\s\-_/:]*(mic|pharmacodynamic)", # AUC/MIC
        rf"{mention}[\s\-_/:]*(creatinine|cr)", # CLCR
        rf"{mention}/fe", # e.g.  F/FE (food/formulation effect)
        rf"m\s*\+\s*{mention}" # e.g. M + F
    ]

    for pattern in invalid_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def disambiguate_multiple(sample, candidate_ids):
    """
    Attempt to resolve ambiguity between candidate param_ids using sample context
    for "km" and "t1/2". Returns a param_id only if exactly one regex matches.

    Args:
        sample (dict): A tokenized sample with keys like "tokens", "text", etc.
        candidate_ids (dict): Dict with key "param_ids" → list of candidate param IDs.

    Returns:
        dict or None: Dict with "param_id" if resolved, else None if ambiguous or unresolved.
    """
    text = basic_preprocessing(sample["text"])
    param_ids = candidate_ids["param_ids"]

    a = 1
    # --- Disambiguate km (Q1 vs Q51) --- #
    if "Q1" in param_ids:
        km_matches = []
        if RATE_CONSTANT_UNIT_REGEX.search(text):
            km_matches.append("Q51")
        if ENZYME_CONTEXT_REGEX.search(text):
            km_matches.append("Q1")
        if len(set(km_matches)) == 1:
            return {"param_id": km_matches[0]}
        elif len(set(km_matches)) > 1:
            return {"param_ids": km_matches} # Ambiguous match
        else:
            return {"param_ids": "Q1"} # No match - return most likely

    # --- Disambiguate t1/2 variants (Q57 vs Q60 vs Q89) --- #
    if "Q57" in param_ids:
        t_half_matches = []
        if T_HALF_Z_CONTEXT_REGEX.search(text):
            t_half_matches.append("Q57")
        if T_HALF_BETA_CONTEXT_REGEX.search(text):
            t_half_matches.append("Q60")
        if T_HALF_GAMMA_CONTEXT_REGEX.search(text):
            t_half_matches.append("Q89")
        if len(set(t_half_matches)) == 1:
            return {"param_id": t_half_matches[0]}
        elif len(set(t_half_matches)) > 1:
            return {"param_ids": t_half_matches} # Ambiguous match
        else:
            return {"param_id": "Q57"}  # No match - return most likely

    return None


def match_categories(pk_tokens: str, mention: str) -> List[str]:
    matched = []
    for pattern, cats in COMPILED_PATTERN_TO_CATEGORIES.items():
        if re.search(pattern, pk_tokens):
            matched.extend(cats)
    if not matched:
        for pattern, cats in COMPILED_PATTERN_TO_CATEGORIES.items():
            if re.search(pattern, mention):
                matched.extend(cats)
    return matched

def match_parameters(pk_tokens: str, mention: str) -> List[str]:
    matched = []
    for pattern, params in COMPILED_PATTERNS_TO_PARAMETERS.items():
        if re.search(pattern, pk_tokens):
            matched.extend(params)
    if not matched:
        for pattern, params in COMPILED_PATTERNS_TO_PARAMETERS.items():
            if re.search(pattern, mention):
                matched.extend(params)
    return matched

def apply_category_priority(categories: List[str]) -> List[str]:
    category_set = set(categories)
    for tier in CATEGORY_PRIORITIES:
        tier_matches = category_set & tier
        if tier_matches:
            return list(tier_matches)
    return list(category_set)


def evaluate_and_log(label, dataset, results, id_to_label, error_path=None):
    print(f"Scores {label}:")
    percentage = (len(results['y_pred']) / len(dataset)) * 100
    print(f"Percentage of Dataset Matched: {percentage:.2f}%")
    print(evaluate(results["y_true"], results["y_pred"], id_to_label=id_to_label))
    errors = print_classification_errors(results, id_to_label=id_to_label)
    if error_path:
        write_jsonl(error_path, errors)