
import ast
from typing import List

import pandas as pd

from pk_el.data_preprocessing import get_text_mention_feature
from pk_el.linkers.exact_linker import match_categories, match_parameters, apply_category_priority
from pk_el.tokenizers.basic_tokenizer import basic_preprocessing
from pk_el.tokenizers.pk_tokenizer import pk_tokenizer


def load_ontology(ontology_path, remove_nil=True):
    """
    Loads and preprocesses the pk_ontology CSV.

    - Normalizes synonym formatting
    - Handles NaNs
    - Optionally removes NIL entries

    Returns:
        ontology_df: Cleaned pandas DataFrame
    """
    df = pd.read_csv(ontology_path)

    # Normalize synonyms column safely
    df["parameter_synonyms"] = df["parameter_synonyms"].apply(normalize_ontology_synonyms)

    # Lowercase + strip text fields
    text_cols = ["parameter_name", "parameter_description", "units", "parameter_category", "category", "category_description"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    # Remove NIL if requested
    if remove_nil:
        df = df[df["parameter_id"] != "Q100"]
        df = df[df["category_id"] != "G100"]

    return df


def normalize_ontology_synonyms(val):
    """Normalise synonyms field from CSV."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            try:
                return ast.literal_eval(val)
            except Exception:
                return []
        else:
            # Assume comma-separated
            parts = [s.strip() for s in val.split(",") if s]
            return parts
    return []


def create_ontology_mappings(ontology_df, include_nil=False):
    """
    Creates mappings from an pk_ontology dataframe.

    Args:
        ontology_df (pd.DataFrame): Ontology data.

    Returns:
        dict: {
            'param_id_to_param_name': ...,
            'param_name_to_param_id': ...,
            'category_id_to_category_name': ...,
            'category_name_to_category_id': ...,
            'param_id_to_category_id': ...
        }
    """
    mappings = {
        "param_id_to_param_name": {},
        "param_name_to_param_id": {},
        "category_id_to_category_name": {},
        "category_name_to_category_id": {},
        "param_id_to_category_id": {},
    }

    for _, row in ontology_df.iterrows():
        param_id = row["parameter_id"]
        param_name = row["parameter_name"]
        category_id = row["category_id"]
        category_name = row.get("parameter_category", "")

        # Main param mappings
        if param_id and param_name:
            mappings["param_id_to_param_name"][param_id] = param_name
            mappings["param_name_to_param_id"][param_name] = param_id

        # Category mappings
        if category_id and category_name:
            mappings["category_id_to_category_name"][category_id] = category_name
            mappings["category_name_to_category_id"][category_name] = category_id

        # Param → Category
        if param_id and category_id:
            mappings["param_id_to_category_id"][param_id] = category_id

    if include_nil:
        mappings["param_id_to_param_name"]["Q100"] = "NIL"
        mappings["param_name_to_param_id"]["NIL"] = "Q100"

    return mappings


def format_ontology_for_llm(ontology_df):
    """
    Formats the pk_ontology nicely for LLM input.
    """
    formatted = []
    for _, entry in ontology_df.iterrows():
        param_name = entry.get('parameter_name', 'N/A')
        param_desc = entry.get('parameter_description', 'N/A')
        param_syns = entry.get('parameter_synonyms', [])
        param_units = entry.get('units', 'N/A')
        param_cat = entry.get('parameter_category', 'N/A')

        syns_formatted = " | ".join(param_syns) if isinstance(param_syns, list) else "N/A"

        formatted_entry = f"""[PARAM_NAME] {param_name}
        [DESC] {param_desc if param_desc else "N/A"}
        [SYN] {syns_formatted if syns_formatted else "N/A"}
        [UNIT] {param_units if param_units else "N/A"}
        [CATEGORY] {param_cat if param_cat else "N/A"}"""

        formatted.append(formatted_entry)

    return "\n\n".join(formatted)


def create_ontology_embedding_feature(row, include_description=True, include_units=True):
    # Extract and normalize values, handling NaNs
    param = row["parameter_name"] if pd.notna(row["parameter_name"]) else ""

    # Parse and clean synonyms
    synonyms = row["parameter_synonyms"]
    if pd.notna(synonyms):
        if isinstance(synonyms, str):
            try:
                synonyms_list = ast.literal_eval(synonyms)
            except (SyntaxError, ValueError):
                synonyms_list = [synonyms]
        elif isinstance(synonyms, list):
            synonyms_list = synonyms
        else:
            synonyms_list = []
        synonyms_clean = " | ".join(s.strip() for s in synonyms_list if isinstance(s, str))
    else:
        synonyms_clean = ""

    description = row["parameter_description"] if pd.notna(row["parameter_description"]) else ""
    units = row["units"] if pd.notna(row["units"]) else ""

    # Build the final string
    parts = [f"[PARAM] {param.strip()}"]
    if synonyms_clean:
        parts.append(f"[SYN] {synonyms_clean}")
    if include_description and description:
        parts.append(f"[DESC] {description.strip()}")
    if include_units and units:
        parts.append(f"[UNIT] {units.strip()}")

    return " ".join(parts)


def prepare_ontology_for_embedding(ontology_dir, include_description=True, include_units=True):
    ontology = pd.read_csv(ontology_dir)
    ontology['parameter_name'] = ontology['parameter_name'].fillna('').str.lower().str.strip()
    ontology = ontology[ontology["parameter_id"] != "Q100"]
    ontology["text_feature"] = ontology.apply(
        lambda row: create_ontology_embedding_feature(
            row,
            include_description=include_description,
            include_units=include_units
        ),
        axis=1
    )
    id_to_label = {row["parameter_id"]: row["text_feature"] for _, row in ontology.iterrows()}
    param_to_id = {row["text_feature"]: row["parameter_id"]  for _, row in ontology.iterrows()}
    return id_to_label, param_to_id


def add_ontology_subset_to_examples(
    examples: List[dict],
    ontology_df: pd.DataFrame,
    param_id_to_category_id: dict,
    use_filtered_subset: bool = True,
) -> List[dict]:
    """
    Adds pk_ontology subset and subsetted parameter IDs to each example.
    """
    print("Adding pk_ontology subset to examples...")

    # Create reverse map for category_id → list of parameter_ids
    category_to_param_ids = {}
    for param_id, cat_id in param_id_to_category_id.items():
        category_to_param_ids.setdefault(cat_id, []).append(param_id)

    for example in examples:
        if "mention" in example:
            mention = example["mention"]
        else:
            mention = get_text_mention_feature(example, special_tokens=False)

        mention_text = basic_preprocessing(mention)
        pk_tokens_string = " ".join(pk_tokenizer(mention_text))

        example["subset_matched"] = False
        example["subsetted_concepts"] = []  # initialize

        if use_filtered_subset:
            selected_categories = match_categories(pk_tokens_string, mention_text)
            selected_categories = apply_category_priority(selected_categories)
            selected_parameters = []

            if not selected_categories:
                selected_parameters = match_parameters(pk_tokens_string, mention_text)

            if selected_categories:
                # Convert categories to parameter IDs
                param_ids = []
                for cat_id in selected_categories:
                    param_ids.extend(category_to_param_ids.get(cat_id, []))

                subset_df = ontology_df[ontology_df["category_id"].isin(selected_categories)]
                example["subsetted_concepts"] = list(set(param_ids))  # deduplicate
                example["subset_matched"] = True

            elif selected_parameters:
                subset_df = ontology_df[ontology_df["parameter_id"].isin(selected_parameters)]
                example["subsetted_concepts"] = list(set(selected_parameters))
                example["subset_matched"] = True

            else:
                subset_df = ontology_df  # fallback
        else:
            subset_df = ontology_df  # no filtering

        example["ontology_subset"] = format_ontology_for_llm(subset_df)

    return examples


def evaluate_subset_matching(
    examples: List[dict],
    param_id_to_category_id: dict,
) -> dict:
    total = len(examples)
    correct_total = 0
    correct_matched = 0
    matched_count = 0
    correct_unmatched = 0
    unmatched_count = 0

    total_non_nil = 0
    correct_non_nil = 0
    total_nil = 0
    nil_with_subset = 0
    multi_category_count = 0
    subset_sizes = []

    for example in examples:
        label = example["label"]
        true_cat = param_id_to_category_id.get(label, "G100")
        is_nil = label == "Q100" or true_cat == "G100"

        if is_nil:
            total_nil += 1
        else:
            total_non_nil += 1

        subsetted = set(example.get("subsetted_concepts", []))

        if len(subsetted) > 1:
            subset_sizes.append(len(subsetted))
            subset_cats = {param_id_to_category_id.get(pid, "G100") for pid in subsetted}
            if len(subset_cats) > 1:
                multi_category_count += 1

        matched = len(subsetted) > 0
        correct = False

        if matched:
            if not is_nil and label in subsetted:
                correct = True
                correct_matched += 1
                correct_non_nil += 1
            elif is_nil:
                nil_with_subset += 1
        else:
            if is_nil:
                correct = True
                correct_unmatched += 1
            else:
                print(f"❌ UNMATCHED ERROR: {example.get('mention', '[missing mention]')}, True: {label}")

        if matched:
            matched_count += 1
        else:
            unmatched_count += 1

        if correct:
            correct_total += 1

    # Metrics
    overall_acc = correct_total / total
    matched_acc = correct_matched / matched_count if matched_count > 0 else 0.0
    unmatched_acc = correct_unmatched / unmatched_count if unmatched_count > 0 else 0.0
    subset_recall_non_nil = correct_non_nil / total_non_nil if total_non_nil > 0 else 0.0
    nil_fp_rate = nil_with_subset / total_nil if total_nil > 0 else 0.0
    avg_reduction = sum(1 - (size /len(param_id_to_category_id)) for size in subset_sizes) / len(subset_sizes) * 100

    # Report
    print("\n=== Subset Evaluation Summary ===")
    print(f"Percentage Matched:          {(matched_count / total):.2%}")
    print(f"Matched-only Accuracy:       {matched_acc:.2%}")
    print(f"Unmatched-only Accuracy:     {unmatched_acc:.2%}")
    print("\n")
    print(f"→ Subset recall (PK only):   {subset_recall_non_nil:.2%}")
    print(f"→ False Positive Rate (NIL): {nil_fp_rate:.2%}")
    print("\n")
    print(f"Examples with >1 category:    {multi_category_count}")
    print(f"Avg. reduction pk_ontology subset size:   {avg_reduction:.2f}")


    return {
        "overall_accuracy": overall_acc,
        "matched_accuracy": matched_acc,
        "unmatched_accuracy": unmatched_acc,
        "subset_recall_non_nil": subset_recall_non_nil,
        "nil_false_positive_rate": nil_fp_rate,
        "total": total,
        "matched": matched_count,
        "unmatched": unmatched_count,
        "correct_total": correct_total,
        "correct_matched": correct_matched,
        "correct_unmatched": correct_unmatched,
        "total_non_nil": total_non_nil,
        "total_nil": total_nil,
        "nil_with_subset": nil_with_subset,
    }


