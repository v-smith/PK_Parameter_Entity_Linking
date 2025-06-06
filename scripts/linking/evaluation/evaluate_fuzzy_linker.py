from pathlib import Path

import typer

from pk_el.data_preprocessing import prep_text_features, prep_table_features, TEXT_DEFAULT_CONFIG, TABLE_DEFAULT_CONFIG
from pk_el.linkers.exact_linker import tokenize_data, create_tokenized_param_names_and_synonyms_to_ids
from pk_el.linkers.fuzzy_linker import run_fuzzy_retrieval_eval
from pk_el.linkers.prompt_linker import split_matched_and_unmatched
from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings, add_ontology_subset_to_examples
from pk_el.utils import read_jsonl


def main(
        ontology_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
        unlinked_sentences_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/test_unlinked_mentions_sentences.jsonl"),
        unlinked_tables_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/test_unlinked_mentions_tables.jsonl"),
        debug: bool = typer.Option(default=False),
        tokenizer_name: str = typer.Option(default="pk"),
        k: int = typer.Option(default=10),
        table_threshold: int = typer.Option(default=80),
        sentence_threshold: int = typer.Option(default=80),
        category_constrained: bool = typer.Option(default=True),
):

    # === Load Ontology and Tokenize Parameters === #
    ontology_df = load_ontology(ontology_path)
    mappings = create_ontology_mappings(ontology_df, include_nil=True)
    tokenized_param_ids = create_tokenized_param_names_and_synonyms_to_ids(ontology_df, tokenizer_name)

    # === Load and Prepare Datasets === #
    sents = list(read_jsonl(unlinked_sentences_file))
    tables = list(read_jsonl(unlinked_tables_file))
    if debug:
        sents, tables = sents[:5], tables[:5]

    sentences = tokenize_data(prep_text_features(sents, TEXT_DEFAULT_CONFIG), tokenizer_name)
    tables = tokenize_data(prep_table_features(tables, TABLE_DEFAULT_CONFIG), tokenizer_name)

    # === Establish Category === #
    for dataset, name in [(tables, "tables"), (sentences, "sentences")]:
        dataset_with_subset = add_ontology_subset_to_examples(
            dataset,
            ontology_df,
            mappings["param_id_to_category_id"],
            use_filtered_subset=True,
        )
        if name == "tables":
            tables = dataset_with_subset
        else:
            sentences = dataset_with_subset

    # Split off NILs
    tables_matched, tables_unmatched = split_matched_and_unmatched(tables)
    sentences_matched, sentences_unmatched = split_matched_and_unmatched(sentences)
    print(f"Tables Matched: {len(tables_matched)}, unmatched: {len(tables_unmatched)}")
    print(f"Sents Matched: {len(sentences_matched)}, unmatched: {len(sentences_unmatched)}")

    # === Evaluate === #
    print("TABLES")
    table_results = run_fuzzy_retrieval_eval(tables_matched, tokenized_param_ids, k, table_threshold, mappings["param_id_to_param_name"],
                                             matching_mode="string", category_constrained=category_constrained)
    tables_unmatched = table_results["unmatched_examples"]
    print(f"Still unmatched tables: {len(tables_unmatched)}")

    print("SENTENCES")
    sentence_results = run_fuzzy_retrieval_eval(sentences_matched, tokenized_param_ids, k, sentence_threshold, mappings["param_id_to_param_name"],
                                            matching_mode="string", category_constrained=category_constrained)
    sentences_unmatched = sentence_results["unmatched_examples"]
    print(f"Still unmatched sentences: {len(sentences_unmatched)}")

    a = 1

if __name__ == "__main__":
    typer.run(main)