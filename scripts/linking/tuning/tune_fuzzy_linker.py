from pathlib import Path

import typer

from pk_el.data_preprocessing import prep_text_features, prep_table_features, TEXT_DEFAULT_CONFIG, TABLE_DEFAULT_CONFIG
from pk_el.linkers.exact_linker import tokenize_data, create_tokenized_param_names_and_synonyms_to_ids
from pk_el.linkers.fuzzy_linker import plot_f1_vs_matched, run_fuzzy_retrieval_eval
from pk_el.linkers.prompt_linker import split_matched_and_unmatched
from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings, add_ontology_subset_to_examples
from pk_el.utils import read_jsonl


def main(
        ontology_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
        train_unlinked_sentences_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/train_unlinked_mentions_sentences.jsonl"),
        train_unlinked_tables_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/train_unlinked_mentions_tables.jsonl"),
        val_unlinked_sentences_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/train_unlinked_mentions_sentences.jsonl"),
        val_unlinked_tables_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/train_unlinked_mentions_tables.jsonl"),
        image_save_path_sentences: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/images/fuzzy_sentences_pk_constrained.png"),
        image_save_path_tables: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/images/fuzzy_tables_pk_constrained.png"),
        debug: bool = typer.Option(default=False),
        tokenizer_name: str = typer.Option(default="pk"),
        k: int = typer.Option(default=10),
        category_constrained: bool = typer.Option(default=True),
):

    # === Load Ontology and Tokenize Parameters === #
    ontology_df = load_ontology(ontology_path)
    mappings = create_ontology_mappings(ontology_df, include_nil=True)
    tokenized_param_ids = create_tokenized_param_names_and_synonyms_to_ids(ontology_df, tokenizer_name)

    # === Load and Prepare Datasets === #
    train_sents = list(read_jsonl(train_unlinked_sentences_file))
    train_tables = list(read_jsonl(train_unlinked_tables_file))
    val_sents = list(read_jsonl(val_unlinked_sentences_file))
    val_tables = list(read_jsonl(val_unlinked_tables_file))
    if debug:
        train_sents, train_tables, val_sents, val_tables = train_sents[:5], train_tables[:5], val_sents[:5], val_tables[:5]

    sents = train_sents + val_sents
    tables = train_tables + val_tables

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

    # === Evaluate Across Match Thresholds === #
    thresholds = [50, 60, 70, 80, 90]
    all_results_sentences = []
    all_results_tables = []

    for t in thresholds:
        print(f"\nEvaluating: k={k}, match_threshold={t}")

        table_metrics = run_fuzzy_retrieval_eval(tables_matched, tokenized_param_ids, k, t, mappings["param_id_to_param_name"],
                                                 matching_mode="string", category_constrained=category_constrained)
        sent_metrics = run_fuzzy_retrieval_eval(sentences_matched, tokenized_param_ids, k, t, mappings["param_id_to_param_name"],
                                                matching_mode="string", category_constrained=category_constrained)

        all_results_tables.append(table_metrics)
        all_results_sentences.append(sent_metrics)


    plot_f1_vs_matched(all_results_tables, dataset_name="Tables - string", score_key="micro_f1_matched", save_path=image_save_path_tables)
    plot_f1_vs_matched(all_results_sentences, dataset_name="Sentences - string", score_key="micro_f1_matched", save_path=image_save_path_sentences)


if __name__ == "__main__":
    typer.run(main)