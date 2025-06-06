from pathlib import Path

import typer

from pk_el.data_preprocessing import prep_text_features, TEXT_DEFAULT_CONFIG, prep_table_features, TABLE_DEFAULT_CONFIG
from pk_el.linkers.exact_linker import tokenize_data, link_mentions_exact, \
    create_tokenized_param_names_and_synonyms_to_ids, evaluate_and_log
from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings
from pk_el.utils import read_jsonl, write_jsonl


def main(
    ontology_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
    sentences_file_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/test.jsonl"),
    tables_file_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/test.jsonl"),
    unlinked_sentences_save_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/test_unlinked_mentions_sentences.jsonl"),
    unlinked_tables_save_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/test_unlinked_mentions_tables.jsonl"),
    errors_sentences_save_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/errors/test_errors_sents.jsonl"),
    errors_tables_save_path: Path = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/errors/test_errors_tables.jsonl"),
    debug: bool = typer.Option(False),
    tokenizer_name: str = typer.Option("pk", help="Options: basic, pk, scispacy."),
):
    # === Load Ontology === #
    ontology_df = load_ontology(ontology_path)
    mappings = create_ontology_mappings(ontology_df, include_nil=True)
    tokenized_param_ids = create_tokenized_param_names_and_synonyms_to_ids(ontology_df, tokenizer_name)

    # === Load Annotated Datasets === #
    sentences = list(read_jsonl(sentences_file_path))
    tables = list(read_jsonl(tables_file_path))
    if debug:
        sentences, tables = sentences[:50], tables[:50]

    # === Tokenize & Prepare Features === #
    tokenized_sentences = prep_text_features(tokenize_data(sentences, tokenizer_name), TEXT_DEFAULT_CONFIG)
    tokenized_tables = prep_table_features(tokenize_data(tables, tokenizer_name), TABLE_DEFAULT_CONFIG)

    # === Entity Linking === #
    exact_results_sent = link_mentions_exact(tokenized_sentences, tokenized_param_ids, table_mention=False)
    exact_results_tab = link_mentions_exact(tokenized_tables, tokenized_param_ids, table_mention=True, text_key="table_context_llm")

    # === Save Unlinked Mentions === #
    write_jsonl(unlinked_sentences_save_path, exact_results_sent["unlinked_samples"])
    write_jsonl(unlinked_tables_save_path, exact_results_tab["unlinked_samples"])
    print(f"\nUnlinked (Sentences): {len(exact_results_sent['unlinked_samples'])}")
    print(f"Unlinked (Tables): {len(exact_results_tab['unlinked_samples'])}\n")

    # === Evaluate === #
    print("\n======================\nEVALUATION:\n")
    print("\n======\nLINKED:\n")

    evaluate_and_log("Tables", tokenized_tables, exact_results_tab, mappings["param_id_to_param_name"], errors_tables_save_path)
    print("\n-----------------------\n")
    evaluate_and_log("Sentences", tokenized_sentences, exact_results_sent, mappings["param_id_to_param_name"], errors_sentences_save_path)

    print("\n=======\nUNLINKED:\n")
    evaluate_and_log("Tables", tokenized_tables, exact_results_tab["unlinked_results"],
                     mappings["param_id_to_param_name"],
                     errors_tables_save_path)
    print("\n-----------------------\n")
    evaluate_and_log("Sentences", tokenized_sentences, exact_results_sent["unlinked_results"],
                     mappings["param_id_to_param_name"],
                     errors_sentences_save_path)

    a = 1
if __name__ == '__main__':
    typer.run(main)
