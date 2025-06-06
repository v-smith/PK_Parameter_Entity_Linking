from datetime import datetime
from pathlib import Path

import typer
from sentence_transformers import SentenceTransformer

from pk_el.data_preprocessing import TEXT_DEFAULT_CONFIG, TABLE_DEFAULT_CONFIG, prep_text_features, prep_table_features
from pk_el.evaluation import split_matched_and_unmatched
from pk_el.linkers.biencoder_linker import evaluate_biencoder_el_by_source
from pk_el.linkers.prompt_linker import remove_unlinked
from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings, prepare_ontology_for_embedding, \
    add_ontology_subset_to_examples
from pk_el.utils import read_jsonl, append_result_to_jsonl

def main(
        ontology_path: Path = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
        sentences_file=typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/validation.jsonl",
            help="Path to full training sentence-level dataset."),
        tables_file=typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/validation.jsonl",
            help="Path to full training table-level dataset."),
        unlinked_sentences_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/validation_unlinked_mentions_sentences.jsonl",
            help="Path to unlinked sentence-level mentions."),
        unlinked_tables_file: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/validation_unlinked_mentions_tables.jsonl",
            help="Path to unlinked table-level mentions."),
        results_save_path: Path = typer.Option(
            "/home/vsmith/PycharmProjects/PKEntityLinker/data/results/zs_biencoder_results_validation.jsonl",
            help="Path to append JSONL-formatted summary results."),
        model_name: str = typer.Option(default='/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/final_mentionwindow_ireval_20epochs_hardnegs1_earlystop_5-0.0005/checkpoint-2400', help="Options are intfloat/e5-small-v2, all-mpnet-base-v2, cambridgeltl/SapBERT-from-PubMedBERT-fulltext, microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
        text_feature: str = typer.Option(default='mention_with_window', help="Options are mention, text_with_tagged_mention, mention_with_window."),
        table_feature: str = typer.Option(default='text_with_tagged_mention', help="Options are mention, text_with_tagged_mention, table_context_retrieval."),
        k: int = typer.Option(default=5),
        match_threshold: float = typer.Option(default=0.80, help="Match threshold"),
        category_constraint: bool = typer.Option(default=False),
        include_ontology_desc: bool = typer.Option(False),
        include_ontology_units: bool = typer.Option(False),
        debug: bool = typer.Option(default=False),
):

    # ========== Load Ontology and Mappings ========== #
    ontology_df = load_ontology(ontology_path)
    mappings = create_ontology_mappings(ontology_df)
    id_to_label_embeds, param_to_id_embeds = prepare_ontology_for_embedding(ontology_path, include_description=include_ontology_desc,
                                                                            include_units=include_ontology_units)

    ############## Load annotated datasets and unlinkable mentions ##############
    all_sentences = list(read_jsonl(sentences_file))
    all_tables = list(read_jsonl(tables_file))
    unlinked_sentences = list(read_jsonl(unlinked_sentences_file))
    unlinked_tables = list(read_jsonl(unlinked_tables_file))
    print(f"Unlinked sentences: {len(unlinked_sentences)}, Unlinked tables: {len(unlinked_tables)}")

    sentences = remove_unlinked(all_sentences, unlinked_sentences)
    tables = remove_unlinked(all_tables, unlinked_tables)
    assert len(sentences) == (len(all_sentences) - len(unlinked_sentences))
    assert len(tables) == (len(all_tables) - len(unlinked_tables))

    if debug:
        sentences = sentences[:5]
        unlinked_sentences = unlinked_sentences[:5]
        tables = tables[:5]
        unlinked_tables = unlinked_tables[:5]

    # ========== Tag and Combine for Evaluation ========== #
    for ex in sentences:
        ex["source"] = "dev"
    for ex in unlinked_sentences:
        ex["source"] = "unlinked"
    sent_eval_all = sentences + unlinked_sentences

    for ex in tables:
        ex["source"] = "dev"
    for ex in unlinked_tables:
        ex["source"] = "unlinked"
    tab_eval_all = tables + unlinked_tables

    # ========== Preprocess and Add Ontology Subsets ========== #
    tab_eval_all = prep_table_features(tab_eval_all, TABLE_DEFAULT_CONFIG)
    sent_eval_all = prep_text_features(sent_eval_all, TEXT_DEFAULT_CONFIG)

    sent_eval_all = add_ontology_subset_to_examples(
        sent_eval_all, ontology_df, mappings["param_id_to_category_id"], use_filtered_subset=True
    )
    tab_eval_all = add_ontology_subset_to_examples(
        tab_eval_all, ontology_df, mappings["param_id_to_category_id"], use_filtered_subset=True
    )

    # Split off NILs
    tables_matched, tables_unmatched = split_matched_and_unmatched(tab_eval_all)
    sentences_matched, sentences_unmatched = split_matched_and_unmatched(sent_eval_all)
    print(f"Tables Category Matched: {len(tables_matched)}, unmatched: {len(tables_unmatched)}")
    print(f"Sents Category Matched: {len(sentences_matched)}, unmatched: {len(sentences_unmatched)}")

    ############## Biencoder Retrieval ##############
    model = SentenceTransformer(model_name)
    # TABLES
    print("\n---------------")
    print(f"TABLES: Feature: {table_feature}")
    results_tables = evaluate_biencoder_el_by_source(
        dataset=tab_eval_all,
        param_to_id=param_to_id_embeds,
        model=model,
        retrieval_feature_name=table_feature,
        k=k,
        match_threshold=match_threshold,
        label_mapping=id_to_label_embeds,
        use_category_constraint=category_constraint,
        plot_conf_matrix=False,
    )

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "features": table_feature,
        "ontology_desc": include_ontology_desc,
        "ontology_units": include_ontology_units,
        "k": k,
        "match_threshold": match_threshold,
        "category_constraint": category_constraint,
        "model": model_name,
        "debug": debug,
        "data type": "tables",
        "results": results_tables,
    }

    append_result_to_jsonl(results_save_path, run_summary)

    print("---------------\n")

    # SENTENCES
    print("\n---------------")
    print(f"SENTENCES: {text_feature}")
    results_sentences = evaluate_biencoder_el_by_source(
        dataset=sent_eval_all,
        param_to_id=param_to_id_embeds,
        model=model,
        retrieval_feature_name=text_feature,
        k=k,
        match_threshold=match_threshold,
        label_mapping=id_to_label_embeds,
        use_category_constraint=category_constraint,
        plot_conf_matrix=False,
    )

    run_summary = {
        "timestamp": datetime.now().isoformat(),
        "features": text_feature,
        "ontology_desc": include_ontology_desc,
        "ontology_units": include_ontology_units,
        "k": k,
        "match_threshold": match_threshold,
        "category_constraint": category_constraint,
        "model": model_name,
        "debug": debug,
        "data type": "sentences",
        "results": results_sentences,
    }

    append_result_to_jsonl(results_save_path, run_summary)

    print("---------------\n\n")

    a = 1

if __name__ == '__main__':
    typer.run(main)