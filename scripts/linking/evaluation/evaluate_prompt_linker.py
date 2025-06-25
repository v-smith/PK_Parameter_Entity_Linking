import os
from pathlib import Path

import openai
import typer

from pk_el.data_preprocessing import prep_text_features, prep_table_features, TEXT_DEFAULT_CONFIG, TABLE_DEFAULT_CONFIG
from pk_el.evaluation import print_classification_errors, split_matched_and_unmatched
from pk_el.linkers.prompt_linker import TABLE_EXAMPLES, SENTENCE_EXAMPLES, SYSTEM_PROMPT_COT, SYSTEM_PROMPT_STANDARD, \
    evaluate_llm_runs, remove_unlinked
from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings, add_ontology_subset_to_examples, \
    format_ontology_for_llm
from pk_el.utils import read_jsonl, write_jsonl


def main(ontology_path: Path = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
         sentences_file=typer.Option(
             default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/test.jsonl",
             help="Path to full training sentence-level dataset."),
         tables_file=typer.Option(
             default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/test.jsonl",
             help="Path to full training table-level dataset."),
         unlinked_sentences_file: Path = typer.Option(
             default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/test_unlinked_mentions_sentences.jsonl"),
         unlinked_tables_file: Path = typer.Option(
             default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/test_unlinked_mentions_tables.jsonl"),
         errors_sentences_save_path: Path = typer.Option(
             "/home/vsmith/PycharmProjects/PKEntityLinker/data/errors/test_errors_sents_prompt.jsonl"),
         errors_tables_save_path: Path = typer.Option(
             "/home/vsmith/PycharmProjects/PKEntityLinker/data/errors/test_errors_tables_prompt.jsonl"),
         open_ai_key: str = typer.Option("your_openai_key"),
         open_ai_org: str = typer.Option("your_openai_org"),
         model_name: str = typer.Option(default="gpt-4o-mini", help="Options: gpt-4o, gpt-4o-mini, gpt-4.1-mini"),
         use_context: bool = typer.Option(True),
         use_context_text: bool = typer.Option(False, help="Give the model only the mention in context"),
         subset_ontology: bool = typer.Option(True),
         use_cot_prompt: bool = typer.Option(False),
         use_examples: bool = typer.Option(True),
         n_runs: int = typer.Option(3),
         debug: bool = typer.Option(default=False),

         ):
    print("\n🧪 Running Prompt Linking Evaluation with the following configuration:")
    print(f"- Prompt Style: {'Chain-of-Thought (CoT)' if use_cot_prompt else 'Standard Instruction'}")
    print(f"- Ontology Subsetting: {'Enabled' if subset_ontology else 'Disabled'}")
    print(f"- Few-Shot Examples: {'Included' if use_examples else 'Zero-Shot'}")
    print(f"- Context: {'Included' if use_context else 'Omitted'}")
    print(f"- Model: {model_name}")
    print(f"- Number of Runs: {n_runs}")

    # ========== Set keys ========== #
    os.environ['OPENAI_API_KEY'] = open_ai_key
    openai.api_key = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_ORGANIZATION"] = open_ai_org

    # ========== Load Ontology and Mappings ========== #
    ontology_df = load_ontology(ontology_path)
    mappings = create_ontology_mappings(ontology_df)

    # ========== Load and Filter Data ========== #
    sentences = list(read_jsonl(sentences_file))
    tables = list(read_jsonl(tables_file))
    unlinked_sentences = list(read_jsonl(unlinked_sentences_file))
    unlinked_tables = list(read_jsonl(unlinked_tables_file))
    print(f"Unlinked sentences: {len(unlinked_sentences)}, Unlinked tables: {len(unlinked_tables)}")

    sentences = remove_unlinked(sentences, unlinked_sentences)
    tables = remove_unlinked(tables, unlinked_tables)

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

    if not subset_ontology:
        # Overwrite subset with full pk_ontology for all examples
        for example in sent_eval_all:
            example["ontology_subset"] = format_ontology_for_llm(ontology_df)
        for example in tab_eval_all:
            example["ontology_subset"] = format_ontology_for_llm(ontology_df)

    # Split off NILs
    tables_matched, tables_unmatched = split_matched_and_unmatched(tab_eval_all)
    sentences_matched, sentences_unmatched = split_matched_and_unmatched(sent_eval_all)
    print(f"Tables Category Matched: {len(tables_matched)}, unmatched: {len(tables_unmatched)}")
    print(f"Sents Category Matched: {len(sentences_matched)}, unmatched: {len(sentences_unmatched)}")

    # ========== Prepare Prompt Elements ========== #
    sentence_examples = SENTENCE_EXAMPLES if use_examples else None
    table_examples = TABLE_EXAMPLES if use_examples else None

    sentence_context_key = "text_context_llm" if use_context else None
    table_context_key = "table_context_llm" if use_context else None

    sentence_text_key = "text_with_tagged_mention" if use_context_text else "mention"
    table_text_key = "text_with_tagged_mention" if use_context_text else "mention"

    system_prompt = SYSTEM_PROMPT_COT if use_cot_prompt else SYSTEM_PROMPT_STANDARD

    # ========== Run LLM Linking ========== #
    print("\n=== TABLES ===")
    table_eval = evaluate_llm_runs(
        dataset=tables_matched,
        model_name=model_name,
        context_key=table_context_key,
        text_key=table_text_key,
        system_prompt=system_prompt,
        param_to_id=mappings["param_name_to_param_id"],
        examples=table_examples,
        id_to_label=mappings["param_id_to_param_name"],
        n_runs=n_runs
    )

    print("\n=== SENTENCES ===")
    sentence_eval = evaluate_llm_runs(
        dataset=sentences_matched,
        model_name=model_name,
        context_key=sentence_context_key,
        text_key=sentence_text_key,
        system_prompt=system_prompt,
        param_to_id=mappings["param_name_to_param_id"],
        examples=sentence_examples,
        id_to_label=mappings["param_id_to_param_name"],
        n_runs=n_runs
    )

    # ========== Save Errors ========== #
    write_jsonl(
        errors_sentences_save_path,
        print_classification_errors(sentence_eval["runs"][0], id_to_label=mappings["param_id_to_param_name"])
    )
    write_jsonl(
        errors_tables_save_path,
        print_classification_errors(table_eval["runs"][0], id_to_label=mappings["param_id_to_param_name"])
    )

    a = 1

if __name__ == '__main__':
    typer.run(main)