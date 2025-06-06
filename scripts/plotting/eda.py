from pathlib import Path

import pandas as pd
import typer

from pk_el.data_exploration import plot_parameter_distribution, plot_category_metrics, \
    plot_parameter_frequency_by_category, calculate_ontology_coverage, calculate_ontology_coverage_by_category, \
    generate_analysis_results, get_single_label_stats
from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings
from pk_el.utils import read_jsonl

def main(ontology_path: Path = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
         sentence_train_data_path: str = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/train.jsonl"),
         sentence_val_data_path: str = typer.Option("/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/validation.jsonl"),
         sentence_test_data_path: str= typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/test.jsonl"),
         table_train_data_path: str = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/train.jsonl"),
         table_val_data_path: str = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/validation.jsonl"),
         table_test_data_path: str = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/test.jsonl")
         ):

    # ========== Load Ontology and Mappings ========== #
    ontology_df = load_ontology(ontology_path, remove_nil=True)
    mappings = create_ontology_mappings(ontology_df)

    # ========== EDA ========== #
    print("\n\n=============Sentence Analysis: =============\n")
    sentence_train_data = list(read_jsonl(sentence_train_data_path))
    sentence_val_data = list(read_jsonl(sentence_val_data_path))
    sentence_test_data = list(read_jsonl(sentence_test_data_path))
    all_sentence_data = sentence_train_data + sentence_test_data + sentence_val_data
    print(f"Number of sentences (train): {len(sentence_train_data)}")
    print(f"Number of sentences (val): {len(sentence_val_data)}")
    print(f"Number of sentences (test): {len(sentence_test_data)}")
    print(f"Total sentences: {len(all_sentence_data)}\n")

    for name, dataset in [("all", all_sentence_data), ("train", sentence_train_data), ("val", sentence_val_data), ("test", sentence_test_data)]:
        print(f"\n{name}\n")
        calculate_ontology_coverage(dataset, ontology_df)
        get_single_label_stats(dataset, label_id="Q100")
        #calculate_ontology_coverage_by_category(dataset, ontology_df)

    # plots
    all_sentence_df = pd.DataFrame([item for item in all_sentence_data if "label" in item], columns=["label"])
    plot_parameter_distribution(all_sentence_df, ontology_df, save_path="/home/vsmith/PycharmProjects/PKEntityLinker/images/eda_param_distribution_sents.png")

    sentence_analysis_results = generate_analysis_results(all_sentence_data, ontology_df)
    plot_category_metrics(sentence_analysis_results,
                          save_path="/home/vsmith/PycharmProjects/PKEntityLinker/images/eda_param_category_metrics_sents.png")
    plot_parameter_frequency_by_category(sentence_analysis_results,
                                         label_mapping=mappings["param_id_to_param_name"],
                                         save_path="/home/vsmith/PycharmProjects/PKEntityLinker/images/eda_param_frequency_by_category_sents.png")

    print("\n\n=============Table Analysis: =============\n")
    table_train_data = list(read_jsonl(table_train_data_path))
    table_val_data = list(read_jsonl(table_val_data_path))
    table_test_data = list(read_jsonl(table_test_data_path))
    all_table_data = table_train_data + table_test_data + table_val_data
    print(f"Number of tables (train): {len(table_train_data)}")
    print(f"Number of tables (val): {len(table_val_data)}")
    print(f"Number of tables (test): {len(table_test_data)}")
    print(f"Total tables: {len(all_table_data)}\n")

    for name, dataset in [("all", all_table_data), ("train", table_train_data), ("val", table_val_data), ("test", table_test_data)]:
        print(f"\n{name}\n")
        calculate_ontology_coverage(dataset, ontology_df)
        get_single_label_stats(dataset, label_id="Q100")
        #calculate_ontology_coverage_by_category(dataset, ontology_df)

    # plots
    all_table_df = pd.DataFrame([item for item in all_table_data if "label" in item], columns=["label"])
    plot_parameter_distribution(all_table_df, ontology_df, save_path="/home/vsmith/PycharmProjects/PKEntityLinker/images/eda_param_distribution_tables.png")

    table_analysis_results = generate_analysis_results(all_table_data, ontology_df)
    plot_category_metrics(table_analysis_results, save_path="/home/vsmith/PycharmProjects/PKEntityLinker/images/eda_param_category_metrics_tables.png")
    plot_parameter_frequency_by_category(table_analysis_results, label_mapping=mappings["param_id_to_param_name"], save_path="/home/vsmith/PycharmProjects/PKEntityLinker/images/eda_param_frequency_by_category_tables.png")

    a = 1

if __name__ == '__main__':
    typer.run(main)