from pathlib import Path

import typer

from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings, add_ontology_subset_to_examples, \
    evaluate_subset_matching
from pk_el.utils import read_jsonl


def main(ontology_path: Path = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
         sentences_file_path: Path = typer.Option(
             "/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/validation.jsonl"),
         tables_file_path: Path = typer.Option(
             "/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/validation.jsonl"),
         debug: bool = typer.Option(default=False),
         ):

    # ========== Load Ontology and Mappings ========== #
    ontology_df = load_ontology(ontology_path)
    mappings = create_ontology_mappings(ontology_df)

    # === Load Annotated Datasets === #
    sentences = list(read_jsonl(sentences_file_path))
    tables = list(read_jsonl(tables_file_path))
    if debug:
        sentences, tables = sentences[:10], tables[:10]

    # ========== Add Ontology Subsets & Get Results ========== #
    for dataset, name in [(tables, "tables"), (sentences, "sentences")]:
        print(f"\n\n#==== {name} ====#")
        dataset_with_subsets = add_ontology_subset_to_examples(
            dataset,
            ontology_df=ontology_df,
            param_id_to_category_id=mappings["param_id_to_category_id"],
            use_filtered_subset=True,
        )

        evaluate_subset_matching(
            dataset_with_subsets,
            param_id_to_category_id=mappings["param_id_to_category_id"],
        )

    a = 1

if __name__ == '__main__':
    typer.run(main)