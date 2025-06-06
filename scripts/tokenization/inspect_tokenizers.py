import typer

from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings
from pk_el.tokenizers.biobert_tokenizer import biobert_tokenizer
from pk_el.tokenizers.pk_tokenizer import pk_tokenizer, basic_preprocessing
from pk_el.tokenizers.scispacy_tokenizer import scispacy_tokenizer


def tokenizer_analysis(tokenizer, ontology_file):
    """Analyzes tokenization for ontology terms."""
    ontology_df = load_ontology(ontology_file)
    mappings = create_ontology_mappings(ontology_df)
    ontology_terms = list(mappings["param_name_to_param_id"].keys())

    for param in ontology_terms:
        preprocessed_param = basic_preprocessing(param)
        tokens = tokenizer(preprocessed_param)
        print(f"('{preprocessed_param}', {tokens})")


def main(
        tokenizer_name: str = typer.Option(default="pk"),
        ontology_file: str = typer.Option(default="/home/vsmith/PycharmProjects/Prodigy_test/data/kb/pk_kb.csv"),
):
    """
    Run tokenizer analysis.

    Options for tokenizer_name:
    - 'pk'       → PK Tokenizer
    - 'scispacy' → SciSpacy Tokenizer
    - 'biobert'  → BioBERT Tokenizer
    """
    typer.echo(f"\nRunning {tokenizer_name} Tokenizer Analysis...\n")

    if tokenizer_name == "pk":
        tokenizer_analysis(pk_tokenizer, ontology_file)
    elif tokenizer_name == "scispacy":
        tokenizer_analysis(scispacy_tokenizer, ontology_file)
    elif tokenizer_name == "biobert":
        tokenizer_analysis(biobert_tokenizer, ontology_file)
    else:
        typer.echo("Invalid tokenizer name! Choose from: 'pk', 'scispacy', 'biobert'")
        raise typer.Exit(code=1)


if __name__ == '__main__':
    typer.run(main)

