from pathlib import Path

import typer
from datasets import concatenate_datasets
from sentence_transformers import SentenceTransformer, losses, models, SentenceTransformerTrainingArguments, \
    SentenceTransformerTrainer, SimilarityFunction
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator
from transformers import EarlyStoppingCallback

from pk_el.data_preprocessing import prep_table_features, prep_text_features, TABLE_DEFAULT_CONFIG, TEXT_DEFAULT_CONFIG
from pk_el.linkers.biencoder_linker import generate_input_examples
from pk_el.linkers.prompt_linker import remove_unlinked
from pk_el.ontology_preprocessing import load_ontology, create_ontology_mappings, prepare_ontology_for_embedding, \
    add_ontology_subset_to_examples
from pk_el.utils import read_jsonl


def main(
        ontology_path: Path = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/kb/pk_kb.csv"),
        train_sentences_file: Path = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/train.jsonl"),
        train_tables_file: Path = typer.Option(default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/train.jsonl"),
        val_sentences_file=typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/sentences/validation.jsonl",
            help="Path to full training sentence-level dataset."),
        val_tables_file=typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/gold_standard/tables/validation.jsonl",
            help="Path to full training table-level dataset."),
        val_unlinked_sentences: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/validation_unlinked_mentions_sentences.jsonl",
            help="Path to unlinked sentence-level mentions."),
        val_unlinked_tables: Path = typer.Option(
            default="/home/vsmith/PycharmProjects/PKEntityLinker/data/unlinked_exact/validation_unlinked_mentions_tables.jsonl",
            help="Path to unlinked table-level mentions."),
        model_name: str = typer.Option(default='intfloat/e5-small-v2'),
        text_feature: str = typer.Option(default='mention_with_window',  help="Options are mention, text_with_tagged_mention, mention_with_window."),
        table_feature: str = typer.Option(default='text_with_tagged_mention', help="Options are mention, text_with_tagged_mention, table_context_retrieval."),
        use_hard_negatives: bool = typer.Option(default=False, help="Use hard negatives"),
        num_hard_negatives: int = typer.Option(default=1, help="Number of hard negatives"),
        use_random_negatives: bool = typer.Option(default=True, help="Use random negatives"),
        num_random_negatives: int = typer.Option(default=2, help="Number of random negatives"),
        use_early_stopping: bool = typer.Option(default=False, help="Use early stopping"),
        seed: int = typer.Option(default=42, help="Random seed."),
        lr: float = typer.Option(default=2e-5, help="Learning rate."),
        debug: bool = typer.Option(default=False),
        run_name: str = typer.Option(default="trial"),
):

    # ========== Load Ontology and Mappings ========== #
    ontology_df = load_ontology(ontology_path)
    mappings = create_ontology_mappings(ontology_df)
    id_to_label_embeds, param_to_id_embeds = prepare_ontology_for_embedding(ontology_path,
                                                                            include_description=True,
                                                                            include_units=False)

    ############## Load annotated datasets and unlinkable mentions ##############
    sentences_train = list(read_jsonl(train_sentences_file))
    tables_train = list(read_jsonl(train_tables_file))

    sentences_val = list(read_jsonl(val_sentences_file))
    tables_val = list(read_jsonl(val_tables_file))
    unlinked_sentences_val = list(read_jsonl(val_unlinked_sentences))
    unlinked_tables_val = list(read_jsonl(val_unlinked_tables))
    linked_sentences_val = remove_unlinked(sentences_val, unlinked_sentences_val)
    linked_tables_val = remove_unlinked(tables_val, unlinked_tables_val)
    assert len(linked_sentences_val) == (len(sentences_val) - len(unlinked_sentences_val))
    assert len(linked_tables_val) == (len(tables_val) - len(unlinked_tables_val))

    if debug:
        sentences_train = sentences_train[:5]
        tables_train = tables_train[:5]
        unlinked_sentences_val = unlinked_sentences_val[:5]
        linked_sentences_val = linked_sentences_val[:5]
        unlinked_tables_val = unlinked_tables_val[:5]
        linked_tables_val = linked_tables_val[:5]

    # ========== Tag and Combine Evaluation Data ========== #
    for ex in linked_sentences_val:
        ex["source"] = "dev"
    for ex in unlinked_sentences_val:
        ex["source"] = "unlinked"
    sent_eval_all = linked_sentences_val + unlinked_sentences_val

    for ex in linked_tables_val:
        ex["source"] = "dev"
    for ex in unlinked_tables_val:
        ex["source"] = "unlinked"
    tab_eval_all = linked_tables_val + unlinked_tables_val

    ############## Prepare Data and Ontology for Training ##############
    tables_train = prep_table_features(tables_train,TABLE_DEFAULT_CONFIG)
    sentences_train = prep_text_features(sentences_train,TEXT_DEFAULT_CONFIG)

    tab_eval_all = prep_table_features(tab_eval_all, TABLE_DEFAULT_CONFIG)
    sent_eval_all = prep_text_features(sent_eval_all, TEXT_DEFAULT_CONFIG)

    sent_eval_all = add_ontology_subset_to_examples(
        sent_eval_all, ontology_df, mappings["param_id_to_category_id"], use_filtered_subset=True
    )
    tab_eval_all = add_ontology_subset_to_examples(
        tab_eval_all, ontology_df, mappings["param_id_to_category_id"], use_filtered_subset=True
    )
    #combined_train_data = tables_train + sentences_train
    combined_val_data = tab_eval_all + sent_eval_all

    # convert to sentence transformer inputs
    train_table_examples = generate_input_examples(tables_train,
                                             id_to_label_embeds,
                                             param_to_category=mappings['param_id_to_category_id'],
                                             mention_key=table_feature,
                                             include_hard_negatives=use_hard_negatives,
                                             num_hard_negs=num_hard_negatives,
                                             include_random_neg=use_random_negatives,
                                             num_random_negs=num_random_negatives,
                                             seed=seed)

    train_sentence_examples = generate_input_examples(sentences_train,
                                                   id_to_label_embeds,
                                                   param_to_category=mappings['param_id_to_category_id'],
                                                   mention_key=text_feature,
                                                   include_hard_negatives=use_hard_negatives,
                                                   num_hard_negs=num_hard_negatives,
                                                   include_random_neg=True,
                                                   num_random_negs=2,
                                                   seed=seed)

    train_examples = concatenate_datasets([train_table_examples, train_sentence_examples])
    print(f"train_examples: {len(train_examples)}")

    val_examples_tables = generate_input_examples(tab_eval_all,
                                           id_to_label_embeds,
                                           param_to_category=mappings['param_id_to_category_id'],
                                           mention_key=table_feature,
                                           include_hard_negatives=False,
                                           include_random_neg=False)


    val_examples_sents = generate_input_examples(sent_eval_all,
                                           id_to_label_embeds,
                                           param_to_category=mappings['param_id_to_category_id'],
                                           mention_key=text_feature,
                                           include_hard_negatives=False,
                                           include_random_neg=False)

    val_examples = concatenate_datasets([val_examples_tables, val_examples_sents])

    val_examples_tables = generate_input_examples(
        tab_eval_all,
        id_to_label_embeds,
        param_to_category=mappings['param_id_to_category_id'],
        mention_key=table_feature,
        include_hard_negatives=False,
        include_random_neg=False,
        return_dataset=False
    )

    val_examples_sents = generate_input_examples(
        sent_eval_all,
        id_to_label_embeds,
        param_to_category=mappings['param_id_to_category_id'],
        mention_key=text_feature,
        include_hard_negatives=False,
        include_random_neg=False,
        return_dataset=False
    )

    val_examples_eval = val_examples_tables + val_examples_sents

    ############## Load Model ##############
    model = SentenceTransformer(model_name)
    tokens = ["[MENTION]", "[/MENTION]", "[PARAM]", "[SYN]", "[DESC]"] #  "[CELL]", "[COL-HEADER]", "[ROW-HEADER]"]
    word_embedding_model = model._first_module()
    word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    ############## Fit Model ##############
    embeds_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples=val_examples_eval,
                                                                        name='val-eval',
                                                                        main_similarity=SimilarityFunction.COSINE,
                                                                        )

    # Build corpus from pk_ontology
    corpus = {k: v for k, v in id_to_label_embeds.items()}

    # Build queries from mentions
    queries = {}
    relevant_docs = {}

    for i, ex in enumerate(combined_val_data):
        qid = f"mention_{i}"
        queries[qid] = ex["mention"]

        gold_id = ex["label"]
        if gold_id != "Q100":
            relevant_docs[qid] = {gold_id}

    ir_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="biomed-el-ir-eval",
    batch_size=32,
    show_progress_bar=True,
    accuracy_at_k=[1],
    mrr_at_k=[5],
    ndcg_at_k=[1],
    precision_recall_at_k=[1]
    )

    train_loss = losses.CosineSimilarityLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir="/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/" + run_name,
        max_steps=8000,
        learning_rate=lr,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=100,
        logging_dir="/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/"  + "logs/" + run_name,
        report_to="tensorboard",
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_biomed-el-ir-eval_cosine_accuracy@1",
        greater_is_better=True,
    )

    if use_early_stopping:

        class DebugEarlyStopping(EarlyStoppingCallback):
            def on_evaluate(self, args, state, control, metrics, **kwargs):
                print(f"[DEBUG] Eval step {state.global_step}, Metrics: {metrics}")
                return super().on_evaluate(args, state, control, metrics, **kwargs)

        early_stopper = DebugEarlyStopping(
            early_stopping_patience=10,
            early_stopping_threshold=0.0005,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_examples,
            eval_dataset=val_examples,
            loss=train_loss,
            evaluator=ir_evaluator,
            callbacks=[early_stopper],
        )
    else:
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_examples,
            eval_dataset=val_examples,
            loss=train_loss,
            evaluator=ir_evaluator,
        )

    trainer.train()
    a = 1

if __name__ == '__main__':
    typer.run(main)