from collections import defaultdict

from matplotlib import pyplot as plt, cm

from pk_el.utils import read_jsonl

MODEL_LABELS = {
        "intfloat/e5-small-v2": "E5",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext": "SapBERT",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext": "PubMedBERT",
        "/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/final_mentionwindow_ireval_20epochs_hardnegs1_earlystop_5-0.0005/checkpoint-2400": "Finetuned",
    }

SUBSET_LABELS = {
    "dev": "Resolved",
    "unlinked": "Unresolved"
}


def plot_linked_unlinked_f1_with_nils(data):
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for entry in data:
        data_type = entry.get("data type")
        model = entry.get("model")
        threshold = entry.get("match_threshold")

        for subset, result in entry["results"].items():
            f1 = result.get("micro_f1")
            nil_prop = result.get("nil_prop", None)
            grouped[data_type][model][subset].append((threshold, f1, nil_prop))

    for data_type in grouped:
        for model in grouped[data_type]:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()

            label_base = MODEL_LABELS.get(model, model)

            color_map = {
                "dev": 'tab:blue',
                "unlinked": 'tab:orange'
            }

            for subset in ["dev", "unlinked"]:
                if subset not in grouped[data_type][model]:
                    continue

                points = sorted(grouped[data_type][model][subset], key=lambda x: x[0])
                thresholds, f1s, nils = zip(*points)
                color = color_map[subset]

                # ✅ Solid line for F1
                ax1.plot(thresholds, f1s, linestyle='-', marker='o',
                         color=color, label=f"{SUBSET_LABELS.get(subset, subset)} F1")

                # ✅ Dashed line for NIL%
                ax2.plot(thresholds, nils, linestyle='--', marker='x',
                         color=color, alpha=0.6, label=f"{SUBSET_LABELS.get(subset, subset)} NIL%")

            #ax1.set_title(f"{label_base} on {data_type.capitalize()} Data")
            ax1.set_xlabel("Match Threshold")
            ax1.set_ylabel("Micro-F1 Score")
            ax2.set_ylabel("NIL Predictions (%)")
            ax1.set_ylim(0, 100)
            ax2.set_ylim(0, 100)
            ax1.grid(True)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize='small')
            plt.tight_layout()
            save_path = "/home/vsmith/PycharmProjects/PKEntityLinker/images/" + f"biencoder_e5_thresholds_vs_f1_{data_type}.png"
            plt.savefig(save_path)
            plt.show()


my_data = list(read_jsonl("/home/vsmith/PycharmProjects/PKEntityLinker/data/results/zs_biencoder_results_validation.jsonl"))
my_data = [x for x in my_data if x["model"] == "intfloat/e5-small-v2"]

plot_linked_unlinked_f1_with_nils(my_data)

a = 1