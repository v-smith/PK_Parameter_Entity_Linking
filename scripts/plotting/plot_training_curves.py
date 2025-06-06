import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def exponential_moving_average(values, alpha=0.1):
    smoothed = []
    for i, v in enumerate(values):
        if i == 0:
            smoothed.append(v)
        else:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

# Path to your tensorboard run directories
run_dirs = {
    "0": "/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/logs/final_texttagged_ireval_20epochs_embedeval",
    "1": "/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/logs/final_texttagged_ireval_20epochs_hardnegs1_embedeval",
    "2": "/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/logs/final_mentionwindow_ireval_20epochs_hardnegs2_embedeval",
    "3": "/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/logs/final_mentionwindow_ireval_20epochs_hardnegs3_embedeval",
    "5" : "/home/vsmith/PycharmProjects/PKEntityLinker/trained_models/logs/final_mentionwindow_ireval_20epochs_hardnegs5_embedeval",
}

def load_scalar_from_event(run_path, tag="eval/accuracy"):
    ea = EventAccumulator(run_path)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        print(f"[WARN] Tag '{tag}' not found in {run_path}")
        return [], []
    scalar_data = ea.Scalars(tag)
    steps = [s.step for s in scalar_data]
    values = [s.value for s in scalar_data]
    return steps, values

# Plotting
cutoff_step = 5000
plt.figure(figsize=(10, 6))

for run_name, path in run_dirs.items():
    steps, values = load_scalar_from_event(path, tag="eval/biomed-el-ir-eval_cosine_accuracy@1")
    if steps:
        # Filter by cutoff
        filtered = [(s, v) for s, v in zip(steps, values) if s <= cutoff_step]
        if not filtered:
            continue
        steps_filtered, values_filtered = zip(*filtered)

        # Plot raw/unsmoothed (light gray)
        plt.plot(steps_filtered, values_filtered, color="lightgray", linewidth=1, alpha=0.5)

        # Plot smoothed (highlighted)
        smoothed_values = exponential_moving_average(values_filtered, alpha=0.3)
        plt.plot(steps_filtered, smoothed_values, label=f"Hard negs: {run_name}", linewidth=1)



#plt.title("Eval Accuracy Across Runs")
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Micro-Accuracy", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0.5, 0.95)  # Y-axis from 0.5 to 0.95# cleaner y-axis range
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="# Hard Negatives", fontsize=10, title_fontsize=11, loc="lower right")
plt.tight_layout()
plt.savefig("/home/vsmith/PycharmProjects/PKEntityLinker/images/biencoder_micro_accuracy_vs_hardnegs.png", dpi=300)
plt.show()


# Plotting
plt.figure(figsize=(10, 6))
for run_name, path in run_dirs.items():
    steps, values = load_scalar_from_event(path, tag="eval/biomed-el-ir-eval_cosine_mrr@5")
    if steps:
        # Filter by cutoff
        filtered = [(s, v) for s, v in zip(steps, values) if s <= cutoff_step]
        if not filtered:
            continue
        steps_filtered, values_filtered = zip(*filtered)

        # Plot raw/unsmoothed (light gray)
        plt.plot(steps_filtered, values_filtered, color="lightgray", linewidth=1, alpha=0.5)

        # Plot smoothed (highlighted)
        smoothed_values = exponential_moving_average(values_filtered, alpha=0.3)
        plt.plot(steps_filtered, smoothed_values, label=f"Hard negs: {run_name}", linewidth=1)


#plt.title("Eval Accuracy Across Runs")
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("MRR", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0.55, 0.95)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="# Hard Negatives", fontsize=10, title_fontsize=11, loc="lower right")
plt.tight_layout()
plt.savefig("/home/vsmith/PycharmProjects/PKEntityLinker/images/biencoder_micro_mrr_vs_hardnegs.png", dpi=300)
plt.show()

a = 1