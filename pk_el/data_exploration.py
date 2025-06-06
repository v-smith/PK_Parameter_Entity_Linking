from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_ontology_coverage(all_data, ontology_df):
    # Extract all parameter_ids from data
    data_labels = [item["label"] for item in all_data if "label" in item and item["label"] is not None]

    # All unique parameter_ids in data
    unique_data_labels = set(data_labels)

    # All unique parameter_ids in the ontology
    kb_parameter_ids = set(ontology_df["parameter_id"])

    # Overall KB coverage
    covered = unique_data_labels & kb_parameter_ids
    coverage_pct = len(covered) / len(kb_parameter_ids) * 100
    print(f"Overall KB Coverage: {coverage_pct:.2f}% ({len(covered)} out of {len(kb_parameter_ids)})")


def calculate_ontology_coverage_by_category(all_data, ontology_df):
    # Extract all parameter_ids from data
    data_labels = [item["label"] for item in all_data if "label" in item and item["label"] is not None]

    # All unique parameter_ids in data
    unique_data_labels = set(data_labels)

    # Add a helper dict: parameter_id -> category_name
    param_to_category = dict(zip(ontology_df["parameter_id"], ontology_df["parameter_category"]))


    # Group KB parameter_ids by category
    category_to_params = ontology_df.groupby("parameter_category")["parameter_id"].apply(set).to_dict()
    category_to_params["NIL"] = {"Q100"}
    a = 1

    # Initialize category-wise coverage
    category_coverage = {}

    for category, param_ids in category_to_params.items():
        covered_params = param_ids & unique_data_labels
        coverage = len(covered_params) / len(param_ids) * 100
        category_coverage[category] = {
            "coverage_pct": coverage,
            "covered": len(covered_params),
            "total": len(param_ids)
        }

    # Display
    for cat, stats in category_coverage.items():
        print(f"Category: {cat} - {stats['coverage_pct']:.2f}% ({stats['covered']}/{stats['total']})")



def get_single_label_stats(data, label_id="Q100"):
    total = len(data)
    label_count = sum(1 for item in data if item.get("label") == label_id)
    percentage = (label_count / total) * 100 if total > 0 else 0
    print(f"{label_id} count: {label_count} ({percentage:.2f}%)")
    return label_count, round(percentage, 2)



def generate_analysis_results(all_data, ontology_df):
    # Add a helper dict: parameter_id -> category_name
    param_to_category = dict(zip(ontology_df["parameter_id"], ontology_df["parameter_category"]))

    # Group KB parameter_ids by category
    category_to_params = ontology_df.groupby("parameter_category")["parameter_id"].apply(set).to_dict()

    # Extract all parameter_ids from data
    data_labels = [item["label"] for item in all_data if "label" in item and item["label"] is not None]

    # All unique parameter_ids in data
    unique_data_labels = set(data_labels)
    # Total labels used in data
    total_label_count = len(data_labels)

    category_coverage = {}
    category_distribution = Counter(
        [param_to_category[label] for label in data_labels if label in param_to_category]
    )

    for category, param_ids in category_to_params.items():
        covered_params = param_ids & unique_data_labels
        label_count_in_cat = category_distribution.get(category, 0)
        category_coverage[category] = {
            "coverage_pct": len(covered_params) / len(param_ids) * 100,
            "distribution_pct": label_count_in_cat / total_label_count * 100,
            "parameter_counts": Counter(
                label for label in data_labels if param_to_category.get(label) == category
            )
        }

    # Build final analysis_results dictionary
    analysis_results = {"category_coverage": category_coverage}

    return analysis_results



def plot_category_metrics(analysis_results, save_path=None):
    categories = [cat for cat in analysis_results['category_coverage'].keys() if cat != 'NIL']
    coverage = [analysis_results['category_coverage'][cat]['coverage_pct'] for cat in categories]
    distribution = [analysis_results['category_coverage'][cat]['distribution_pct'] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, coverage, width, label='Coverage %')
    rects2 = ax.bar(x + width / 2, distribution, width, label='Distribution %')

    ax.set_ylabel('Percentage')
    #ax.set_title('Category Coverage vs Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()



def plot_parameter_frequency_by_category(analysis_results, label_mapping, save_path=None):
    """
    Create a grouped bar chart showing parameter frequencies by category
    """
    # Prepare data
    data = []
    for cat in analysis_results['category_coverage'].keys():
        param_counts = analysis_results['category_coverage'][cat]['parameter_counts']
        for param_id, count in param_counts.items():
            param_name = label_mapping.get(param_id, param_id)
            data.append({
                'Category': cat,
                'Parameter': param_name,
                'Count': count
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create grouped bar chart with larger figure size
    plt.figure(figsize=(20, 12))  # Increased figure size

    # Get unique parameters and categories
    params = df['Parameter'].unique()
    categories = df['Category'].unique()

    # Set the positions of the bars on the x-axis
    x = np.arange(len(params))
    width = 1.5 / len(categories)  # Increased bar width

    # Plot bars for each category
    for i, category in enumerate(categories):
        category_data = df[df['Category'] == category]
        counts = [category_data[category_data['Parameter'] == param]['Count'].values[0]
                  if len(category_data[category_data['Parameter'] == param]) > 0 else 0
                  for param in params]

        plt.bar(x + i * width, counts, width, label=category)

    # Customize the plot
    plt.xlabel('Parameters', labelpad=20)  # Added padding for label
    plt.ylabel('Frequency')
    #plt.title('Parameter Frequencies by Category', pad=20)  # Added padding for title

    # Rotate parameter names vertically
    plt.xticks(x + width * len(categories) / 2, params, rotation=90, ha='center', va='top')

    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.subplots_adjust(bottom=0.2)  # Make room for parameter names
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return df


def plot_parameter_distribution(dataset_df, kb_df, label_col='label', save_path=None):
    """
    Plot distribution of parameter labels, including parameters with zero occurrences.

    Args:
        dataset_df: DataFrame containing annotations
        kb_dict: Knowledge base dictionary
        label_col: Name of column containing parameter labels
        save_path: Optional path to save the plot
    """
    # Create label mapping
    kb_dict = kb_df.to_dict('records')
    label_mapping = {d["parameter_id"]: d["parameter_name"] for d in kb_dict}
    #label_mapping["Q100"] = "NIL"

    # Count parameter label occurrences
    label_counts = Counter(dataset_df[label_col].dropna())

    # Add zero counts for labels not present in dataset
    for label_id in label_mapping:
        if label_id not in label_counts:
            label_counts[label_id] = 0

    # Remove 'Q100' if present
    label_counts.pop("Q100", None)

    # Sort by frequency
    sorted_labels_and_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_label_ids = [item[0] for item in sorted_labels_and_counts]
    sorted_counts = [item[1] for item in sorted_labels_and_counts]
    sorted_label_names = [label_mapping.get(label_id, label_id) for label_id in sorted_label_ids]

    # Create plot
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(sorted_counts)), sorted_counts)

    # Customize plot
    plt.xlabel("Parameters")
    plt.ylabel("Frequency")
    #plt.title("Parameter Label Distribution")

    # Set x-ticks with parameter names
    plt.xticks(range(len(sorted_counts)), sorted_label_names, rotation=90, ha='right')

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

    # Return the sorted data for potential further analysis
    return {
        'label_names': sorted_label_names,
        'counts': sorted_counts,
        'label_ids': sorted_label_ids
    }

