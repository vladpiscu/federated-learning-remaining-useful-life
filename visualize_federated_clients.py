import argparse
import glob
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from preprocessing.preprocessing import COLUMNS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize RUL distributions and sensor variance differences for federated clients."
    )
    parser.add_argument(
        "--data-dir",
        default="CMAPSSData",
        help="Directory containing CMAPSSData training files.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of clients to visualize. Default: 5",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Directory to save visualization outputs. Default: visualizations",
    )
    return parser.parse_args()


def load_all_training_data(data_dir: str):
    """
    Load all training files and combine them with unit_id offsetting (same logic as train.py and run_federated.py).
    
    Returns:
        Combined DataFrame with all training data
    """
    train_files = sorted(glob.glob(os.path.join(data_dir, "train_*.txt")))

    if not train_files:
        raise FileNotFoundError(f"No training files found in: {data_dir}")

    print(f"\nFound {len(train_files)} training file(s):")
    for f in train_files:
        print(f"  - {f}")

    print("\nLoading all training files...")
    all_dataframes = []
    max_unit_id = 0

    for train_file in train_files:
        print(f"  Loading: {train_file}")
        df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMNS)
        # Adjust unit_ids to be unique across all files (same as train.py)
        df["unit_id"] = df["unit_id"] + max_unit_id
        max_unit_id = int(df["unit_id"].max())
        all_dataframes.append(df)
        print(f"    Loaded {len(df)} rows, unit_ids: {df['unit_id'].min()}-{df['unit_id'].max()}")

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"\nTotal rows loaded: {len(combined_df)}")
    print(f"Total unique unit_ids: {len(combined_df['unit_id'].unique())}")
    print(f"Unit ID range: {combined_df['unit_id'].min()} - {combined_df['unit_id'].max()}")

    return combined_df


def split_dataset_into_partitions(data_df, num_partitions: int) -> List:
    """
    Split the entire dataset into N partitions for federated learning.
    Each partition will contain data from multiple units.
    Same logic as run_federated.py.
    
    Args:
        data_df: Combined DataFrame with all training data
        num_partitions: Number of partitions (clients) to create
    
    Returns:
        List of DataFrames, one per partition
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be > 0")
    
    if num_partitions == 1:
        return [data_df.copy()]
    
    # Get unique unit_ids
    unique_unit_ids = sorted(data_df["unit_id"].unique().tolist())
    num_units = len(unique_unit_ids)
    
    print(f"\nSplitting {num_units} units across {num_partitions} clients...")
    
    # Calculate how many units per partition
    units_per_partition = num_units // num_partitions
    remainder = num_units % num_partitions
    
    partitions = []
    unit_idx = 0
    
    for i in range(num_partitions):
        # Distribute remainder units to first partitions
        num_units_for_partition = units_per_partition + (1 if i < remainder else 0)
        
        # Get unit_ids for this partition
        partition_unit_ids = unique_unit_ids[unit_idx:unit_idx + num_units_for_partition]
        unit_idx += num_units_for_partition
        
        # Filter data for this partition
        partition_df = data_df[data_df["unit_id"].isin(partition_unit_ids)].copy()
        partitions.append(partition_df)
        
        print(f"  Client {i}: {len(partition_unit_ids)} units, {len(partition_df)} rows")
    
    return partitions


def compute_rul_for_partitions(partitions: List) -> List:
    """
    Compute RUL for each partition.
    
    Returns:
        List of DataFrames with RUL computed
    """
    partitions_with_rul = []
    for i, partition_df in enumerate(partitions):
        partition_df = partition_df.copy()
        # Compute RUL: RUL = max(time_cycles) - current(time_cycles) for each unit
        partition_df["RUL"] = partition_df.groupby("unit_id")["time_cycles"].transform(
            lambda x: x.max() - x
        )
        partitions_with_rul.append(partition_df)
    return partitions_with_rul


def visualize_rul_distributions(partitions_with_rul: List, output_dir: str):
    """
    Visualize RUL distributions for each client.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating RUL distribution visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # Prepare data for visualization
    all_rul_data = []
    for i, partition_df in enumerate(partitions_with_rul):
        all_rul_data.append({
            'client_id': i,
            'rul': partition_df['RUL'].values,
            'num_units': partition_df['unit_id'].nunique(),
            'num_samples': len(partition_df),
        })
    
    # 1. Overlapping histograms
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(partitions_with_rul)))
    
    for i, data in enumerate(all_rul_data):
        ax.hist(data['rul'], bins=50, alpha=0.6, label=f"Client {data['client_id']} (n={data['num_units']} units)", 
                color=colors[i], density=True)
    
    ax.set_xlabel('Remaining Useful Life (RUL)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('RUL Distribution Across Federated Clients (Overlapping)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "rul_distributions_overlapping.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 2. Side-by-side subplots
    n_clients = len(partitions_with_rul)
    n_cols = min(3, n_clients)
    n_rows = (n_clients + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_clients == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for i, data in enumerate(all_rul_data):
        ax = axes[i]
        ax.hist(data['rul'], bins=50, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.5)
        ax.set_xlabel('RUL', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f"Client {data['client_id']}\n({data['num_units']} units, {data['num_samples']} samples)", 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_clients, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('RUL Distribution per Client', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "rul_distributions_per_client.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 3. Box plot comparison
    fig, ax = plt.subplots(figsize=(max(10, n_clients * 1.5), 6))
    
    rul_data_for_box = []
    client_labels = []
    for data in all_rul_data:
        rul_data_for_box.append(data['rul'])
        client_labels.append(f"Client {data['client_id']}\n({data['num_units']} units)")
    
    bp = ax.boxplot(rul_data_for_box, labels=client_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors[:n_clients]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Remaining Useful Life (RUL)', fontsize=12)
    ax.set_xlabel('Client', fontsize=12)
    ax.set_title('RUL Distribution Comparison Across Clients (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "rul_distributions_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 4. Violin plot for better distribution shape
    fig, ax = plt.subplots(figsize=(max(10, n_clients * 1.5), 6))
    
    # Prepare data in long format for seaborn
    df_long = pd.DataFrame()
    for data in all_rul_data:
        temp_df = pd.DataFrame({
            'RUL': data['rul'],
            'Client': f"Client {data['client_id']}\n({data['num_units']} units)"
        })
        df_long = pd.concat([df_long, temp_df], ignore_index=True)
    
    sns.violinplot(data=df_long, x='Client', y='RUL', ax=ax, palette=colors[:n_clients], inner='box')
    ax.set_ylabel('Remaining Useful Life (RUL)', fontsize=12)
    ax.set_xlabel('Client', fontsize=12)
    ax.set_title('RUL Distribution Comparison Across Clients (Violin Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "rul_distributions_violin.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 5. Summary statistics table
    print("\nRUL Summary Statistics per Client:")
    print("=" * 80)
    print(f"{'Client':<10} {'Units':<10} {'Samples':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    for data in all_rul_data:
        mean_rul = np.mean(data['rul'])
        std_rul = np.std(data['rul'])
        min_rul = np.min(data['rul'])
        max_rul = np.max(data['rul'])
        print(f"Client {data['client_id']:<5} {data['num_units']:<10} {data['num_samples']:<12} "
              f"{mean_rul:<12.2f} {std_rul:<12.2f} {min_rul:<12.2f} {max_rul:<12.2f}")
    print("=" * 80)


def visualize_sensor_variance(partitions_with_rul: List, output_dir: str):
    """
    Visualize sensor variance differences across clients.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating sensor variance visualizations...")
    
    # Get sensor columns
    sensor_columns = [col for col in COLUMNS if col.startswith("sensor")]
    
    # Compute variance for each sensor in each client
    client_variances = []
    for i, partition_df in enumerate(partitions_with_rul):
        variances = partition_df[sensor_columns].var()
        client_variances.append({
            'client_id': i,
            'variances': variances,
            'num_units': partition_df['unit_id'].nunique(),
        })
    
    # 1. Heatmap of variances across clients
    fig, ax = plt.subplots(figsize=(14, max(6, len(partitions_with_rul) * 0.8)))
    
    # Create variance matrix
    variance_matrix = np.array([cv['variances'].values for cv in client_variances])
    client_labels = [f"Client {cv['client_id']}\n({cv['num_units']} units)" for cv in client_variances]
    
    im = ax.imshow(variance_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(sensor_columns)))
    ax.set_xticklabels(sensor_columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(client_labels)))
    ax.set_yticklabels(client_labels, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Variance', fontsize=11)
    
    # Add text annotations
    for i in range(len(client_labels)):
        for j in range(len(sensor_columns)):
            text = ax.text(j, i, f'{variance_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=7)
    
    ax.set_title('Sensor Variance Across Federated Clients (Heatmap)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, "sensor_variance_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 2. Bar plot comparing variance for selected sensors
    # Select sensors with highest average variance
    avg_variances = np.mean(variance_matrix, axis=0)
    top_sensors_idx = np.argsort(avg_variances)[-10:]  # Top 10 sensors
    top_sensors = [sensor_columns[i] for i in top_sensors_idx]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(top_sensors))
    width = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(client_variances)))
    
    for i, cv in enumerate(client_variances):
        variances = cv['variances'][top_sensors].values
        offset = (i - len(client_variances) / 2 + 0.5) * width
        ax.bar(x + offset, variances, width, label=f"Client {cv['client_id']}", 
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Sensor', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('Top 10 Sensors by Average Variance Across Clients', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_sensors, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "sensor_variance_top_sensors.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 3. Variance difference from global average
    global_avg_variance = np.mean(variance_matrix, axis=0)
    
    fig, axes = plt.subplots(len(client_variances), 1, figsize=(14, 3 * len(client_variances)))
    if len(client_variances) == 1:
        axes = [axes]
    
    for idx, cv in enumerate(client_variances):
        ax = axes[idx]
        variances = cv['variances'].values
        differences = variances - global_avg_variance
        
        colors_diff = ['red' if d < 0 else 'green' for d in differences]
        ax.bar(range(len(sensor_columns)), differences, color=colors_diff, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel(f"Variance Difference\n(Client {cv['client_id']} - Global Avg)", fontsize=10)
        ax.set_title(f"Client {cv['client_id']} Sensor Variance vs Global Average", 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(sensor_columns)))
        ax.set_xticklabels(sensor_columns, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "sensor_variance_differences.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    # 4. Summary statistics
    print("\nSensor Variance Summary Statistics:")
    print("=" * 80)
    print(f"{'Client':<10} {'Units':<10} {'Mean Variance':<15} {'Std Variance':<15} {'Min Variance':<15} {'Max Variance':<15}")
    print("-" * 80)
    for cv in client_variances:
        variances = cv['variances'].values
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        min_var = np.min(variances)
        max_var = np.max(variances)
        print(f"Client {cv['client_id']:<5} {cv['num_units']:<10} {mean_var:<15.4f} "
              f"{std_var:<15.4f} {min_var:<15.4f} {max_var:<15.4f}")
    print("=" * 80)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Federated Client Data Visualization")
    print("=" * 60)
    
    # Load all training data
    try:
        all_data_df = load_all_training_data(args.data_dir)
    except Exception as e:
        print(f"\nError loading training files: {e}")
        return 1
    
    # Split dataset into partitions (same as run_federated.py)
    try:
        data_partitions = split_dataset_into_partitions(all_data_df, args.num_clients)
        print(f"\nDataset split into {len(data_partitions)} partitions")
    except Exception as e:
        print(f"\nError splitting dataset: {e}")
        return 1
    
    # Compute RUL for each partition
    partitions_with_rul = compute_rul_for_partitions(data_partitions)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize RUL distributions
    visualize_rul_distributions(partitions_with_rul, args.output_dir)
    
    # Visualize sensor variance differences
    visualize_sensor_variance(partitions_with_rul, args.output_dir)
    
    print(f"\n" + "=" * 60)
    print(f"Visualization complete! Outputs saved to: {args.output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
