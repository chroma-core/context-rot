import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import os
import sys
from typing import Optional, Tuple

def create_niah_heatmap(csv_path: str, 
                       title: Optional[str] = None,
                       output_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 6)) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['llm_judge_output'])
    print(f"Loaded {len(df)} valid samples from {csv_path}")
    
    df['accuracy'] = df['llm_judge_output'].apply(
        lambda x: 1 if str(x).lower() == 'true' else 0
    )
    
    all_input_lengths = sorted(df['approximate_input_length'].unique())
    all_needle_depths = sorted(df['needle_depth'].unique())
    
    pivot_table = df.groupby(['approximate_input_length', 'needle_depth'])['accuracy'].mean().reset_index()
    pivot_table = pivot_table.pivot(index='needle_depth', columns='approximate_input_length', values='accuracy')
    
    heatmap_data = pd.DataFrame(
        index=all_needle_depths,
        columns=all_input_lengths,
        dtype=float
    )
    
    for depth in all_needle_depths:
        for length in all_input_lengths:
            if depth in pivot_table.index and length in pivot_table.columns:
                value = pivot_table.loc[depth, length]
                if pd.notna(value):
                    heatmap_data.loc[depth, length] = value
    
    plt.figure(figsize=figsize)
    
    colors = ['white', '#F28E2B']
    cmap = ListedColormap(colors)
    cmap.set_bad(color='lightgrey')
    
    im = plt.imshow(heatmap_data.values, 
                    cmap=cmap, 
                    aspect='auto',
                    vmin=0, vmax=1,
                    origin='lower')
    
    length_labels = []
    for length in all_input_lengths:
        if length < 1000:
            length_labels.append(str(length))
        else:
            length_labels.append(f"{int(length/1000)}K")
    
    plt.xticks(range(len(all_input_lengths)), length_labels)
    plt.yticks(range(len(all_needle_depths)), [f"{int(d)}%" for d in all_needle_depths])
    
    if title is None:
        title = f"NIAH Performance - {os.path.basename(csv_path)}"
    plt.title(title)
    plt.xlabel('Input Length (tokens)')
    plt.ylabel('Needle Depth (%)')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        print(f"Heatmap saved to: {output_path}")
    plt.close()
    
    overall_accuracy = df['accuracy'].mean()
    print(f"\nOverall Accuracy: {overall_accuracy:.3f}")
    print(f"Total Samples: {len(df)}")
    
    return df

 
def main():
    parser = argparse.ArgumentParser(description='Create NIAH performance heatmap')
    parser.add_argument('--csv-path', type=str, required=True)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    
    args = parser.parse_args()
    
    try:
        create_niah_heatmap(
            csv_path=args.csv_path,
            title=args.title,
            output_path=args.output_path
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()