import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualize_longmemeval_results(focused_filepath: str, full_filepath: str, model_name: str, output_path: str):
    focused_df = pd.read_csv(focused_filepath)
    full_df = pd.read_csv(full_filepath)
    
    focused_mean = focused_df['llm_judge_output'].mean()
    full_mean = full_df['llm_judge_output'].mean()
    
    focused_color = "#EB4026"
    full_color = "#3A76E5"
    
    plt.figure(figsize=(8, 6))
    
    bars = plt.bar(['Focused', 'Full'], [focused_mean, full_mean], color=[focused_color, full_color])
    plt.ylim(0, 1)
    plt.ylabel('Average Score')
    plt.title(f'LongMemEval Overall Performance - {model_name}')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize LongMemEval results')
    
    parser.add_argument('--focused-path', type=str, required=True,
                       help='Path to focused results CSV file')
    parser.add_argument('--full-path', type=str, required=True,
                       help='Path to full results CSV file')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name for plot titles')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Output path for PNG file')
    
    args = parser.parse_args()
    
    try:
        visualize_longmemeval_results(
            focused_filepath=args.focused_path,
            full_filepath=args.full_path,
            model_name=args.model_name,
            output_path=args.output_path
        )
        print(f"Visualization saved to: {args.output_path}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()