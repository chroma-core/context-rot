import argparse
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein
import dotenv

dotenv.load_dotenv()

def normalized_levenshtein_score(gold: str, pred: str) -> float:
    if not gold or not pred:
        return 0.0
    
    distance = Levenshtein.distance(gold, pred)
    max_len = max(len(gold), len(pred))
    return 1 - (distance / max_len)

def check_delta(output: str, gold: str) -> int:
    return len(gold.split()) - len(output.split())

def modified_word_present(row: pd.Series, modified_word: str, output_column: str) -> bool:
    if pd.isna(row[output_column]):
        return False
    
    if row['index'] == row['num_words'] - 1:
        unique_word = " " + modified_word
    else:
        unique_word = modified_word + " "
    
    return unique_word in row[output_column]

def check_correct_position(row: pd.Series, modified_word: str, output_column: str) -> bool:
    if not row['modified_word_present'] or pd.isna(row[output_column]):
        return False
    
    try:
        if row['index'] == row['num_words'] - 1:
            unique_word = " " + modified_word
        else:
            unique_word = modified_word + " "
        
        gold_index = row["gold"].index(unique_word)
        output_index = row[output_column].index(unique_word)
        return gold_index == output_index
    except ValueError:
        return False

def other_word_exists(output: str, common_word: str, modified_word: str) -> bool:
    if pd.isna(output):
        return True
    
    words = output.split()
    
    for i, word in enumerate(words):
        if word not in [common_word, modified_word]:
            return True
    
    return False

def filter_refusals(df: pd.DataFrame, common_word: str, modified_word: str, output_column: str):
    other_word_present_list = []
    
    for i, row in df.iterrows():
        if pd.isna(row[output_column]):
            other_word_present = True
        else:
            other_word_present = other_word_exists(row[output_column], common_word, modified_word)
        
        other_word_present_list.append(other_word_present)
    
    other_word_df = df[other_word_present_list]
    
    if len(other_word_df) > 0:
        df_refusals = other_word_df[
            (pd.isna(other_word_df[output_column])) | 
            (other_word_df[output_column].str.count(common_word) < 15)
        ]
    else:
        df_refusals = other_word_df
    
    print(f"Number of refusals: {len(df_refusals)} out of {len(df)}, {len(df_refusals)/len(df) * 100:.1f}%")
    
    return df_refusals if len(df_refusals) > 0 else None

def create_binned_plot(df: pd.DataFrame, unique_num_words: list[int], metric_column: str, 
                      ylabel: str, title: str, color: str, output_path: str):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, num_words in enumerate(unique_num_words):
        df_subset = df[df["num_words"] == num_words].copy()
        
        if metric_column == 'correct_position':
            other_word_mask = []
            for _, row in df_subset.iterrows():
                other_word_mask.append(other_word_exists(row['output'], df['common_word'].iloc[0], 
                                                       df['modified_word'].iloc[0]))
            df_subset = df_subset[~np.array(other_word_mask)]
            df_subset = df_subset[df_subset['modified_word_present']]
        
        bins = np.linspace(0, num_words-1, 21)
        bin_values = []
        
        for j in range(20):
            mask = (df_subset["index"] >= bins[j]) & (df_subset["index"] < bins[j+1])
            if mask.any():
                if metric_column == 'delta':
                    bin_values.append(df_subset[mask][metric_column].mean())
                else:
                    bin_values.append(df_subset[mask][metric_column].mean())
            else:
                bin_values.append(0)
        
        x_positions = np.linspace(0, 100, 20)
        
        if metric_column == 'delta':
            for idx, val in enumerate(bin_values):
                xpos = x_positions[idx]
                if val < 0:
                    axes[i].bar(xpos, val, color=color, alpha=0.7, width=4, hatch='///', 
                               edgecolor='white', linewidth=0.8)
                else:
                    axes[i].bar(xpos, val, color=color, alpha=0.7, width=4)
            axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
        else:
            axes[i].bar(x_positions, bin_values, color=color, alpha=0.7, width=4)
        
        axes[i].set_title(f'{num_words} words')
        axes[i].set_ylabel(ylabel)
        axes[i].set_xlabel('Modified Word Position (%)')
        axes[i].set_xlim(-5, 105)
        
        if metric_column == 'delta':
            min_val = min(bin_values)
            max_val = max(bin_values)
            pad = (max_val - min_val) * 0.1 if max_val != min_val else 1
            axes[i].set_ylim(min_val - pad, max_val + pad)
        else:
            axes[i].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=18, y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_token_count_plot(df: pd.DataFrame, model_name: str, common_word: str, 
                           modified_word: str, output_path: str):
    color = "#90B8B6"
    
    plt.figure(figsize=(10, 6))
    
    num_bins = 12
    min_token = max(df["token_count"].min(), 1)
    bins = np.logspace(np.log10(min_token), np.log10(df["token_count"].max()), num_bins + 1)
    df["token_bin"] = pd.cut(df["token_count"], bins=bins, include_lowest=True, labels=False)
    
    bin_centers = []
    avg_scores = []
    for bin_idx in range(num_bins):
        bin_df = df[df["token_bin"] == bin_idx]
        if not bin_df.empty:
            avg_scores.append(bin_df["levenshtein_score"].mean())
            left = bins[bin_idx]
            right = bins[bin_idx + 1]
            bin_centers.append(np.sqrt(left * right))
    
    plt.plot(bin_centers, avg_scores, marker='o', linestyle='-', color=color)
    plt.xscale('log')
    plt.xlabel('Input Length (Tokens)')
    plt.ylabel('Average Normalized Levenshtein Score')
    plt.title(f'Repeated "{common_word}", one "{modified_word}" - {model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_repeated_words(input_path: str, output_dir: str, common_word: str, 
                           modified_word: str, model_name: str):
    df = pd.read_csv(input_path)
    
    df["num_words"] = df["id"].str.split("_").str[0].astype(int)
    df["index"] = df["id"].str.split("_").str[1].astype(int)
    
    levenshtein_scores = []
    for i, row in df.iterrows():
        score = normalized_levenshtein_score(row["gold"], row["output"])
        levenshtein_scores.append(score)
    
    df["levenshtein_score"] = levenshtein_scores
    
    df["modified_word_present"] = df.apply(
        lambda row: modified_word_present(row, modified_word, "output"), axis=1
    )
    
    df["correct_position"] = df.apply(
        lambda row: check_correct_position(row, modified_word, "output"), axis=1
    )
    
    df["delta"] = df.apply(
        lambda row: check_delta(row["output"], row["gold"]), axis=1
    )
    
    df["common_word"] = common_word
    df["modified_word"] = modified_word
    
    refusals = filter_refusals(df, common_word, modified_word, "output")
    if refusals is not None:
        filtered_df = df[~df["output"].isin(refusals["output"])]
    else:
        filtered_df = df
    
    print(f"Length of filtered df: {len(filtered_df)}")
    
    unique_num_words = sorted(filtered_df["num_words"].unique())
    
    os.makedirs(output_dir, exist_ok=True)
    
    create_token_count_plot(filtered_df, model_name, common_word, modified_word,
                           os.path.join(output_dir, "token_count_performance.png"))
    
    create_binned_plot(filtered_df, unique_num_words, "levenshtein_score",
                      "Normalized Levenshtein Score", 
                      f"Normalized Levenshtein Score - {model_name}",
                      "#2FB874", os.path.join(output_dir, "levenshtein_score.png"))
    
    create_binned_plot(filtered_df, unique_num_words, "modified_word_present",
                      "Modified Word Present", 
                      f"Modified Word Present - {model_name}",
                      "#EA5412", os.path.join(output_dir, "modified_word_present.png"))
    
    create_binned_plot(filtered_df, unique_num_words, "correct_position",
                      "Position Accuracy", 
                      f"Position Accuracy - {model_name}",
                      "#EBB125", os.path.join(output_dir, "position_accuracy.png"))

    create_binned_plot(filtered_df, unique_num_words, "delta",
                      "num_words - output length", 
                      f"Number of Words Delta - {model_name}",
                      "#7E8E9E", os.path.join(output_dir, "word_count_delta.png"))
    
    filtered_df.to_csv(os.path.join(output_dir, "evaluated_results.csv"), index=False)
    
    summary_scores = {}
    for num_words in unique_num_words:
        df_subset = filtered_df[filtered_df["num_words"] == num_words]
        summary_scores[num_words] = df_subset["levenshtein_score"].mean()
    
    print(f"Summary scores by word count: {summary_scores}")
    print(f"Overall average Levenshtein score: {filtered_df['levenshtein_score'].mean():.4f}")
    
    return filtered_df, summary_scores

def main():
    parser = argparse.ArgumentParser(description='Evaluate repeated words experiment results')
    
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to input CSV file with model outputs')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save evaluation results and plots')
    parser.add_argument('--common-word', type=str, required=True,
                       help='Common word that was repeated')
    parser.add_argument('--modified-word', type=str, required=True,
                       help='Modified word that was inserted')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name for plot titles')
    
    args = parser.parse_args()
    
    try:
        evaluate_repeated_words(
            input_path=args.input_path,
            output_dir=args.output_dir,
            common_word=args.common_word,
            modified_word=args.modified_word,
            model_name=args.model_name
        )
        
        print(f"Evaluation complete. Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()