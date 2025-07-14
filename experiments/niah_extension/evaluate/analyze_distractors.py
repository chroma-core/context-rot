import argparse
import sys
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dotenv

dotenv.load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.llm_judge import LLMJudge

def create_histogram_for_file(csv_path: str, visual_output_path: str = None, model_name: str = None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['distractor_label'])
    
    df['distractor_label'] = df['distractor_label'].astype(str)
    
    plt.figure(figsize=(10, 6))
    
    value_counts = df['distractor_label'].value_counts().sort_index()
    normalized_counts = value_counts / value_counts.sum()
    
    plt.bar(normalized_counts.index.astype(str), normalized_counts.values, 
            alpha=0.7, edgecolor='none')
    plt.xlabel('Distractor Selection')
    plt.ylabel('Proportion')
    plt.title(f'Distribution of Hallucinations for {model_name}')
    plt.grid(axis='y', alpha=0.3)
    
    if visual_output_path:
        plt.savefig(visual_output_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {visual_output_path}")
    plt.close()
    
    print(f"Total valid samples: {len(df)}")
    print(f"Value counts:\n{df['distractor_label'].value_counts().sort_index()}")
    
    return df

def main():
    DEFAULT_PROMPT = """
        Based on this model output, which response does it give to: {question}
        
        Choose the option that most closely aligns, or go with none of the above (-1) if none align or multiple options are mentioned. 
        If an option is mentioned but it is not definitive and there is hesitation, mentioning "however" or something that indicates slight hesitation, go with none of the above (-1).
        If it is mentioned that there is inconsistency in the document, go with none of the above (-1).

        Options:
        {distractors}

        Model Output: {output}

        Again, if multiple options are mentioned or there is any hesitation, go with none of the above (-1).

        Instructions: Output the number and number only. If none of the specific options are mentioned, output -1.
        """
         
      
    parser = argparse.ArgumentParser(description='Analyze NIAH distractors using LLM judge')
    
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT,
                       help='Judge prompt template (use {output}, {question}, {correct_answer}, {distractors} as placeholders)')
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to output CSV file')
    parser.add_argument('--visual-path', type=str, required=True,
                       help='Path to visual output file')
    parser.add_argument('--model-name', type=str, default='gpt-4.1-2025-04-14',
                       help='Model name to use (default: gpt-4.1-2025-04-14)')
    parser.add_argument('--output-column', type=str, default='output',
                       help='Column name containing model outputs (default: output)')
    parser.add_argument('--question-column', type=str, default='question',
                       help='Column name containing questions (default: question)')
    parser.add_argument('--correct-answer-column', type=str, default='answer',
                       help='Column name containing correct answers (default: answer)')
    parser.add_argument('--max-context-length', type=int, default=1_047_576,
                       help='Maximum context length in tokens (default: 1_047_576)')
    parser.add_argument('--max-tokens-per-minute', type=int, default=2_000_000,
                       help='Maximum tokens per minute for rate limiting (default: 2_000_000)')
    parser.add_argument('--distractors-file', type=str, default=None,
                       help='Path to JSON file containing distractors')
    args = parser.parse_args()
    
    try:
        judge = LLMJudge(
            prompt=args.prompt,
            model_name=args.model_name,
            output_column=args.output_column,
            question_column=args.question_column,
            correct_answer_column=args.correct_answer_column,
            distractors_file=args.distractors_file
        )
        
        judge.analyze_distractors(
            input_path=args.input_path,
            output_path=args.output_path,
            max_context_length=args.max_context_length,
            max_tokens_per_minute=args.max_tokens_per_minute
        )

        create_histogram_for_file(args.output_path, args.visual_path, args.model_name)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()