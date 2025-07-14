import argparse
import sys
import os
import dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.llm_judge import LLMJudge

dotenv.load_dotenv()

def main():
    DEFAULT_PROMPT = """
        Given this question and the CORRECT answer, determine whether the response is correct (meaning it factually aligns with the correct answer). 
        You must only respond with "true" or "false".
        If the response is partially incorrect, such as a typo, respond with "false".
        If the repsonse contains a snippet of text or additional supporting information, while still maintaining the correct answer without changing the meaning, respond with "true".
        If the response starts with anything like "here is the most relevant information in the documents: ", respond with "true". This is fine as long as the following content aligns with the correct answer.

        Question: {question}

        CORRECT answer: {correct_answer}

        Response to judge: {output}

        Instructions: Respond with only "true" if the response factually aligns with the correct answer, or "false" if it does not. Do not provide any explanation - just "true" or "false".
        """
      
    parser = argparse.ArgumentParser(description='Evaluate NIAH results using LLM judge')
    
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT,
                       help='Judge prompt template (use {output}, {question}, {correct_answer} as placeholders)')
    parser.add_argument('--input-path', type=str, required=True,
                       help='Path to input (model output) CSV file')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Path to output CSV file')
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
    
    args = parser.parse_args()
    
    try:
        judge = LLMJudge(
            prompt=args.prompt,
            model_name=args.model_name,
            output_column=args.output_column,
            question_column=args.question_column,
            correct_answer_column=args.correct_answer_column
        )
        
        judge.evaluate(
            input_path=args.input_path,
            output_path=args.output_path,
            max_context_length=args.max_context_length,
            max_tokens_per_minute=args.max_tokens_per_minute
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()