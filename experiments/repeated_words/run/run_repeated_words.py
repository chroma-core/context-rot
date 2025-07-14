import argparse
import sys
import os
import pandas as pd
import tiktoken
from tqdm import tqdm
import dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.providers.openai import OpenAIProvider
from models.providers.anthropic import AnthropicProvider
from models.providers.google import GoogleProvider

dotenv.load_dotenv()


def create_variations(common_word: str, modified_word: str, num_words: int) -> tuple[list[str], list[str], pd.DataFrame]:
    if num_words < 100:
        indices = list(range(num_words))
    else:
        step = num_words // 100
        indices = list(range(0, num_words, step))
        if indices[-1] != num_words - 1:
            indices.append(num_words - 1)

    variations = [
        " ".join([modified_word if j == i else common_word for j in range(num_words)])
        for i in indices
    ]
    ids = [str(i) for i in indices]
    df = pd.DataFrame({"index": ids, "gold": variations})
    return variations, ids, df


def create_input_df(common_word: str, modified_word: str, model_max_output_tokens: int) -> pd.DataFrame:
    num_word_variations = [25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]

    custom_ids = []
    prompts = []
    gold = []
    input_tokens_list = []
    max_output_tokens_list = []

    encoding = tiktoken.get_encoding("o200k_base")

    for num_words in num_word_variations:
        variations, ids, df = create_variations(common_word, modified_word, num_words)

        for id, variation in zip(ids, variations):
            prompt = f"Simply replicate the following text, output the exact same text: {variation}"

            input_tokens = len(encoding.encode(prompt, disallowed_special=()))
            
            max_output_tokens = input_tokens * 2

            if max_output_tokens > model_max_output_tokens:
                print(f"Output tokens ({max_output_tokens}) exceeds max output tokens ({model_max_output_tokens})")
                break

            input_tokens_list.append(input_tokens)
            max_output_tokens_list.append(max_output_tokens)
            custom_id = f"{num_words}_{id}"
            custom_ids.append(custom_id)
            prompts.append(prompt)
            gold.append(variation)

    df = pd.DataFrame({
        "id": custom_ids, 
        "prompt": prompts, 
        "gold": gold, 
        "token_count": input_tokens_list, 
        "max_output_tokens": max_output_tokens_list
    })

    return df


def get_provider(provider_name: str, model_name: str = None):
    if provider_name.lower() == 'openai':
        return OpenAIProvider()
    elif provider_name.lower() == 'anthropic':
        return AnthropicProvider()
    elif provider_name.lower() == 'google':
        return GoogleProvider(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Available providers: openai, anthropic, google")


def main():
    parser = argparse.ArgumentParser(description='Run repeated words experiment')
    
    parser.add_argument('--provider', type=str, required=True, 
                       choices=['openai', 'anthropic', 'google'],
                       help='Provider to use')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Output path for results')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model to run')
    parser.add_argument('--common-word', type=str, required=True,
                       help='Common word to repeat')
    parser.add_argument('--modified-word', type=str, required=True,
                       help='Modified word to insert')
    parser.add_argument('--model-max-output-tokens', type=int, required=True,
                       help='Maximum output tokens for the model')
    parser.add_argument('--max-context-length', type=int, required=True,
                       help='Maximum context length in tokens')
    parser.add_argument('--max-tokens-per-minute', type=int, required=True,
                       help='Maximum tokens per minute for rate limits')
    
    args = parser.parse_args()
    
    try:
        print(f"Creating input data for {args.common_word} | {args.modified_word}")
        input_df = create_input_df(args.common_word, args.modified_word, args.model_max_output_tokens)

        input_path = os.path.join(f"../../data/repeated_words_input_{args.common_word}_{args.modified_word}.csv")
        input_df.to_csv(input_path, index=False)
        print(f"Input data saved to: {input_path}")
        
        provider = get_provider(args.provider, args.model_name)
        
        print(f"Running {args.provider} provider with {args.model_name}")
        provider.main(
            input_path=input_path,
            output_path=args.output_path,
            input_column='prompt',
            output_column='output',
            model_name=args.model_name,
            max_context_length=args.max_context_length,
            max_tokens_per_minute=args.max_tokens_per_minute
        )
        
        print(f"Results saved to: {args.output_path}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()