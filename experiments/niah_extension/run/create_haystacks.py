import argparse
import sys
import os
import glob
import random
import tiktoken
import pandas as pd
from tqdm import tqdm


def load_text_files(haystack_folder: str) -> list[str]:
    txt_files = glob.glob(os.path.join(haystack_folder, "*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {haystack_folder}")
    
    texts = []
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.append(f.read().strip())
    
    print(f"Loaded {len(texts)} text files from {haystack_folder}")
    return texts


def build_haystack_sequential(texts: list[str], target_tokens: int, tokenizer) -> str:
    haystack = ""
    text_index = 0
    
    while len(tokenizer.encode(haystack)) < target_tokens:
        next_text = texts[text_index % len(texts)]
        test_haystack = haystack + next_text + "\n\n"
        
        test_tokens = len(tokenizer.encode(test_haystack))
        if test_tokens > target_tokens:
            current_tokens = len(tokenizer.encode(haystack))
            remaining_tokens = target_tokens - current_tokens
            
            if remaining_tokens > 0:
                text_tokens = tokenizer.encode(next_text + "\n\n")
                truncated_tokens = text_tokens[:remaining_tokens]
                haystack += tokenizer.decode(truncated_tokens)
            break
        else:
            haystack = test_haystack
            
        text_index += 1
    
    return haystack


def build_haystack_shuffled(texts: list[str], target_tokens: int, tokenizer) -> str:
    all_chunks = []
    for text in texts:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for sentence in sentences:
            sentence_text = sentence + ". "
            all_chunks.append({
                'text': sentence_text,
                'token_count': len(tokenizer.encode(sentence_text))
            })
    
    available_chunks = all_chunks.copy()
    random.shuffle(available_chunks)
    
    context_parts = []
    current_tokens = 0
    chunk_index = 0
    
    while current_tokens < target_tokens:
        if chunk_index >= len(available_chunks):
            random.shuffle(available_chunks)
            chunk_index = 0
        
        chunk = available_chunks[chunk_index]
        chunk_text = chunk['text']
        chunk_tokens = chunk.get('token_count', len(tokenizer.encode(chunk_text)))
        
        if current_tokens + chunk_tokens > target_tokens:
            if current_tokens > 0:
                break
            tokens_needed = target_tokens
            chunk_tokens_list = tokenizer.encode(chunk_text)
            truncated_tokens = chunk_tokens_list[:tokens_needed]
            chunk_text = tokenizer.decode(truncated_tokens)
            current_tokens = tokens_needed
            context_parts.append(chunk_text)
            break
        
        context_parts.append(chunk_text)
        current_tokens += chunk_tokens
        chunk_index += 1
    
    return " ".join(context_parts)


def insert_needle_at_depth(haystack: str, needle: str, depth_percent: float, tokenizer) -> str:
    haystack_tokens = tokenizer.encode(haystack)
    needle_tokens = tokenizer.encode(needle)
    
    if depth_percent == 100:
        new_tokens = haystack_tokens + needle_tokens
    elif depth_percent == 0:
        new_tokens = needle_tokens + haystack_tokens
    else:
        insertion_point = int(len(haystack_tokens) * (depth_percent / 100))
        
        period_token = tokenizer.encode('.')[0]
        while insertion_point > 0 and haystack_tokens[insertion_point - 1] != period_token:
            insertion_point -= 1
        
        new_tokens = haystack_tokens[:insertion_point] + needle_tokens + haystack_tokens[insertion_point:]
    
    return tokenizer.decode(new_tokens)


def insert_distractors_randomly(haystack: str, distractors: list[str]) -> str:
    if not distractors:
        return haystack
    
    sentences = haystack.split('. ')
    if len(sentences) < 2:
        return haystack
    
    result_sentences = sentences.copy()
    
    for distractor in distractors:
        if len(result_sentences) > 1:
            insert_pos = random.randint(1, len(result_sentences) - 1)
            clean_distractor = distractor.rstrip('.')
            result_sentences.insert(insert_pos, f" {clean_distractor}")
    
    return '. '.join(result_sentences)


def create_niah_prompt(haystack_with_needle: str, retrieval_question: str) -> str:
    system_template = f"""You are a helpful AI bot that answers questions for a user. Keep your response short and direct

    <document_content>
    {haystack_with_needle}
    <document_content>

    Here is the user question:
    <question>
    {retrieval_question}
    <question>
    
    Don't give information outside the document or repeat your findings.
    Assistant: Here is the most relevant information in the documents:
    """
    
    return system_template


def create_haystacks(haystack_folder: str, needle: str, shuffled: bool, output_folder: str, 
                    question: str, distractors: list[str] = None):
    os.makedirs(output_folder, exist_ok=True)
    tokenizer = tiktoken.get_encoding("o200k_base")
    
    texts = load_text_files(haystack_folder)
    
    input_lengths = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 900_000]
    depths = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    sample_prompt = create_niah_prompt("SAMPLE_CONTEXT", question)
    overhead_tokens = len(tokenizer.encode(sample_prompt.replace("SAMPLE_CONTEXT", "")))
    
    results = []
    
    print(f"Creating {'shuffled' if shuffled else 'sequential'} prompts...")
    if distractors:
        print(f"Adding {len(distractors)} distractors to haystacks")
    
    for input_length in tqdm(input_lengths, desc="Input lengths"):
        needle_tokens = len(tokenizer.encode(needle))
        available_context_tokens = input_length - overhead_tokens - needle_tokens
        
        if available_context_tokens <= 100:
            print(f"Skipping input length {input_length} - too small for needle and overhead")
            continue
        
        if shuffled:
            base_haystack = build_haystack_shuffled(texts, available_context_tokens, tokenizer)
        else:
            base_haystack = build_haystack_sequential(texts, available_context_tokens, tokenizer)
        
        for depth in depths:
            haystack_with_distractors = insert_distractors_randomly(base_haystack, distractors)
            haystack_with_needle = insert_needle_at_depth(haystack_with_distractors, needle, depth, tokenizer)
            
            full_prompt = create_niah_prompt(haystack_with_needle, question)
            
            actual_tokens = len(tokenizer.encode(full_prompt))
            
            results.append({
                'token_count': actual_tokens,
                'approximate_input_length': input_length,
                'needle_depth': depth,
                'prompt': full_prompt,
                'question': question,
                'answer': needle
            })
    
    results_df = pd.DataFrame(results)
    mode = "shuffled" if shuffled else "sequential"
    distractor_suffix = "_with_distractors" if distractors else ""
    output_path = os.path.join(output_folder, f"niah_prompts_{mode}{distractor_suffix}.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Created {len(results)} NIAH prompts")
    print(f"Results saved to {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Create NIAH prompts with needles for experiments')
    
    parser.add_argument('--haystack-folder', type=str, required=True,
                       help='Folder containing .txt files to use as haystack content')
    parser.add_argument('--needle', type=str, required=True,
                       help='Needle text to insert into haystacks')
    parser.add_argument('--question', type=str, required=True,
                       help='Question to ask about the needle')
    parser.add_argument('--shuffled', action='store_true',
                       help='Use shuffled mode (randomize sentence order)')
    parser.add_argument('--output-folder', type=str, required=True,
                       help='Output folder for generated CSV file')
    parser.add_argument('--distractors', type=str, nargs='*', default=None,
                       help='Optional distractor strings to randomly insert into haystacks')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.haystack_folder):
            raise ValueError(f"Haystack folder does not exist: {args.haystack_folder}")
        
        if not args.needle.strip():
            raise ValueError("Needle cannot be empty")
            
        if not args.question.strip():
            raise ValueError("Question cannot be empty")
        
        distractors = [d.strip() for d in args.distractors if d.strip()] if args.distractors else None
        
        create_haystacks(
            haystack_folder=args.haystack_folder,
            needle=args.needle,
            shuffled=args.shuffled,
            output_folder=args.output_folder,
            question=args.question,
            distractors=distractors
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()  