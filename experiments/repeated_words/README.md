# Repeated Words

We design a controlled task in which the model must replicate a sequence of repeated words, with a single unique word inserted at a specific position. The prompt explicitly instructs the model to reproduce the input text exactly.

## Setup

Ensure you have the required dependencies and API keys configured:
- OpenAI API key: Set `OPENAI_API_KEY` environment variable
- Anthropic API key: Set `ANTHROPIC_API_KEY` environment variable  
- Google API: Set `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_MODEL_PATH` environment variables

## Usage

### Step 1: Run Inference

We use `run_repeated_words.py` to generate data and run inference for a given model and provider. We use GPT-4.1 from OpenAI as an example.

```bash
cd experiments/repeated_words
python run/run_repeated_words.py \
    --provider openai \
    --model-name gpt-4.1-2025-04-14 \
    --output-path ../../results/gpt_4_1_repeated_words_apple_apples.csv \
    --common-word apple \
    --modified-word apples \
    --model-max-output-tokens 32_768 \
    --max-context-length 1_047_576  \
    --max-tokens-per-minute 2_000_000
```
Note: this takes a while to run due to the high number of output tokens

### Step 2: Evaluate Results

Use `evaluate_repeated_words.py` to analyze model outputs and generate visualizations:

```bash
python evaluate/evaluate_repeated_words.py \
    --input-path ../../results/gpt_4_1_repeated_words_apple_apples.csv \
    --output-dir ../../results/gpt_4_1_repeated_words_apple_apples_evaluated.csv\
    --common-word apple \
    --modified-word apples \
    --model-name "GPT-4.1"
```

## Parameters

### Model Inference (`run_repeated_words.py`)
- `--provider`: LLM provider (openai, anthropic, google)
- `--model-name`: Specific model to use
- `--common-word`: Word to repeat throughout the sequence
- `--modified-word`: Word to insert at one position
- `--model-max-output-tokens`: Maximum output tokens for the model
- `--max-context-length`: Maximum context length in tokens
- `--max-tokens-per-minute`: Rate limiting

### Evaluation (`evaluate_repeated_words.py`)
- `--input-path`: Path to CSV file with model outputs
- `--output-dir`: Directory to save evaluation results and plots
- `--common-word`: Common word that was repeated
- `--modified-word`: Modified word that was inserted
- `--model-name`: Model name for plot titles

## Generated Visualizations

- `token_count_performance.png`: Performance vs input token count
- `levenshtein_score.png`: Normalized Levenshtein distance by word position
- `modified_word_present.png`: Modified word presence rates by position
- `position_accuracy.png`: Position accuracy rates by word position
- `word_count_delta.png`: Word count differences by position