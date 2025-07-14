# LongMemEval

To evaluate these models in a more realistic setting, we use [LongMemEval](https://arxiv.org/abs/2410.10813), a long-context benchmark for conversational question-answering. 

Using long inputs for chat assistants is a common approach for maintaining relevant history for subsequent chats. To incorporate “memory” into a chat assistant, a naive approach would be to include the full chat history into the prompt for following chats. This requires the model to perform two tasks, typically performed in one call: find relevant parts of the conversation history (retrieval), then synthesize them in a way that is useful to an incoming query (reasoning).

In an ideal case, the model would be given only the relevant parts so it can focus solely on reasoning. Adding irrelevant context adds the additional step of identifying what is relevant, forcing the model to perform two tasks simultaneously.

We systematically test the effect of adding this additional step with increased input length through two conditions:
- Focused input, containing only the relevant parts and so the model just has to do simple reasoning.
- Full input, which utilizes the full 113k token LongMemEval input that includes irrelevant context. In this case, the model has to perform retrieval across the long context in addition to reasoning.


## Setup

### Step 1: Set API keys

Ensure you have the required dependencies and API keys configured:
- OpenAI API key: Set `OPENAI_API_KEY` environment variable
- Anthropic API key: Set `ANTHROPIC_API_KEY` environment variable  
- Google API: Set `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_MODEL_PATH` environment variables

### Step 2: Download datasets

Download the cleaned datasets [here](https://drive.google.com/drive/folders/1AS1oytdcCH3y6p-DNuaNYI7I48IFgbfe?usp=sharing), and save them under `data/`.


## Usage

### Step 1: Run Inference

We use `run_longmemeval.py` to run inference for a given model and provider. We use GPT-4.1 from OpenAI as an example.

First, run the model on the focused dataset:
```bash
cd experiments/longmemeval
python run/run_longmemeval.py \
    --provider openai \
    --input-path ../../data/cleaned_longmemeval_s_focused.csv \
    --output-path ../../results/gpt_4_1_longmemeval_focused_results.csv \
    --input-column focused_prompt \
    --output-column output \
    --model-name gpt-4.1-2025-04-14 \
    --max-context-length 1_047_576  \
    --max-tokens-per-minute 2_000_000
```

Then on the full dataset:
```bash
python run/run_longmemeval.py \
    --provider openai \
    --input-path ../../data/cleaned_longmemeval_s_full.csv \
    --output-path ../../results/gpt_4_1_longmemeval_full_results.csv \
    --input-column full_prompt \
    --output-column output \
    --model-name gpt-4.1-2025-04-14 \
    --max-context-length 1_047_576  \
    --max-tokens-per-minute 2_000_000
```

### Step 2: Evaluate Results with LLM Judge

Use the LLM judge from `evaluate_longmemeval.py` to evaluate model outputs:

Focsued dataset:
```bash
python evaluate/evaluate_longmemeval.py \
    --input-path ../../results/gpt_4_1_longmemeval_focused_results.csv \
    --output-path ../../results/gpt_4_1_longmemeval_focused_evaluated.csv \
    --model-name gpt-4.1-2025-04-14 \
    --output-column output \
    --question-column question \
    --correct-answer-column answer \
    --max-context-length 1_047_576  \
    --max-tokens-per-minute 2_000_000
```

Full dataset:
```bash
python evaluate/evaluate_longmemeval.py \
    --input-path ../../results/gpt_4_1_longmemeval_full_results.csv \
    --output-path ../../results/gpt_4_1_longmemeval_full_evaluated.csv \
    --model-name gpt-4.1-2025-04-14 \
    --output-column output \
    --question-column question \
    --correct-answer-column answer \
    --max-context-length 1_047_576  \
    --max-tokens-per-minute 2_000_000
```

### Step 3: Visualize Results

Generate plots:

```bash
python evaluate/visualize.py \
    --focused-path ../../results/gpt_4_1_longmemeval_focused_evaluated.csv \
    --full-path ../../results/gpt_4_1_longmemeval_full_evaluated.csv \
    --model-name "GPT-4.1" \
    --output-path ../../results/gpt_4_1_longmemeval.png
```

## Parameters

### Model Inference (`run_longmemeval.py`)
- `--provider`: LLM provider (openai, anthropic, google)
- `--model-name`: Specific model to use
- `--input-path`: Input CSV file path
- `--output-path`: Output CSV file path
- `--input-column`: Column containing prompts
- `--output-column`: Column for model outputs
- `--max-context-length`: Maximum context length in tokens
- `--max-tokens-per-minute`: Rate limiting

### Evaluation (`evaluate_longmemeval.py`)
- `--input-path`: Input CSV with model outputs
- `--output-path`: Output CSV with evaluations
- `--model-name`: Judge model name
- `--output-column`: Column with model outputs (default: output)
- `--question-column`: Column with questions (default: question)
- `--correct-answer-column`: Column with correct answers (default: answer)

### Visualization (`visualize.py`)
- `--focused-path`: Path to focused results CSV
- `--full-path`: Path to full results CSV
- `--model-name`: Model name for plot titles
- `--output-path`: Output PNG file path