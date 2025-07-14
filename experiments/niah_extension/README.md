# Needle in a Haystack Extension

We extend the standard [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) task, to investigate model behavior in previously underexplored settings. We examine the effects of needles with semantic, rather than direct lexical matches, as well as the effects of introducing variations to the haystack content. 

PG essays, as well as the specific needles and distractors used in the technical report can be downloaded from [here](https://drive.google.com/drive/folders/14uHYF65yu7cNGANungZX1NRboqwHHuVB?usp=sharing).

The arXiv dataset used in the report can be downloaded from [here](https://huggingface.co/datasets/jamescalam/ai-arxiv2).

## Environment Variables

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_MODEL_PATH`

## Usage

### 1. Create Haystacks

Generate NIAH prompts with needle at various depths and context lengths:

```bash
cd experiments/niah_extension
python run/create_haystacks.py \
  --haystack-folder ../../data/PaulGrahamEssays \
  --needle "It sometimes surprises people when I tell them I write every week. I was also surprised when my friend from my freshman year History course was doing the same thing, but looking back, I only wish I started earlier." \
  --question "What was the best writing advice I got from my college classmate?" \
  --output-folder ../../data/niah_prompts 
```

This creates haystacks that preserve the original structure of PG essays with a low-similarity needle-question pair.

**Parameters:**
- `--haystack-folder`: Directory containing .txt files for context
- `--needle`: Text to insert (the needle)
- `--question`: Question about the needle
- `--output-folder`: Output directory for generated CSV
- `--shuffled`: Randomize sentence order (optional)
- `--distractors`: Optional distractor strings


### 2. Run Inference

```bash
python run/run_niah_extension.py \
    --provider openai \
    --input-path ../../data/niah_prompts/niah_prompts_sequential.csv \
    --output-path ../../results/gpt_4_1_niah_results.csv \
    --input-column prompt \
    --output-column output \
    --model-name gpt-4.1-2025-04-14 \
    --max-context-length 1_047_576  \
    --max-tokens-per-minute 2_000_000
```

**Parameters:**
- `--provider`: Provider to use (openai, anthropic, google)
- `--input-path`: Path to input CSV from step 1
- `--output-path`: Output CSV path
- `--model-name`: Model identifier
- `--max-context-length`: Maximum context length in tokens
- `--max-tokens-per-minute`: Rate limiting

### 3. Evaluate Results

Use LLM judge to evaluate response correctness:

```bash
python evaluate/evaluate_niah_extension.py \
  --input-path ../../results/gpt_4_1_niah_results.csv \
  --output-path ../../results/gpt_4_1_niah_evaluated.csv \
  --model-name gpt-4.1-2025-04-14 \
  --output-column output \
  --question-column question \
  --correct-answer-column answer
```

**Parameters:**
- `--input-path`: Results CSV from step 2
- `--output-path`: Output CSV with evaluation
- `--model-name`: Judge model (default: gpt-4.1-2025-04-14)
- `--output-column`: Column containing model responses
- `--question-column`: Column containing questions
- `--correct-answer-column`: Column containing correct answers

**Output:** CSV with additional `llm_judge_output` column (true/false)

### 4. Visualize Performance

Generate heatmap showing accuracy across context lengths and needle depths:

```bash
python evaluate/visualize.py \
  --csv-path ../../results/gpt_4_1_niah_evaluated.csv \
  --output-path ../../results/gpt_4_1_heatmap.png \
  --title "NIAH Performance - GPT-4.1"
```

**Parameters:**
- `--csv-path`: Evaluated results CSV
- `--output-path`: Output image path (optional)
- `--title`: Custom heatmap title (optional)

### Analyze Distractors

If you choose to add distractors, analyze which distractors the model selected by:

```bash
python evaluate/analyze_distractors.py \
  --input-path ../../results/gpt_4_1_niah_distractors_evaluated.csv \
  --output-path ../../results/gpt_4_1_niah_distractors_analyzed.csv \
  --visual-path ../../results/gpt_4_1_niah_distractors_analyzed.png \
  --model-name gpt-4.1-2025-04-14 \
  --distractors-file ../../data/pg_distractors.json
```

**Parameters:**
- `--input-path`: Results CSV from step
- `--output-path`: Output CSV with distractor analysis
- `--distractors-file`: JSON file containing distractor options
- `--model-name`: Judge model

### Sample Distractors

`../../data/pg_distractors.json` contains example distractors for testing:

```json
{
  "distractor_0": {
    "distractor": "The best writing tip I received from my college professor was to write everyday.",
    "rewrite_for_analysis": "write everyday"
  },
  "distractor_1": {
    "distractor": "The worst writing advice I got from my college classmate was to write each essay in five different styles.",
    "rewrite_for_analysis": "write each essay in five different styles"
  }
}
```

`rewrite_for_analysis` is a simplified version of the distractors for LLM judging.