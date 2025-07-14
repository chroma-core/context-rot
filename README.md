# Context Rot: How Increasing Input Tokens Impacts LLM Performance

This repository contains the toolkit for replicating results from our [technical report](https://research.trychroma.com/context-rot).

## Motivation

Large Language Models (LLMs) are typically presumed to process context uniformly—that is, the model should handle the 10,000th token just as reliably as the 100th. However, in practice, this assumption does not hold. We observe that model performance varies significantly as input length changes, even on simple tasks.

<p align="center">
  <img src="images/image.png" alt="repeated words results" width="1000"/><br>
  <span style="font-size: 1em; color: #555;">Latest Models on Repeated Words Task</span>
</p>

## Experiments

Our experiments are organized under the `experiments/` folder:

### 1. **NIAH Extension** (`experiments/niah_extension/`)
Extension of [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) to examine the effects of needles with semantic, rather than direct lexical matches, as well as the effects of introducing variations to the haystack content. 

### 2. **LongMemEval** (`experiments/longmemeval/`)
[LongMemEval](https://arxiv.org/abs/2410.10813) task.

### 3. **Repeated Words** (`experiments/repeated_words/`)
Tests model performance on replicating a sequence of repeated words.

Each experiment contains detailed instructions in their respective `README.md` files.

## Data

Datasets can be downloaded [here](https://drive.google.com/drive/folders/1FuOysriSotnYasJUbZJzn31SWt85_3yf?usp=drive_link).

## Quick Start

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables:
   - **OpenAI**: `OPENAI_API_KEY`
   - **Anthropic**: `ANTHROPIC_API_KEY`
   - **Google**: `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_MODEL_PATH`

5. Navigate to specific experiment folder and follow README instructions

## Citation
If you find this work useful, please cite our technical report:
```
@techreport{hong2025context,
  title = {Context Rot: How Increasing Input Tokens Impacts LLM Performance},
  author = {Hong, Kelly and Troynikov, Anton and Huber, Jeff},
  year = {2025},
  month = {July},
  institution = {Chroma},
  url = {https://research.trychroma.com/context-rot},
}
```
