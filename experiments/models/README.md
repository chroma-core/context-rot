# Models

We support OpenAI, Anthropic, and Google providers. While Qwen is included in our evaluation results, we don't provide a Qwen implementation due to the specificity in custom deployments.

## File Structure

```
models/
├── README.md
├── base_provider.py          # Abstract base class for all providers
├── llm_judge.py             # LLM judge for evaluation
└── providers/
    ├── openai.py            # OpenAI provider implementation
    ├── anthropic.py         # Anthropic provider implementation  
    └── google.py            # Google provider implementation
```

## Environment Variables

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_MODEL_PATH`

## Adding New Providers

To add a new provider, inherit from `BaseProvider` and implement:
- `process_single_prompt()`: Process a single prompt
- `get_client()`: Initialize API client