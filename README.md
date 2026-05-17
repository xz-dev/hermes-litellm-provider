# Hermes LiteLLM Provider

Hermes model-provider plugin for a local or remote LiteLLM proxy.

## Features

- Uses LiteLLM's OpenAI-compatible `/v1` API.
- Discovers models from `/v1/model/info` first, then falls back to `/v1/models`.
- Filters obvious non-chat models such as embeddings and image-generation models.
- Preserves LiteLLM reasoning effort values, including `minimal`.
- Sends `reasoning_effort` only when LiteLLM metadata reports support for it.

## Install

Copy this directory to your Hermes model provider plugins directory:

```bash
mkdir -p ~/.hermes/plugins/model-providers/litellm
cp __init__.py plugin.yaml ~/.hermes/plugins/model-providers/litellm/
```

Configure Hermes to use LiteLLM:

```yaml
model:
  provider: litellm
  default: glm-5.1
  base_url: http://127.0.0.1:4000/v1
```

Configure credentials with Hermes' normal provider credential flow. The plugin
declares `LITELLM_API_KEY` as its provider key name, but reads it through Hermes'
credential resolver instead of calling `os.getenv()` directly.

```bash
hermes auth add litellm
```

Hermes environment-based credentials still work if your Hermes setup already
uses `.env`, because Hermes' resolver handles that layer. The plugin itself
defaults to `http://127.0.0.1:4000/v1` when no provider base URL is resolved.

## Test

```bash
hermes --provider litellm -m glm-5.1 -z "Reply only: OK"
```

Tool-call smoke test:

```bash
hermes --provider litellm -m glm-5.1 -t terminal -z "Use terminal to run printf ping, then answer with the command output only"
```

## Notes

For Ollama Cloud behind LiteLLM, prefer an OpenAI-compatible route:

```yaml
model: openai/*
api_base: https://ollama.com/v1
```

Avoid `ollama/* + https://ollama.com` for models where structured tool calls must be preserved.
