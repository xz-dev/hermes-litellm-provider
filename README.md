# Hermes LiteLLM Provider

Hermes model-provider plugin for a local or remote LiteLLM proxy.

## Features

- Uses LiteLLM's OpenAI-compatible `/v1` API.
- Discovers models from `/v1/model/info` first, then falls back to `/v1/models`.
- Filters obvious non-chat models such as embeddings and image-generation models.
- Preserves LiteLLM reasoning effort values, including `minimal`.
- Sends `reasoning_effort` only when LiteLLM metadata reports support for it.

## Install

Hermes discovers model-provider plugins from:

```bash
~/.hermes/plugins/model-providers/<provider-name>/
```

Install this provider as `litellm`:

```bash
mkdir -p ~/.hermes/plugins/model-providers/litellm
cp __init__.py plugin.yaml ~/.hermes/plugins/model-providers/litellm/
hermes gateway restart
```

`hermes plugins install owner/repo` is the standard installer for general Hermes
plugins. Model-provider discovery is path-based, so this provider currently uses
the model-provider directory directly.

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

## Development

Keep this repository as the source of truth, then deploy to a remote Hermes host
only for testing:

```bash
make check
make deploy REMOTE=xz@100.65.231.22
make restart REMOTE=xz@100.65.231.22
make test REMOTE=xz@100.65.231.22 MODEL=glm-5.1
```

The deploy target copies only `__init__.py` and `plugin.yaml` into the remote
model-provider directory and clears the remote bytecode cache.

## Notes

### Context Length

Hermes' current `ProviderProfile.fetch_models()` interface only returns model
IDs (`list[str]`). This plugin can use `/v1/model/info` to improve model
matching and filtering, but it cannot return `max_input_tokens` or context
length metadata back through that interface.

Context length should be resolved in Hermes core's model metadata layer. A good
resolution order is:

1. Explicit user config.
2. LiteLLM `/v1/model/info` fields such as `max_input_tokens` or `max_tokens`.
3. `models.dev`, using `litellm_params.model` when available.
4. Provider-specific defaults.
5. Hermes fallback default.

This keeps the plugin focused on provider registration, model discovery, and
request-time compatibility while leaving context sizing to the component that
can actually apply it.

### Ollama Cloud Routing

For Ollama Cloud behind LiteLLM, prefer an OpenAI-compatible route:

```yaml
model: openai/*
api_base: https://ollama.com/v1
```

Avoid `ollama/* + https://ollama.com` for models where structured tool calls must be preserved.
