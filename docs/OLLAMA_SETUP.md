# Setting up Ollama

Ollama supplies optional generated summaries and chat prose. Without it,
extraction, extractive summaries, indexing, search, and comparison still work;
health and response fields report the missing capability.

## Quick Start (Local Development)

First complete the [repository quick start](../README.md#quick-start).
Install Ollama using its [official platform instructions](https://ollama.com/download),
then pull the model configured by default:

```bash
ollama pull llama3.2:3b
ollama serve
```

Desktop installations may already run the Ollama service. Check it before
starting a second instance:

```bash
curl http://localhost:11434/api/tags
```

The relevant settings in [.env.example](../.env.example) are
`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT`. Start the API from the
repository root:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Inspect <http://127.0.0.1:8000/api/health>; the LLM component reports whether
the configured service/model is available. In containers, `localhost` means
the container itself; use a reachable host/service URL from that container.

## Troubleshooting

| Problem | Check |
|---|---|
| Connection refused | `ollama serve`, the configured URL, and network access from the API process/container. |
| Model absent | `ollama list`, then pull the exact name in `OLLAMA_MODEL`. |
| Model too slow or large | Choose a model suited to the machine and set its exact name; no size/latency guarantee is implied. |
| Port already in use | Identify the existing listener or start the API on another port with `--port`. |
| Python dependency/import error | Reinstall `requirements.txt`; avoid independent unpinned package upgrades. |

## API quick reference

Use [the API reference](../README.md#api). Chat/document requests require a
bearer token from `POST /api/auth/token`. Model weights may download from
external registries; document context goes to the configured Ollama endpoint.
See [data handling](SECURITY.md#data-handling) and
[production deployment](DEPLOYMENT.md).
