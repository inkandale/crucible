# crucible

Serves a trained crucible model to Innkeeper's crucible enrich plugin. At start it downloads one model **bundle** from Hugging Face, scores the bundle's fixture cards, and refuses to serve if any value differs from what the trainer recorded: a score by more than 1 point, anything else at all. It then answers `POST /score` with `{"values": {...}, "meta": {"source": "crucible", "version": ...}}`, one card at a time.

A bundle is a revision of the Hugging Face model repo holding `model.onnx`, `tokenizer.json`, `crucible.json` (field order, truncation, heads, cutoffs) and `fixtures.json`. crucible-trainer's exporter writes it.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `CRUCIBLE_REVISION` | required | The bundle's tag, e.g. `v2` |
| `CRUCIBLE_MODEL` | `dgibbons/crucible` | The Hugging Face model repo |
| `HF_TOKEN` | none | A read token; the model repo is private |
| `HF_HOME` | `/cache` in the image | Where the bundle downloads |
| `CRUCIBLE_BUNDLE` | none | A local bundle directory, used instead of downloading |
| `PORT` | `8080` | The listen port |

## Ship a new model version

1. Export the bundle in crucible-trainer, upload it, and tag the commit:

   ```sh
   hf upload dgibbons/crucible bundles/v3 . --private
   hf repos tag create dgibbons/crucible v3
   ```

2. Set `revision: "v3"` in `chart/values.yaml` and merge to `main`. Argo CD rolls it out. A bundle that doesn't reproduce its fixtures crash-loops and the old pod keeps serving.

A merge to `main` also builds `ghcr.io/inkandale/crucible:<sha>` and pins it in `chart/Chart.yaml`'s `appVersion`.

## Run locally

```sh
uv sync
CRUCIBLE_BUNDLE=path/to/bundle uv run python -m crucible
curl -s localhost:8080/score -d '{"data": {"name": "Ann", "description": "An elf."}}'
```

Tests need no model or network: `uv run pytest`.
