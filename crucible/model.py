"""One crucible bundle, loaded into onnxruntime, and the start-up check against its fixtures."""
import json
import os
import threading
from pathlib import Path

import numpy as np

from .encode import encode, fields
from .values import values


class Crucible:
    def __init__(self, session, tok, cfg: dict):
        self.session, self.tok, self.cfg = session, tok, cfg
        tok.no_truncation()  # encode() truncates by field; the tokenizer must hand back every token
        tok.no_padding()
        if missing := [m for m in cfg["markers"] if tok.token_to_id(m) is None]:
            raise SystemExit(f"tokenizer.json lacks the field markers {missing}")
        self.outputs = [f"class:{h}" for h in cfg["classes"]] + ["class:format", "score", "binary"]
        self.lock = threading.Lock()  # one card at a time; onnxruntime already uses every core for one

    def score(self, data: dict) -> dict:
        ids = np.array([encode(self.tok, fields(data, self.cfg["max_field_chars"]), self.cfg)], dtype=np.int64)
        with self.lock:
            out = self.session.run(self.outputs, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        return values({name: o[0] for name, o in zip(self.outputs, out)}, self.cfg)


def bundle() -> Path:
    """The bundle directory: $CRUCIBLE_BUNDLE, or $CRUCIBLE_MODEL at $CRUCIBLE_REVISION from Hugging Face."""
    if local := os.environ.get("CRUCIBLE_BUNDLE"):
        return Path(local)
    if not (revision := os.environ.get("CRUCIBLE_REVISION")):
        raise SystemExit("set CRUCIBLE_REVISION to the bundle's tag (e.g. v2), or CRUCIBLE_BUNDLE to a local bundle")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=os.environ.get("CRUCIBLE_MODEL", "dgibbons/crucible"), revision=revision,
                                  token=os.environ.get("HF_TOKEN") or None))


def load(path: Path) -> tuple[Crucible, list]:
    import onnxruntime
    from tokenizers import Tokenizer
    opts = onnxruntime.SessionOptions()
    # Each card's length differs; an arena keeps the largest card's buffers forever, and those reach gigabytes.
    opts.enable_cpu_mem_arena, opts.enable_mem_pattern = False, False
    # onnxruntime's default of one thread per core runs slower on CPUs with efficiency cores.
    opts.intra_op_num_threads = int(os.environ.get("CRUCIBLE_THREADS", "0"))
    session = onnxruntime.InferenceSession(str(path / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    cfg = json.loads((path / "crucible.json").read_text())
    return Crucible(session, Tokenizer.from_file(str(path / "tokenizer.json")), cfg), json.loads((path / "fixtures.json").read_text())


def check(crucible: Crucible, fixtures: list) -> list[str]:
    """Every way the model's values differ from the fixtures': scores by more than 1 point, anything else at all."""
    if not fixtures:
        return ["fixtures.json has no cards"]
    scores = {"quality" if h == "overall" else h.removeprefix("dim:") for h in crucible.cfg["score_heads"]} | {"severity"}
    problems = []
    for n, fx in enumerate(fixtures):
        got, want = crucible.score(fx["data"]), fx["values"]
        for k in sorted(got.keys() | want.keys()):
            g, w = got.get(k), want.get(k)
            close = k in scores and isinstance(g, float) and isinstance(w, (int, float)) and abs(g - w) <= 1.0
            if g != w and not close:
                problems.append(f"fixture {n} {k}: got {g!r}, want {w!r}")
    return problems
