"""A card's logits as the values crucible's enrich plugin stores in Innkeeper: crucible-trainer's publish.values."""
import numpy as np


def sigmoid(x) -> list[float]:
    with np.errstate(over="ignore"):  # exp overflows to inf for very negative logits, which still gives 0
        return (1 / (1 + np.exp(-np.asarray(x, dtype=np.float32)))).tolist()  # float32, as the trainer's torch heads


def values(out: dict, cfg: dict) -> dict:
    """out maps each head's output name to its one card's logits."""
    v = {}
    for h, s in zip(cfg["score_heads"], sigmoid(out["score"])):
        v["quality" if h == "overall" else h.removeprefix("dim:")] = round(s * 100, 1)
    for head, labels in cfg["classes"].items():
        v[head] = labels[int(np.argmax(out[f"class:{head}"]))]
    if "ratings" in cfg:  # bundles before v3b publish the rating head's label as it is
        x = np.asarray(out["class:rating"], dtype=np.float32)
        p = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
        v["rating"] = cfg["ratings"][min(int(np.argmax(p)), len(cfg["ratings"]) - 1)]  # extreme reads as nsfw
        v["severity"] = round(float(p[cfg["classes"]["rating"].index("extreme")]) * 100, 1)
    v["format"] = cfg["format_names"][cfg["formats"][int(np.argmax(out["class:format"]))]]
    cutoffs = cfg["cutoffs"]
    probs = dict(zip(cfg["binary"], sigmoid(out["binary"])))
    hit = {h for h, p in probs.items() if p >= cutoffs.get(h, cutoffs["*"])}
    v["flags"] = sorted(h.removeprefix("flag:").replace("_", " ") for h in hit if h.startswith("flag:") or h == "low_effort")
    v["contradictions"] = sorted(" vs ".join(h.split(":")[1:]) for h in hit if h.startswith("contradiction:"))
    picks = sorted(((probs[h], int(h.split(":")[1])) for h in hit if h.startswith("suggestion:")), reverse=True)
    v["suggestions"] = [cfg["bank"][str(i)] for _, i in picks if str(i) in cfg["bank"]][:cfg["max_suggestions"]]
    return v
