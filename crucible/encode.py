"""A card as model input: crucible-trainer's card view and encode(), without span labels."""
FIELDS = ("name", "description", "personality", "scenario", "first_mes", "mes_example",
          "system_prompt", "post_history_instructions", "creator_notes")


def card_data(card: dict) -> dict:
    """Return the CCv3 data object of a wrapped card, or the card if it is already bare."""
    return card["data"] if isinstance(card, dict) and "spec" in card and isinstance(card.get("data"), dict) else card


def lorebook(data: dict, max_chars: int) -> str:
    book = data.get("character_book")
    entries = book.get("entries") if isinstance(book, dict) else None
    lines = []
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and isinstance(e.get("content"), str):
            keys = ", ".join(k for k in e.get("keys") or [] if isinstance(k, str))
            lines.append(f"{keys}: {e['content']}" if keys else e["content"])
    return "\n".join(lines)[:max_chars]


def fields(card: dict, max_chars: int) -> dict:
    """The text of each field the model reads."""
    d = card_data(card)
    out = {f: str(d.get(f) or "")[:max_chars] for f in FIELDS}
    out["lorebook"] = lorebook(d, max_chars)
    out["tags"] = ", ".join(sorted({t.strip().lower() for t in d.get("tags") or [] if isinstance(t, str) and t.strip()}))
    return out


def encode(tok, text: dict, cfg: dict) -> list[int]:
    """Input ids: [CLS], then each present field's marker and tokens in cfg's order, then [SEP]."""
    parts = {f: tok.encode(text[f], add_special_tokens=False).ids for f in cfg["order"] if text.get(f)}
    over = sum(map(len, parts.values())) - (cfg["max_tokens"] - 2 - len(parts))  # [CLS], [SEP], one marker per field
    for f in cfg["truncate"]:
        if over <= 0:
            break
        if f in parts:
            cut = min(over, len(parts[f]))
            del parts[f][len(parts[f]) - cut:]
            over -= cut
    ids = [cfg["cls_token_id"]]
    for f, marker in zip(cfg["order"], cfg["markers"]):
        if f in parts:
            ids += [tok.token_to_id(marker), *parts[f]]
    return ids + [cfg["sep_token_id"]]
