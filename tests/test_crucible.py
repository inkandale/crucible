import http.client
import json
import math
import threading

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from crucible.encode import encode, fields
from crucible.model import Crucible, check
from crucible.server import MAX_BODY, make_server
from crucible.values import values

ORDER = ["name", "description", "first_mes", "tags", "lorebook"]
CFG = {
    "version": "v-test", "max_tokens": 64, "order": ORDER,
    "truncate": ["lorebook", "tags", "description", "name", "first_mes"],
    "markers": [f"[{f}]" for f in ORDER], "cls_token_id": 1, "sep_token_id": 2,
    "classes": {"card_type": ["character", "scenario"], "rating": ["sfw", "nsfw"], "pov": ["first", "second", "third"]},
    "formats": ["prose", "wpp"], "format_names": {"prose": "Prose", "wpp": "W++"},
    "score_heads": ["overall", "dim:prose"],
    "binary": ["low_effort", "flag:minor_risk", "flag:gore", "contradiction:name:description",
               "suggestion:1", "suggestion:2", "suggestion:3", "suggestion:9"],
    "tags": ["elf"], "bank": {"1": "Add a greeting.", "2": "Trim the description.", "3": "Name the setting."},
    "cutoffs": {"*": 0.5, "flag:minor_risk": 0.3}, "max_suggestions": 2, "max_field_chars": 1000,
}


def logit(p):
    return math.log(p / (1 - p))


LOGITS = {
    "class:card_type": [0, 1], "class:rating": [2, 0], "class:pov": [0, 0, 1], "class:format": [0, 3],
    "score": [0, 2], "tags": [0],
    # low_effort and the contradiction pass 0.5; minor_risk passes only its own 0.3; gore misses.
    "binary": [1, logit(0.4), logit(0.4), 1, 1, 3, 2, 5],
}
VALUES = {
    "quality": 50.0, "prose": 88.1, "card_type": "scenario", "rating": "sfw", "pov": "third", "format": "W++",
    "flags": ["low effort", "minor risk"], "contradictions": ["name vs description"],
    "suggestions": ["Trim the description.", "Name the setting."],  # most likely first; 9 isn't in the bank
}
CARD = {"name": "a", "description": "b c", "first_mes": "d", "tags": ["B ", "a", " "]}


class FakeSession:
    def __init__(self):
        self.feeds = []

    def run(self, names, feed):
        self.feeds.append(feed)
        return [np.array([LOGITS[n]], dtype=np.float32) for n in names]


def tokenizer():
    words = ["[UNK]", "[CLS]", "[SEP]", "a", "b", "c", "d"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(CFG["markers"])
    return tok


def m(f):
    return 7 + ORDER.index(f)


def test_encode_layout():
    tok = tokenizer()
    ids = encode(tok, fields(CARD, 1000), CFG)
    assert ids == [1, m("name"), 3, m("description"), 4, 5, m("first_mes"), 6, m("tags"), 3, 0, 4, 2]


def test_wrapped_card_reads_its_data():
    tok = tokenizer()
    assert encode(tok, fields({"spec": "chara_card_v3", "data": CARD}, 1000), CFG) == encode(tok, fields(CARD, 1000), CFG)


def test_truncation_cuts_fields_from_the_end_in_truncate_order():
    tok = tokenizer()
    card = {"name": "a", "description": "b c", "first_mes": "d",
            "character_book": {"entries": [{"keys": ["a"], "content": "b b b"}]}}  # "a: b b b", 5 tokens
    text = fields(card, 1000)
    assert text["lorebook"] == "a: b b b"
    # 4 fields, 9 tokens: a budget of 6 cuts 3 from the lorebook's end.
    assert encode(tok, text, dict(CFG, max_tokens=12)) == [1, m("name"), 3, m("description"), 4, 5, m("first_mes"), 6,
                                                           m("lorebook"), 3, 0, 2]
    # A budget of 2 empties the lorebook, then cuts description; the fields keep their markers.
    assert encode(tok, text, dict(CFG, max_tokens=8)) == [1, m("name"), 3, m("description"), m("first_mes"), 6,
                                                          m("lorebook"), 2]


def test_field_chars_are_capped():
    assert fields({"name": "abcdef"}, 3)["name"] == "abc"


def test_values():
    assert Crucible(FakeSession(), tokenizer(), CFG).score(CARD) == VALUES


def test_rating_from_the_content_heads():
    classes = {"card_type": CFG["classes"]["card_type"], "pov": CFG["classes"]["pov"],
               "sexual": ["none", "suggestive", "explicit"], "violence": ["none", "mild", "graphic"]}
    cfg = dict(CFG, classes=classes, ratings=["sfw", "suggestive", "nsfw"], extreme_flags=["flag:minor_risk"])
    out = {k: np.array(v, dtype=np.float32) for k, v in LOGITS.items() if k != "class:rating"}
    v = values(dict(out, **{"class:sexual": np.array([3, 0, 0]), "class:violence": np.array([0, 3, 0])}), cfg)
    assert (v["sexual"], v["violence"], v["rating"]) == ("none", "mild", "suggestive")  # minor_risk hits, but nothing sexual
    v = values(dict(out, **{"class:sexual": np.array([0, 3, 0]), "class:violence": np.array([3, 0, 0])}), cfg)
    assert v["rating"] == "extreme"


def test_feeds_int64_ids_with_a_full_mask():
    session = FakeSession()
    Crucible(session, tokenizer(), CFG).score(CARD)
    feed = session.feeds[0]
    assert feed["input_ids"].dtype == np.int64 and feed["input_ids"].shape == (1, 13)
    assert (feed["attention_mask"] == 1).all()


def test_missing_marker_refuses_the_bundle():
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    with pytest.raises(SystemExit):
        Crucible(FakeSession(), tok, CFG)


def test_check():
    c = Crucible(FakeSession(), tokenizer(), CFG)
    assert check(c, [{"data": CARD, "values": VALUES}]) == []
    assert check(c, [{"data": CARD, "values": dict(VALUES, quality=50.9)}]) == []  # within a point
    assert check(c, [{"data": CARD, "values": dict(VALUES, quality=51.5)}]) == ["fixture 0 quality: got 50.0, want 51.5"]
    assert len(check(c, [{"data": CARD, "values": dict(VALUES, flags=["low effort"])}])) == 1
    assert len(check(c, [{"data": CARD, "values": {k: v for k, v in VALUES.items() if k != "pov"}}])) == 1
    assert check(c, []) == ["fixtures.json has no cards"]


@pytest.fixture
def server():
    s = make_server(Crucible(FakeSession(), tokenizer(), CFG), "127.0.0.1", 0)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield s.server_address[1]
    s.shutdown()


def post(port, body: bytes):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("POST", "/score", body, {"Content-Type": "application/json"})
    r = conn.getresponse()
    return r.status, json.loads(r.read())


def test_score_reply(server):
    status, body = post(server, json.dumps({"card_id": 1, "revision_id": 2, "data": CARD}).encode())
    assert status == 200
    assert body == {"values": VALUES, "meta": {"source": "crucible", "version": "v-test"}}


@pytest.mark.parametrize("body", [b"{}", b'{"card_id": 1}', b'{"data": "x"}', b"not json", b""])
def test_score_without_data_is_400(server, body):
    assert post(server, body)[0] == 400


def test_score_over_the_cap_is_413(server):
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.putrequest("POST", "/score")
    conn.putheader("Content-Length", str(MAX_BODY + 1))
    conn.endheaders()
    assert conn.getresponse().status == 413


def test_healthz_and_404(server):
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("GET", "/healthz")
    assert conn.getresponse().status == 200
    conn = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    conn.request("GET", "/nope")
    assert conn.getresponse().status == 404
