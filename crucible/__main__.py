import os

from .model import bundle, check, load
from .server import make_server

crucible, fixtures = load(bundle())
if problems := check(crucible, fixtures):
    raise SystemExit(f"crucible {crucible.cfg['version']} doesn't reproduce its fixtures, so it won't serve:\n" + "\n".join(problems))
port = int(os.environ.get("PORT", "8080"))
print(f"crucible {crucible.cfg['version']} matches its {len(fixtures)} fixtures; serving on :{port}", flush=True)
make_server(crucible, "0.0.0.0", port).serve_forever()
