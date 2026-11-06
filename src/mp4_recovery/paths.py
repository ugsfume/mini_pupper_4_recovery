"""Where the package's inputs live."""
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]     # the source checkout: assumes the editable install the installer makes
DATA = PACKAGE / "data"
ASSETS = PACKAGE / "robot" / "assets"
SKU2_XML = ASSETS / "sku_2" / "pupper.xml"
MASSES_JSON = DATA / "sku2_masses.json"
RUNS = REPO / "runs"
POLICIES = REPO / "policies"


TEXT_SUFFIXES = {".xml", ".json", ".py", ".csv", ".md", ".txt"}


def sha256(path) -> str:
    """The file's digest; text sources are hashed with LF line endings, so a Windows checkout gives the same digest."""
    p = Path(path)
    b = p.read_bytes()
    if p.suffix.lower() in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()
