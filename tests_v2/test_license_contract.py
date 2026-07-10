from __future__ import annotations

import hashlib
from pathlib import Path


LICENSE_PATH = Path(__file__).parents[1] / "LICENSE"
GITHUB_APACHE_2_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"


def test_license_is_byte_exact_github_apache_2_standard_text() -> None:
    content = LICENSE_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == GITHUB_APACHE_2_SHA256
    assert content.startswith(b"                                 Apache License\n")
    assert b"TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in content
    assert b"Version 2.0, January 2004" in content
    assert content.endswith(b"   limitations under the License.\n")
