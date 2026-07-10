from __future__ import annotations

import hashlib
from pathlib import Path


LICENSE_PATH = Path(__file__).parents[1] / "LICENSE"
GITHUB_APACHE_2_SHA256 = "c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c"


def test_license_is_byte_exact_github_apache_2_standard_text() -> None:
    content = LICENSE_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == GITHUB_APACHE_2_SHA256
    assert content.startswith(b"                                 Apache License\n")
    assert b"TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in content
    assert b"Version 2.0, January 2004" in content
    assert content.endswith(b"   limitations under the License.\n\n")
