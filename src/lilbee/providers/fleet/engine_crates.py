"""The MPL-2.0 crates of the engine packs, with the sha256 that crates.io lists for each.

A pack build adds a line here for each MPL-2.0 crate of its locked dependency graph, taken
from the ``checksum`` field of its ``Cargo.lock``. The install check reads no network.
"""

from __future__ import annotations

from collections.abc import Mapping

_SOURCES = """
cssparser 0.36.0 dae61cf9c0abb83bd659dab65b7e4e38d8236824c85f0f804f173567bda257d2
cssparser-macros 0.6.1 13b588ba4ac1a99f7f2964d24b3d896ddc6bf847ee3855dbd4366f058cfcd331
dtoa-short 0.3.5 cd1511a7b6a56299bd043a9c167a6d2bfb37bf84a6dfceaba651168adfb43c87
option-ext 0.2.0 04744f49eae99ab78e0d5c0b603ab218f515ea8cfe5a456d7629ad883a3b6e7d
selectors 0.33.0 feef350c36147532e1b79ea5c1f3791373e61cbd9a6a2615413b3807bb164fb7
symphonia 0.5.5 5773a4c030a19d9bfaa090f49746ff35c75dfddfa700df7a5939d5e076a57039
symphonia-bundle-flac 0.5.5 c91565e180aea25d9b80a910c546802526ffd0072d0b8974e3ebe59b686c9976
symphonia-bundle-mp3 0.5.5 4872dd6bb56bf5eac799e3e957aa1981086c3e613b27e0ac23b176054f7c57ed
symphonia-codec-pcm 0.5.5 4e89d716c01541ad3ebe7c91ce4c8d38a7cf266a3f7b2f090b108fb0cb031d95
symphonia-codec-vorbis 0.5.5 f025837c309cd69ffef572750b4a2257b59552c5399a5e49707cc5b1b85d1c73
symphonia-core 0.5.5 ea00cc4f79b7f6bb7ff87eddc065a1066f3a43fe1875979056672c9ef948c2af
symphonia-format-isomp4 0.5.5 243739585d11f81daf8dac8d9f3d18cc7898f6c09a259675fc364b382c30e0a5
symphonia-format-ogg 0.5.5 2b4955c67c1ed3aa8ae8428d04ca8397fbef6a19b2b051e73b5da8b1435639cb
symphonia-format-riff 0.5.5 c2d7c3df0e7d94efb68401d81906eae73c02b40d5ec1a141962c592d0f11a96f
symphonia-metadata 0.5.5 36306ff42b9ffe6e5afc99d49e121e0bd62fe79b9db7b9681d48e29fa19e6b16
symphonia-utils-xiph 0.5.5 ee27c85ab799a338446b68eec77abf42e1a6f1bb490656e121c6e27bfbab9f16
"""

MPL_CRATE_CHECKSUMS: Mapping[tuple[str, str], str] = {
    (name, version): digest
    for name, version, digest in (line.split() for line in _SOURCES.strip().splitlines())
}

_PYPI_SOURCES = """
tqdm 4.70.1 cefd0eca11b2a37a3aee776544d4f4ae913f02688135b5556b8788dfa474afc4
"""
"""The MPL-2.0 Python packages of the engine packs, with the sha256 of the source archive on PyPI.

A pack that bundles a Python tree adds a line here for each MPL-2.0 package of its requirements.
"""

MPL_PYPI_CHECKSUMS: Mapping[tuple[str, str], str] = {
    (name, version): digest
    for name, version, digest in (line.split() for line in _PYPI_SOURCES.strip().splitlines())
}
