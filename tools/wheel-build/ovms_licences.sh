#!/bin/sh
# Copy the licence texts of the OpenVINO Model Server archive into the engine wheel project.
#
# Usage: ovms_licences.sh <unpacked-server-dir> <wheel-project-dir>
#
# Three texts of the archive stay out of the wheel, because the wheel holds no code that they cover:
# - espeak-ng: no library links libespeak-ng, and only the speech code of libopenvino_genai loads it.
# - ngsolve (LGPL-2.1): a search of every binary in the archive finds no ngsolve or netgen symbol.
# - ittnotify under GPL-2.0-only: the ittnotify sources in the binaries carry the licence
#   "BSD-3-Clause OR GPL-2.0-only", and the wheel takes the BSD-3-Clause one, whose text it ships.
# The only GPL code in the wheel is libgit2, which ovms_wheel.sh refuses to bundle without a ruling.
set -eu

test "$#" -eq 2 || {
  echo "usage: ovms_licences.sh <unpacked-server-dir> <wheel-project-dir>" >&2
  exit 2
}
server="$1" project="$2"
mkdir -p "${project}/THIRD-PARTY"
cp "${server}/LICENSE" "${project}/LICENSE-ovms"
cp -R "${server}/thirdparty-licenses/." "${project}/THIRD-PARTY/"
for name in espeak-ng.LICENSE.txt ngsolve.LICENSE.txt ittnotify-GPL-2.0-only.txt; do
  rm -f "${project}/THIRD-PARTY/${name}"
done
