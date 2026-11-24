#!/usr/bin/env bats
# tools/wheel-build/ci_mirror.sh with MIRROR_REFRESH_DAYS: a cache blob is
# re-published once its age bucket rolls over, and a restore reads the current
# bucket, then the one before it.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/tools/wheel-build/ci_mirror.sh"
  ACTION="${REPO_ROOT}/.github/actions/nuitka-object-cache/action.yml"
  MIRROR="${BATS_TEST_TMPDIR}/mirror"
  CALLS="${BATS_TEST_TMPDIR}/calls"
  SRC="${BATS_TEST_TMPDIR}/src"
  DEST="${BATS_TEST_TMPDIR}/dest"
  mkdir -p "${MIRROR}" "${SRC}" "${BATS_TEST_TMPDIR}/bin"
  : > "${CALLS}"
  echo object > "${SRC}/obj"

  # A curl that serves the release API and the asset store from ${MIRROR}.
  cat > "${BATS_TEST_TMPDIR}/bin/curl" <<'STUB'
#!/usr/bin/env bash
out=""; data=""; method=GET; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out="$2"; shift ;;
    --data-binary) data="${2#@}"; shift ;;
    -X) method="$2"; shift ;;
    -H|--retry|--retry-delay|-d) shift ;;
    -*) ;;
    *) url="$1" ;;
  esac
  shift
done
case "${method} ${url}" in
  "POST https://uploads.github.com/"*)
    name="${url##*name=}"
    cp "${data}" "${MIRROR}/${name}"
    echo "upload ${name}" >> "${CALLS}" ;;
  "GET https://api.github.com/repos/o/r/releases/tags/"*) echo '{"id": 7}' ;;
  "GET https://api.github.com/repos/o/r/releases/7/assets"*)
    printf '['
    for f in "${MIRROR}"/*; do [ -e "${f}" ] && printf '{"id":1,"name":"%s"},' "${f##*/}"; done
    printf '{}]' ;;
  *)
    asset="${url##*/}"
    echo "get ${asset}" >> "${CALLS}"
    [ -f "${MIRROR}/${asset}" ] || exit 22
    [ -z "${out}" ] || cp "${MIRROR}/${asset}" "${out}" ;;
esac
STUB
  chmod +x "${BATS_TEST_TMPDIR}/bin/curl"
  export PATH="${BATS_TEST_TMPDIR}/bin:${PATH}"
  export MIRROR CALLS
  export GITHUB_REPOSITORY=o/r GH_TOKEN=t
  export MIRROR_REFRESH_DAYS=30
}

at_day() {  # day-number command...
  local day="$1"; shift
  MIRROR_NOW=$((day * 86400)) bash "${SCRIPT}" "$@"
}

@test "a publish names the asset after the age bucket" {
  run at_day 65 publish k "${SRC}"
  [ "$status" -eq 0 ]
  [ -f "${MIRROR}/k.b2.tar.gz" ]
}

@test "a second publish in the same bucket leaves the blob alone" {
  at_day 65 publish k "${SRC}"
  run at_day 80 publish k "${SRC}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"already carries"* ]]
  [ "$(grep -c '^upload' "${CALLS}")" -eq 1 ]
}

@test "a publish in the next bucket writes a fresh blob" {
  at_day 65 publish k "${SRC}"
  run at_day 95 publish k "${SRC}"
  [ "$status" -eq 0 ]
  [ -f "${MIRROR}/k.b3.tar.gz" ]
  [ -f "${MIRROR}/k.b2.tar.gz" ]
}

@test "a restore prefers the current bucket" {
  at_day 65 publish k "${SRC}"
  at_day 95 publish k "${SRC}"
  run at_day 100 fetch k "${DEST}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"k.b3.tar.gz"* ]]
}

@test "a restore falls back to the previous bucket" {
  at_day 65 publish k "${SRC}"
  run at_day 95 fetch k "${DEST}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"k.b2.tar.gz"* ]]
  [ -f "${DEST}/obj" ]
}

@test "a restore does not reach back two buckets" {
  at_day 65 publish k "${SRC}"
  run at_day 125 fetch k "${DEST}"
  [ "$status" -eq 1 ]
  [ ! -e "${DEST}/obj" ]
}

@test "without a refresh period the asset name is the plain key" {
  unset MIRROR_REFRESH_DAYS
  run at_day 65 publish k "${SRC}"
  [ "$status" -eq 0 ]
  [ -f "${MIRROR}/k.tar.gz" ]
  run at_day 400 fetch k "${DEST}"
  [ "$status" -eq 0 ]
}

@test "the Nuitka cache action sets a refresh period for both phases" {
  run yq -r '[.runs.steps[].env.MIRROR_REFRESH_DAYS] | map(select(. != null)) | length' "${ACTION}"
  [ "$status" -eq 0 ]
  [ "$output" = "2" ]
}
