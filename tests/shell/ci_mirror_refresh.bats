#!/usr/bin/env bats
# tools/wheel-build/ci_mirror.sh with MIRROR_REFRESH_DAYS: a cache blob is
# re-published once its age bucket rolls over, and a restore reads the current
# bucket, then the one before it. A publish that does not upload exits non-zero,
# and every caller lets that step fail without failing the build.

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
    if [ -n "${REFUSE_UPLOAD:-}" ]; then
      echo "curl: (22) The requested URL returned error: 403" >&2
      exit 22
    fi
    name="${url##*name=}"
    cp "${data}" "${MIRROR}/${name}"
    echo "upload ${name}" >> "${CALLS}" ;;
  "GET https://api.github.com/repos/o/r/releases/tags/"*)
    [ -z "${NO_RELEASE:-}" ] || exit 22
    echo '{"id": 7}' ;;
  "GET https://api.github.com/repos/o/r/releases/7/assets"*)
    # Pages as the API does: per_page (default 30) and page (default 1).
    per=30; page=1
    [[ "${url}" =~ [?\&]per_page=([0-9]+) ]] && per="${BASH_REMATCH[1]}"
    [[ "${url}" =~ [?\&]page=([0-9]+) ]] && page="${BASH_REMATCH[1]}"
    files=()
    for f in "${MIRROR}"/*; do [ -e "${f}" ] && files+=("${f##*/}"); done
    printf '['
    for name in "${files[@]:$(( (page - 1) * per )):${per}}"; do printf '{"id":1,"name":"%s"},' "${name}"; done
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
  [ "$status" -eq 3 ]
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

@test "a refresh period that is not a whole number of days is refused by name" {
  for bad in 0 -3 abc 1.5 "3 days"; do
    MIRROR_REFRESH_DAYS="${bad}" run at_day 65 publish k "${SRC}"
    [ "$status" -eq 2 ]
    [[ "$output" == *"MIRROR_REFRESH_DAYS must be a whole number"* ]]
    MIRROR_REFRESH_DAYS="${bad}" run at_day 65 fetch k "${DEST}"
    [ "$status" -eq 2 ]
    MIRROR_REFRESH_DAYS="${bad}" run at_day 65 check
    [ "$status" -eq 2 ]
  done
  [ ! -e "${MIRROR}/k.b2.tar.gz" ]
}

@test "an empty refresh period counts as unset" {
  MIRROR_REFRESH_DAYS="" run at_day 65 publish k "${SRC}"
  [ "$status" -eq 0 ]
  [ -f "${MIRROR}/k.tar.gz" ]
}

# A read-only token gets the release and is refused the upload. Every caller
# runs the publish with continue-on-error, so a non-zero exit shows the step
# red while the build goes on.
@test "a refused upload fails the publish and names the reason" {
  REFUSE_UPLOAD=1 run at_day 65 publish k "${SRC}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Mirror upload of k.b2.tar.gz failed: curl: (22) The requested URL returned error: 403"* ]]
  [ ! -e "${MIRROR}/k.b2.tar.gz" ]
}

@test "a pack that fails fails the publish and names the reason" {
  printf '#!/bin/sh\nexit 1\n' > "${BATS_TEST_TMPDIR}/bin/tar"
  chmod +x "${BATS_TEST_TMPDIR}/bin/tar"
  run at_day 65 publish k "${SRC}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not pack ${SRC}; skipping the mirror upload."* ]]
  [ "$(grep -c '^upload' "${CALLS}")" -eq 0 ]
}

@test "an asset over the release limit fails the publish and names the reason" {
  # The packed size the script reads, 1 byte over 2 GiB.
  printf '#!/bin/sh\necho 2147483649\n' > "${BATS_TEST_TMPDIR}/bin/wc"
  chmod +x "${BATS_TEST_TMPDIR}/bin/wc"
  run at_day 65 publish k "${SRC}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Packed k.b2.tar.gz is 2048 MB, over the 2048 MB release-asset limit; skipping the mirror upload."* ]]
  [ "$(grep -c '^upload' "${CALLS}")" -eq 0 ]
}

@test "a release that cannot be found or made fails the publish and names the reason" {
  NO_RELEASE=1 run at_day 65 publish k "${SRC}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"Could not resolve or create engine-binaries; skipping the mirror upload."* ]]
  [ "$(grep -c '^upload' "${CALLS}")" -eq 0 ]
}

# The live mirror holds several pages of assets. One already there past the
# first page must count as carried, not be uploaded again.
@test "a publish finds an asset past the first page of the release" {
  for i in $(seq -w 1 150); do : > "${MIRROR}/a-${i}.tar.gz"; done
  : > "${MIRROR}/k.b2.tar.gz"
  run at_day 65 publish k "${SRC}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"Mirror already carries k.b2.tar.gz."* ]]
  [ "$(grep -c '^upload' "${CALLS}")" -eq 0 ]
}

# The steps that run a publish: the engine action anywhere, and the Nuitka
# cache action in its publish phase. One continue-on-error value per step.
publish_steps() {  # yq path to a step list, file
  yq -r "$1 | select(.uses == \"./.github/actions/publish-engine-asset\" or (.uses == \"./.github/actions/nuitka-object-cache\" and .with.phase == \"publish\")) | .[\"continue-on-error\"] // \"none\"" "$2"
}

@test "every caller of the publish swallows its failure" {
  found=0
  for f in "${REPO_ROOT}"/.github/workflows/*.yml "${REPO_ROOT}"/.github/actions/*/action.yml; do
    case "${f}" in
      */workflows/*) values=$(publish_steps '.jobs[].steps[]?' "${f}") ;;
      *) values=$(publish_steps '.runs.steps[]?' "${f}") ;;
    esac
    for v in ${values}; do
      echo "${f#"${REPO_ROOT}"/}: ${v}"
      [ "${v}" = "true" ]
      found=$((found + 1))
    done
  done
  [ "${found}" -eq 4 ]
}

@test "the Nuitka cache action sets a refresh period for the check, the restore and the publish" {
  run yq -r '[.runs.steps[].env.MIRROR_REFRESH_DAYS] | map(select(. != null)) | length' "${ACTION}"
  [ "$status" -eq 0 ]
  [ "$output" = "3" ]
}

@test "the action checks the refresh period first, with no condition and no continue-on-error" {
  run yq -r '.runs.steps[0] | [.name, (.["continue-on-error"] // "none"), (.if // "none"), .run] | join("|")' "${ACTION}"
  [ "$status" -eq 0 ]
  [ "$output" = "Check the refresh period|none|none|bash tools/wheel-build/ci_mirror.sh check" ]
}

@test "no workflow lets the restore phase continue past an error" {
  for wf in release.yml build-gpu-executables.yml; do
    run yq -r '.jobs[].steps[] | select(.id == "nuitka-mirror") | (.["continue-on-error"] // "none")' "${REPO_ROOT}/.github/workflows/${wf}"
    [ "$status" -eq 0 ]
    [ "$output" = "none" ]
  done
}

# Runs the action's restore step as written, for a key and a day, and prints its outputs.
run_restore_step() {  # day
  local day="$1" script
  script="$(yq -r '.runs.steps[] | select(.id == "restore") | .run' "${ACTION}")"
  script="${script//\$\{\{ inputs.key \}\}/k}"
  script="${script//\$\{\{ inputs.cache-dir \}\}/${DEST}}"
  : > "${BATS_TEST_TMPDIR}/gh-output"
  (cd "${REPO_ROOT}" && GITHUB_OUTPUT="${BATS_TEST_TMPDIR}/gh-output" MIRROR_NOW=$((day * 86400)) bash -c "${script}") > /dev/null 2>&1
  cat "${BATS_TEST_TMPDIR}/gh-output"
}

# The workflows publish when the current bucket was not restored.
publish_gate_open() {  # outputs
  [[ "$1" != *"current=true"* ]]
}

@test "both workflows publish on the current bucket, not on any restore" {
  for wf in release.yml build-gpu-executables.yml; do
    run yq -r '.jobs[].steps[] | select(.name == "Publish the Nuitka cache to the mirror") | .if' "${REPO_ROOT}/.github/workflows/${wf}"
    [ "$status" -eq 0 ]
    [[ "$output" == *"steps.nuitka-mirror.outputs.current != 'true'"* ]]
    [[ "$output" != *"outputs.restored"* ]]
  done
}

@test "the first build of a new bucket restores the old blob and still publishes the new one" {
  # Bucket 2 starts cold, publishes, and the next bucket must not start cold.
  out="$(run_restore_step 65)"
  [[ "$out" == *"restored=false"* ]]
  publish_gate_open "${out}"
  at_day 65 publish k "${SRC}"

  out="$(run_restore_step 95)"
  [[ "$out" == *"restored=true"* ]]
  [[ "$out" == *"current=false"* ]]
  publish_gate_open "${out}"
  at_day 95 publish k "${SRC}"
  [ -f "${MIRROR}/k.b3.tar.gz" ]

  out="$(run_restore_step 100)"
  [[ "$out" == *"restored=true"* ]]
  [[ "$out" == *"current=true"* ]]
  run publish_gate_open "${out}"
  [ "$status" -ne 0 ]
}

@test "with no refresh period a restore sets current and restored together" {
  unset MIRROR_REFRESH_DAYS
  at_day 65 publish k "${SRC}"
  out="$(run_restore_step 400)"
  [[ "$out" == *"restored=true"* ]]
  [[ "$out" == *"current=true"* ]]
}
