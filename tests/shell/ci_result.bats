#!/usr/bin/env bats
# scripts/ci_result.sh decides the one required check from the needed jobs' results.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/scripts/ci_result.sh"
  CI="${REPO_ROOT}/.github/workflows/ci.yml"
}

# needs name:result ... -> the JSON GitHub gives for toJSON(needs).
needs() {
  local entry out=""
  for entry in "$@"; do
    out="${out}${out:+,}\"${entry%%:*}\":{\"result\":\"${entry#*:}\",\"outputs\":{}}"
  done
  printf '{%s}' "${out}"
}

gate() {
  NEEDS_JSON="$1" bash "${SCRIPT}" changes
}

@test "every needed job passing passes" {
  run gate "$(needs changes:success lint:success test:success)"
  [ "$status" -eq 0 ]
}

@test "jobs skipped because their paths did not change pass" {
  run gate "$(needs changes:success lint:skipped test:skipped flake:skipped)"
  [ "$status" -eq 0 ]
}

@test "a failed job fails the gate and is named" {
  run gate "$(needs changes:success lint:success test:failure)"
  [ "$status" -eq 1 ]
  [[ "$output" == *"test"* ]]
}

@test "a cancelled job fails the gate" {
  run gate "$(needs changes:success lint:cancelled test:skipped)"
  [ "$status" -eq 1 ]
  [[ "$output" == *"lint"* ]]
}

@test "a skipped path filter fails the gate, so nothing passes untested" {
  run gate "$(needs changes:skipped lint:skipped test:skipped)"
  [ "$status" -eq 1 ]
  [[ "$output" == *"changes"* ]]
}

@test "a failed path filter fails the gate" {
  run gate "$(needs changes:failure lint:skipped)"
  [ "$status" -eq 1 ]
}

@test "a result the gate does not know fails it" {
  run gate "$(needs changes:success lint:neutral)"
  [ "$status" -eq 1 ]
}

@test "an empty needs map fails the gate" {
  run gate '{}'
  [ "$status" -eq 1 ]
}

# Plain jq-style paths only: the laptop's yq is the jq wrapper, CI's is the Go one.
job_needs() {
  yq -r '.jobs | to_entries | .[] | .key + "=" + ((.value.needs // []) | join(","))' "${CI}"
}

@test "ci-result needs every other job in the workflow" {
  run job_needs
  [ "$status" -eq 0 ]
  others=$(printf '%s\n' "$output" | cut -d= -f1 | grep -vx 'ci-result' | sort | paste -sd, -)
  [ -n "${others}" ]
  needed=$(printf '%s\n' "$output" | grep '^ci-result=' | cut -d= -f2 | tr , '\n' | sort | paste -sd, -)
  [ "${needed}" = "${others}" ]
}

@test "ci-result runs whatever the needed jobs concluded" {
  run yq -r '.jobs["ci-result"].if' "${CI}"
  [ "$output" = "always()" ]
}

@test "every job behind the path filter waits for it" {
  run job_needs
  [ "$status" -eq 0 ]
  missing=$(printf '%s\n' "$output" | grep -v -e '^changes=' -e '^ci-result=' | grep -v -E '=(.*,)?changes(,|$)' || true)
  [ -z "${missing}" ]
}

@test "the workflow starts for every pull request and every push to main" {
  # The jq wrapper reads the bare key `on` as a boolean, so the filter is read as text.
  run grep -c -E '^    paths' "${CI}"
  [ "$output" = "0" ]
  run grep -c -E '^  (pull_request|push):' "${CI}"
  [ "$output" = "2" ]
}

@test "ci-result decides with this script and requires the path filter to succeed" {
  run yq -r '.jobs["ci-result"].steps[] | select(.run != null) | .run' "${CI}"
  [ "$output" = "bash scripts/ci_result.sh changes" ]
  run yq -r '.jobs["ci-result"].steps[] | select(.run != null) | .env.NEEDS_JSON' "${CI}"
  [ "$output" = '${{ toJSON(needs) }}' ]
}

@test "the flake job evaluates the flake against its pins" {
  run yq -r '.jobs.flake.if, (.jobs.flake.steps[] | select(.run != null) | .run)' "${CI}"
  [ "${lines[0]}" = "needs.changes.outputs.flake == 'true'" ]
  [[ "$output" == *"nix flake check --all-systems --no-build"* ]]
  [[ "$output" == *"p.drvPath"* ]]
}

# The path filter's two groups, read as the YAML the filter action reads.
path_group() {
  yq -r '.jobs.changes.steps[] | select(.id == "filter") | .with.filters' "${CI}" | yq -r ".$1[]"
}

@test "the flake group is exactly the flake's inputs and this workflow" {
  run path_group flake
  [ "$status" -eq 0 ]
  expected=$(printf '%s\n' flake.nix flake.lock sources.json sources-cuda.json sources-rocm.json \
    sources-compat.json .github/workflows/ci.yml)
  [ "$output" = "${expected}" ]
}

@test "the code group covers every tree the code jobs test" {
  run path_group code
  [ "$status" -eq 0 ]
  for p in 'src/**' 'tests/**' pyproject.toml uv.lock Makefile engine-versions.env \
    scripts/check_style_rules.py 'scripts/*.sh' 'scripts/qa/**' 'tools/qa/**' 'packages/lilbee/**' \
    'tools/wheel-build/**' 'packaging/tools/**' '.github/workflows/**' '.github/actions/**'; do
    printf '%s\n' "$output" | grep -qxF -e "${p}" || { echo "code group lost ${p}" >&2; return 1; }
  done
}

@test "the path filter defines the code and flake groups and no other" {
  run bash -c "yq -r '.jobs.changes.steps[] | select(.id == \"filter\") | .with.filters' '${CI}' | yq -r 'keys | join(\",\")'"
  [ "$output" = "code,flake" ]
  run yq -r '.jobs.changes.outputs | keys | join(",")' "${CI}"
  [ "$output" = "code,flake" ]
}

@test "each job runs on the path group it tests" {
  run yq -r '.jobs | to_entries | .[] | select(.key != "changes" and .key != "ci-result") | .key + "=" + .value.if' "${CI}"
  [ "$status" -eq 0 ]
  expected=$(printf '%s\n' \
    "lint=needs.changes.outputs.code == 'true'" \
    "shell=needs.changes.outputs.code == 'true'" \
    "npm-launcher=needs.changes.outputs.code == 'true'" \
    "fuzz-smoke=needs.changes.outputs.code == 'true'" \
    "test=needs.changes.outputs.code == 'true'" \
    "integration=needs.changes.outputs.code == 'true'" \
    "flake=needs.changes.outputs.flake == 'true'")
  [ "$output" = "${expected}" ]
}
