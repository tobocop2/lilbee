#!/usr/bin/env bats
# tools/wheel-build/check_ik_repro.sh: the rule that two CUDA builds of one ik commit are equal when
# libggml.so has equal host code and every other file is byte-equal. The tests build tiny shared libraries
# with cc and change one section at a time, so a rule that drops a section or skips the byte compare fails here.
# CHECK_IK_REPRO points at another copy of the script; the default is the one in the tree.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  CHECK="${CHECK_IK_REPRO:-${REPO_ROOT}/tools/wheel-build/check_ik_repro.sh}"
  command -v cc > /dev/null || skip "cc is not installed"
  command -v objcopy > /dev/null || skip "objcopy is not installed"
  [ "$(uname -s)" = Linux ] || skip "the test needs ELF shared libraries"
  A="${BATS_TEST_TMPDIR}/a"
  B="${BATS_TEST_TMPDIR}/b"
  mkdir -p "${A}" "${B}"
  printf 'server\n' > "${A}/llama-server"
  cp "${A}/llama-server" "${B}/llama-server"
  printf 'llama\n' > "${A}/libllama.so"
  cp "${A}/libllama.so" "${B}/libllama.so"
  printf 'fatbin one\n' > "${BATS_TEST_TMPDIR}/fat1"
  printf 'fatbin two\n' > "${BATS_TEST_TMPDIR}/fat2"
}

# lib <output> <symbol> <addend> <message>: a shared library with code, a string and an exported symbol.
lib() {
  local src="${BATS_TEST_TMPDIR}/src.c"
  printf 'static const char msg[] = "%s";\nconst char *get_msg(void) { return msg; }\nint %s(int x) { return x + %s; }\n' \
    "$4" "$2" "$3" > "${src}"
  cc -shared -fPIC -O0 -o "$1" "${src}"
}

# Two libggml.so files that differ only in the .nv_fatbin section.
fatbin_pair() {
  lib "${A}/libggml.so" ggml_add 1 "hello"
  cp "${A}/libggml.so" "${B}/libggml.so"
  objcopy --add-section .nv_fatbin="${BATS_TEST_TMPDIR}/fat1" "${A}/libggml.so"
  objcopy --add-section .nv_fatbin="${BATS_TEST_TMPDIR}/fat2" "${B}/libggml.so"
}

@test "builds that differ only in the GPU fatbin are accepted" {
  fatbin_pair
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"SAME libggml.so:.text"* ]]
  [[ "$output" == *"SAME libggml.so:.rodata"* ]]
  [[ "$output" == *"SAME libggml.so:.dynsym"* ]]
  [[ "$output" == *"SAME libggml.so:.dynstr"* ]]
  [[ "$output" == *"ACCEPT"* ]]
}

@test "identical builds are accepted" {
  lib "${A}/libggml.so" ggml_add 1 "hello"
  cp "${A}/libggml.so" "${B}/libggml.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 0 ]
  [[ "$output" == *"SAME libggml.so"* ]]
}

@test "a different .text is rejected" {
  fatbin_pair
  lib "${B}/libggml.so" ggml_add 2 "hello"
  objcopy --add-section .nv_fatbin="${BATS_TEST_TMPDIR}/fat2" "${B}/libggml.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libggml.so:.text"* ]]
  [[ "$output" == *"REJECT"* ]]
}

@test "a different .rodata is rejected" {
  fatbin_pair
  lib "${B}/libggml.so" ggml_add 1 "jello"
  objcopy --add-section .nv_fatbin="${BATS_TEST_TMPDIR}/fat2" "${B}/libggml.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libggml.so:.rodata"* ]]
  [[ "$output" == *"REJECT"* ]]
}

@test "a different .dynstr is rejected" {
  fatbin_pair
  lib "${B}/libggml.so" ggml_sub 1 "hello"
  objcopy --add-section .nv_fatbin="${BATS_TEST_TMPDIR}/fat2" "${B}/libggml.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libggml.so:.dynstr"* ]]
}

@test "a different .dynsym is rejected" {
  fatbin_pair
  # A third exported symbol changes the symbol table; the strings differ too, so only .dynsym is asserted.
  printf 'int extra(void) { return 0; }\n' > "${BATS_TEST_TMPDIR}/extra.c"
  cc -shared -fPIC -O0 -o "${B}/libggml.so" "${BATS_TEST_TMPDIR}/src.c" "${BATS_TEST_TMPDIR}/extra.c"
  objcopy --add-section .nv_fatbin="${BATS_TEST_TMPDIR}/fat2" "${B}/libggml.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libggml.so:.dynsym"* ]]
}

@test "a file other than libggml.so is byte-compared" {
  fatbin_pair
  printf 'other\n' > "${B}/libllama.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libllama.so"* ]]
  [[ "$output" == *"SAME llama-server"* ]]
  [[ "$output" == *"REJECT"* ]]
}

@test "a missing file is rejected" {
  fatbin_pair
  mv "${B}/llama-server" "${BATS_TEST_TMPDIR}/gone"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF llama-server"* ]]
}

@test "a libggml.so that is not an ELF file is rejected when it differs" {
  printf 'one\n' > "${A}/libggml.so"
  printf 'two\n' > "${B}/libggml.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libggml.so"* ]]
  [[ "$output" == *"REJECT"* ]]
}

@test "a failed extract of one build is a difference, never a stale equal" {
  fatbin_pair
  printf 'not an ELF file\n' > "${B}/libggml.so"
  run "${CHECK}" "${A}" "${B}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libggml.so:.text"* ]]
  [[ "$output" != *"ACCEPT"* ]]
}

@test "three builds are rejected when the third differs in .text" {
  fatbin_pair
  local C="${BATS_TEST_TMPDIR}/c"
  mkdir -p "${C}"
  cp "${A}/llama-server" "${A}/libllama.so" "${C}/"
  lib "${C}/libggml.so" ggml_add 3 "hello"
  run "${CHECK}" "${A}" "${B}" "${C}"
  [ "$status" -eq 1 ]
  [[ "$output" == *"DIFF libggml.so:.text"* ]]
}

@test "one folder is a usage error" {
  run "${CHECK}" "${A}"
  [ "$status" -eq 2 ]
}
