#!/usr/bin/env bats
# tools/wheel-build/collect_vendored_licences.sh finds the vendored licence texts in a checkout and in fetched trees, and names the terms it finds.

setup() {
  REPO_ROOT="$(cd "${BATS_TEST_DIRNAME}/../.." && pwd)"
  SCRIPT="${REPO_ROOT}/tools/wheel-build/collect_vendored_licences.sh"
  SRC="${BATS_TEST_TMPDIR}/llama.cpp"
  OUT="${BATS_TEST_TMPDIR}/out"
  TERMS="${BATS_TEST_TMPDIR}/terms"
  mkdir -p "${SRC}/vendor/httplib" "${SRC}/vendor/stb" "${SRC}/vendor/miniaudio" "${SRC}/common"
  echo "Permission is hereby granted" > "${SRC}/vendor/httplib/LICENSE"
  printf '%s\n' "This software is available under 2 licenses" "ALTERNATIVE A - MIT License" > "${SRC}/vendor/stb/stb_image.h"
  printf '%s\n' "ALTERNATIVE 1 - Public Domain" "This is free and unencumbered software released into the public domain." \
    "ALTERNATIVE 2 - MIT No Attribution" > "${SRC}/vendor/miniaudio/miniaudio.h"
}

@test "names the Unlicense and MIT-0 terms of the copied texts" {
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  [ "$(sort "${TERMS}")" = "$(printf 'MIT-0\nUnlicense')" ]
  [ -s "${OUT}/vendor/miniaudio/miniaudio.h.NOTICE" ]
}

@test "names only the terms that the copied texts hold" {
  printf '%s\n' "ALTERNATIVE 1 - Public Domain" "ALTERNATIVE 2 - MIT No Attribution" > "${SRC}/vendor/miniaudio/miniaudio.h"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  [ "$(cat "${TERMS}")" = "MIT-0" ]
}

@test "finds a licence file and a source header in places that no list names" {
  mkdir -p "${SRC}/ggml/src/newlib"
  echo "Redistribution and use in source and binary forms" > "${SRC}/ggml/src/newlib/COPYING"
  printf '%s\n' "// Copyright 2024 Someone Else" "// Permission is hereby granted, free of charge" "" "int x;" > "${SRC}/common/other.cpp"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  [ -s "${OUT}/ggml/src/newlib/COPYING" ]
  grep -q "Someone Else" "${OUT}/common/other.cpp.NOTICE"
  run grep -c "int x" "${OUT}/common/other.cpp.NOTICE"
  [ "$output" = "0" ]
}

@test "copies the whole header of a file that holds its grant in a second comment block" {
  printf '%s\n' "/*" "   Intro" "*/" "" "/*" "   This is free and unencumbered software released into the public domain." "*/" "int x;" > "${SRC}/common/b64.hpp"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  grep -q "free and unencumbered" "${OUT}/common/b64.hpp.NOTICE"
}

@test "skips the project's own headers and the directories that no wheel ships" {
  mkdir -p "${SRC}/tests" "${SRC}/ggml/src/ggml-cann"
  echo "Permission is hereby granted" > "${SRC}/tests/LICENSE"
  printf '%s\n' "Copyright (c) 2023 The ggml authors" "Permission is hereby granted" > "${SRC}/ggml/src/ggml-cann/common.h"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  [ ! -e "${OUT}/tests/LICENSE" ]
  [ ! -e "${OUT}/ggml/src/ggml-cann/common.h.NOTICE" ]
}

@test "fails when a checkout holds no licence file at all" {
  EMPTY="${BATS_TEST_TMPDIR}/empty"
  mkdir -p "${EMPTY}"
  run "${SCRIPT}" "${EMPTY}" "${OUT}" "${TERMS}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"no licence file or header found"* ]]
}

@test "fails when a header no longer holds its licence text" {
  echo "no licence here" > "${SRC}/vendor/stb/stb_image.h"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"stb_image.h: no 'ALTERNATIVE A - MIT License' in the licence text"* ]]
}

@test "prunes only the top-level folders that hold no shipped code" {
  mkdir -p "${SRC}/vendor/lib/docs" "${SRC}/src/models"
  echo "Permission is hereby granted" > "${SRC}/vendor/lib/docs/LICENSE"
  echo "Permission is hereby granted" > "${SRC}/src/models/COPYING"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  [ -s "${OUT}/vendor/lib/docs/LICENSE" ]
  [ -s "${OUT}/src/models/COPYING" ]
}

@test "copies the notice of a file whose first line is larger than a pipe buffer" {
  {
    echo "/* Copyright 2024 Big Banner. Permission is hereby granted, free of charge */"
    head -c 400000 /dev/zero | tr '\0' 'x'
    echo
  } > "${SRC}/common/bundle.js"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  grep -q "Big Banner" "${OUT}/common/bundle.js.NOTICE"
}

@test "writes no notice for a grant that sits in code, and says so" {
  printf '%s\n' '#include <string>' 'const char *s = "Permission is hereby granted";' > "${SRC}/common/quote.cpp"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  [ ! -e "${OUT}/common/quote.cpp.NOTICE" ]
  [[ "$output" == *"common/quote.cpp: the grant is not in the leading comment"* ]]
}

@test "skips a folder named with --skip" {
  mkdir -p "${SRC}/tools/ui/vendors"
  echo "Permission is hereby granted" > "${SRC}/tools/ui/vendors/LICENSE"
  run "${SCRIPT}" --skip tools/ui "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  [ ! -e "${OUT}/tools/ui/vendors/LICENSE" ]
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ -s "${OUT}/tools/ui/vendors/LICENSE" ]
}

@test "copies the licence files of every tree that the build fetched" {
  DEPS="${BATS_TEST_TMPDIR}/_deps"
  mkdir -p "${DEPS}/boringssl-src/third_party/fiat" "${DEPS}/boringssl-src/crypto" "${DEPS}/other-subbuild"
  echo "Apache License" > "${DEPS}/boringssl-src/LICENSE"
  echo "fiat text" > "${DEPS}/boringssl-src/third_party/fiat/LICENSE"
  echo "int x;" > "${DEPS}/boringssl-src/crypto/a.c"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}" "${DEPS}"
  [ "$status" -eq 0 ]
  [ -s "${OUT}/_deps/boringssl/LICENSE" ]
  [ -s "${OUT}/_deps/boringssl/third_party/fiat/LICENSE" ]
  [ ! -e "${OUT}/_deps/boringssl/crypto/a.c" ]
}

@test "fails when a fetched tree holds no licence file" {
  DEPS="${BATS_TEST_TMPDIR}/_deps"
  mkdir -p "${DEPS}/unlicensed-src"
  echo "int x;" > "${DEPS}/unlicensed-src/a.c"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}" "${DEPS}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"the fetched tree unlicensed holds no licence file"* ]]
}

@test "lists every copied file in MANIFEST, and not MANIFEST itself" {
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  grep -qx "vendor/httplib/LICENSE" "${OUT}/MANIFEST"
  grep -qx "vendor/miniaudio/miniaudio.h.NOTICE" "${OUT}/MANIFEST"
  run grep -c "MANIFEST" "${OUT}/MANIFEST"
  [ "$output" = "0" ]
  [ "$(wc -l < "${OUT}/MANIFEST")" -eq 3 ]
}

@test "fails when the deps folder does not exist" {
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}" "${BATS_TEST_TMPDIR}/no-such-folder"
  [ "$status" -ne 0 ]
  [[ "$output" == *"the deps folder ${BATS_TEST_TMPDIR}/no-such-folder does not exist"* ]]
}

@test "fails when the deps folder holds no fetched tree" {
  DEPS="${BATS_TEST_TMPDIR}/_deps"
  mkdir -p "${DEPS}/boringssl-build" "${DEPS}/boringssl-subbuild"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}" "${DEPS}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"the deps folder ${DEPS} holds no fetched tree"* ]]
}

@test "names the LLVM exception and Boost terms of a fetched licence text" {
  DEPS="${BATS_TEST_TMPDIR}/_deps"
  mkdir -p "${DEPS}/cccl-src"
  printf '%s\n' "---- LLVM Exceptions to the Apache 2.0 License ----" "Boost Software License - Version 1.0 - August 17th, 2003" \
    > "${DEPS}/cccl-src/LICENSE"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}" "${DEPS}"
  [ "$status" -eq 0 ]
  [ -s "${OUT}/_deps/cccl/LICENSE" ]
  [ "$(sort "${TERMS}")" = "$(printf 'Apache-2.0 WITH LLVM-exception\nBSL-1.0\nMIT-0\nUnlicense')" ]
}
