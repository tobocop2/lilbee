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

# The headers below are the first lines of the files in the ik_llama.cpp tree at the pinned commit.
write_ik_vendored_tree() {
  mkdir -p "${SRC}/vendor/nlohmann" "${SRC}/vendor/cpp-httplib" "${SRC}/ggml/src/llamafile"
  printf '%s\n' "MIT License" "" "Copyright (c) 2023-2024 The ggml authors" "" \
    "Permission is hereby granted, free of charge, to any person obtaining a copy" "of this software, to deal in the Software without restriction." \
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND." > "${SRC}/LICENSE"
  printf '%s\n' "//     __ _____ _____ _____" "//  __|  |   __|     |   | |  JSON for Modern C++" \
    "//" "// SPDX-FileCopyrightText: 2013 - 2025 Niels Lohmann <https://nlohmann.me>" "// SPDX-License-Identifier: MIT" "" \
    "/****************************************************************************\\" " * Note on documentation: see the online docs." \
    "\\****************************************************************************/" "#include <algorithm>" > "${SRC}/vendor/nlohmann/json.hpp"
  printf '%s\n' "//" "//  httplib.h" "//" "//  Copyright (c) 2026 Yuji Hirose. All rights reserved." "//  MIT License" "//" "" \
    "#define CPPHTTPLIB_VERSION \"0.46.0\"" > "${SRC}/vendor/cpp-httplib/httplib.h"
  printf '%s\n' "// Copyright 2024 Mozilla Foundation" "//" "// Permission is hereby granted, free of charge, to any person obtaining" \
    "// a copy of this software." "" "#include <cstdint>" > "${SRC}/ggml/src/llamafile/sgemm.cpp"
}

@test "writes the notice of every vendored MIT file of the ik_llama.cpp tree" {
  write_ik_vendored_tree
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  for path in vendor/nlohmann/json.hpp vendor/cpp-httplib/httplib.h ggml/src/llamafile/sgemm.cpp; do
    [ -s "${OUT}/${path}.NOTICE" ]
    grep -qx "${path}.NOTICE" "${OUT}/MANIFEST"
    grep -q "Permission is hereby granted" "${OUT}/${path}.NOTICE"
  done
  grep -q "Niels Lohmann" "${OUT}/vendor/nlohmann/json.hpp.NOTICE"
  grep -q "Yuji Hirose" "${OUT}/vendor/cpp-httplib/httplib.h.NOTICE"
  grep -q "Mozilla Foundation" "${OUT}/ggml/src/llamafile/sgemm.cpp.NOTICE"
}

@test "takes the terms of a vendored MIT file from LICENSE and keeps code out of the notice" {
  write_ik_vendored_tree
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  grep -q "WITHOUT WARRANTY" "${OUT}/vendor/nlohmann/json.hpp.NOTICE"
  run grep -c -E "Note on documentation|CPPHTTPLIB_VERSION|ggml authors" "${OUT}/vendor/nlohmann/json.hpp.NOTICE" "${OUT}/vendor/cpp-httplib/httplib.h.NOTICE"
  [ "${lines[0]}" = "${OUT}/vendor/nlohmann/json.hpp.NOTICE:0" ]
  [ "${lines[1]}" = "${OUT}/vendor/cpp-httplib/httplib.h.NOTICE:0" ]
}

@test "fails when a vendored MIT file names no copyright holder" {
  write_ik_vendored_tree
  printf '%s\n' "// SPDX-License-Identifier: MIT" "" "int x;" > "${SRC}/vendor/nlohmann/json.hpp"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"vendor/nlohmann/json.hpp: no copyright line in the header, or no MIT terms in LICENSE"* ]]
}

@test "fails when LICENSE holds no MIT terms for a vendored MIT file" {
  write_ik_vendored_tree
  echo "Some other licence" > "${SRC}/LICENSE"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"vendor/cpp-httplib/httplib.h: no copyright line in the header, or no MIT terms in LICENSE"* ]] \
    || [[ "$output" == *"vendor/nlohmann/json.hpp: no copyright line in the header, or no MIT terms in LICENSE"* ]]
}

@test "writes the notice of the stb resize header when the checkout has it" {
  printf '%s\n' "/* stb_image_resize2 - v2.18 - public domain image resizing */" "int x;" "" \
    "This software is available under 2 licenses -- choose whichever you prefer." "ALTERNATIVE A - MIT License" \
    "Copyright (c) 2017 Sean Barrett" > "${SRC}/vendor/stb/stb_image_resize2.h"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -eq 0 ]
  grep -q "Sean Barrett" "${OUT}/vendor/stb/stb_image_resize2.h.NOTICE"
  grep -qx "vendor/stb/stb_image_resize2.h.NOTICE" "${OUT}/MANIFEST"
  run grep -c "int x;" "${OUT}/vendor/stb/stb_image_resize2.h.NOTICE"
  [ "$output" = 0 ]
}

@test "fails when the stb resize header holds no licence text" {
  echo "no licence here" > "${SRC}/vendor/stb/stb_image_resize2.h"
  run "${SCRIPT}" "${SRC}" "${OUT}" "${TERMS}"
  [ "$status" -ne 0 ]
  [[ "$output" == *"stb_image_resize2.h: no 'ALTERNATIVE A - MIT License' in the licence text"* ]]
}
