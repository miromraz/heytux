#!/usr/bin/env bash
# Build llama.cpp (CPU-only) for the Pixel 4a running BengalOS (aarch64, glibc 2.43).
#
# Cross-compiles inside a debian:trixie container (arm64 glibc 2.41 <= 2.43 target)
# so llumik's host config is never touched and no qemu/binfmt is needed.
# Target CPU: 2x Cortex-A76 + 6x A55 -> -march=armv8.2-a+dotprod+fp16, -mtune=cortex-a76
# (dotprod yes; no i8mm, no SVE). Q4_0/IQ4_NL runtime repack ON, KleidiAI ON (dotprod kernels).
# Self-contained: shared libs with RPATH $ORIGIN:$ORIGIN/../lib + static libstdc++/libgcc.
#
# Run on llumik (x86_64, docker). Produces ./out/llama.cpp-<TAG>.tar.gz.
#   ssh llumik 'bash -s' < build-llama-cpp.sh
set -euo pipefail

TAG="${TAG:-v0.7.0}"
JOBS="${JOBS:-8}"
WORK="$(mktemp -d)"
cid=""
# clean the tmp dir and (if created) the container on ANY exit, incl. a failed docker build/create
trap 'rm -rf "$WORK"; [ -n "$cid" ] && docker rm -f "$cid" >/dev/null 2>&1 || true' EXIT
OUT="${OUT:-$PWD/out}"
mkdir -p "$OUT"

cat > "$WORK/toolchain.cmake" <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

cat > "$WORK/Dockerfile" <<EOF
FROM debian:trixie
RUN apt-get update && apt-get install -y --no-install-recommends \\
      crossbuild-essential-arm64 cmake ninja-build git ca-certificates pkg-config && \\
    rm -rf /var/lib/apt/lists/*
COPY toolchain.cmake /toolchain.cmake
RUN git clone --depth 1 --branch ${TAG} https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
RUN cmake -S . -B build -G Ninja \\
      -DCMAKE_TOOLCHAIN_FILE=/toolchain.cmake \\
      -DCMAKE_BUILD_TYPE=Release \\
      -DGGML_NATIVE=OFF \\
      -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 \\
      -DGGML_CPU_REPACK=ON \\
      -DGGML_CPU_KLEIDIAI=ON \\
      -DGGML_BACKEND_DL=OFF \\
      -DBUILD_SHARED_LIBS=ON \\
      -DLLAMA_CURL=OFF \\
      -DLLAMA_BUILD_TESTS=OFF \\
      -DLLAMA_BUILD_TOOLS=ON \\
      -DLLAMA_BUILD_EXAMPLES=ON \\
      -DLLAMA_BUILD_SERVER=ON \\
      -DCMAKE_INSTALL_RPATH='\$ORIGIN:\$ORIGIN/../lib' \\
      -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \\
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \\
      -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \\
      -DCMAKE_C_FLAGS="-mtune=cortex-a76" \\
      -DCMAKE_CXX_FLAGS="-mtune=cortex-a76"
RUN cmake --build build -j${JOBS} --target \\
      llama-server llama-cli llama-bench llama-mtmd-cli llama-embedding llama-quantize
# assemble self-contained install tree
RUN set -e; PFX=/opt/llama.cpp-${TAG}; mkdir -p \$PFX/bin \$PFX/lib; \\
    cp build/bin/llama-server build/bin/llama-cli build/bin/llama-bench \\
       build/bin/llama-mtmd-cli build/bin/llama-embedding build/bin/llama-quantize \$PFX/bin/; \\
    cp -a build/bin/*.so* \$PFX/lib/; \\
    echo ${TAG} > \$PFX/TAG; \\
    tar -C /opt -czf /llama.cpp-${TAG}.tar.gz llama.cpp-${TAG}
EOF

docker build -t llama-cross:${TAG} "$WORK"
cid="$(docker create llama-cross:${TAG})"   # EXIT trap (set at top) removes it even if docker cp fails
docker cp "$cid:/llama.cpp-${TAG}.tar.gz" "$OUT/llama.cpp-${TAG}.tar.gz"
echo "ARTIFACT: $OUT/llama.cpp-${TAG}.tar.gz"
