#!/bin/bash
# Build the CRAM kernel + dax selftests, cross for arm64 and native for x86_64,
# inside the cram-build container (NEVER on the dev host; builds run on llumik).
#
#   ARM64 build  -> proves CONFIG_CRAM compiles for the Pixel's architecture.
#   X86_64 build -> what we actually boot (the harness provisioning is x86-native:
#                   nd_e820 / memmap= e820 pmem), measured under KVM.
#
# Usage (on llumik):
#   docker run --rm -v ~/work/cram/linux:/src -v $PWD:/bench cram-build:1 \
#       bash /bench/build-kernel.sh <arm64|x86_64|both>
set -euo pipefail
SRC=/src
FRAG=/bench/cram.config
J=${J:-8}
what=${1:-both}

merge_and_build() {
    local arch=$1 cross=$2 defcfg=$3 imgtarget=$4 imgpath=$5 out=$6
    echo "===== building $arch into $out ====="
    mkdir -p "$SRC/$out"
    make -C "$SRC" O="$out" ARCH="$arch" ${cross:+CROSS_COMPILE=$cross} "$defcfg"
    "$SRC/scripts/kconfig/merge_config.sh" -O "$SRC/$out" -m \
        "$SRC/$out/.config" "$FRAG"
    make -C "$SRC" O="$out" ARCH="$arch" ${cross:+CROSS_COMPILE=$cross} olddefconfig
    echo "--- effective CRAM/dax/zram config ---"
    grep -E "CONFIG_CRAM|CONFIG_DEV_DAX_KMEM|CONFIG_ZRAM_DEF|CONFIG_NUMA=" "$SRC/$out/.config" || true
    make -C "$SRC" O="$out" ARCH="$arch" ${cross:+CROSS_COMPILE=$cross} -j"$J" "$imgtarget"
    make -C "$SRC" O="$out" ARCH="$arch" ${cross:+CROSS_COMPILE=$cross} -j"$J" modules
    echo "--- artifacts ---"
    ls -lh "$SRC/$out/mm/cram.o" "$SRC/$out/$imgpath"
}

build_selftests() {
    local arch=$1 cross=$2
    echo "===== building dax selftests ($arch) ====="
    make -C "$SRC/tools/testing/selftests/dax" ARCH="$arch" \
        ${cross:+CROSS_COMPILE=$cross} || true
    ls "$SRC"/tools/testing/selftests/dax/cram_*_tool 2>/dev/null || true
}

case "$what" in
  arm64|both)
    merge_and_build arm64 aarch64-linux-gnu- defconfig Image arch/arm64/boot/Image build-arm64
    ;;
esac
case "$what" in
  x86_64|both)
    merge_and_build x86_64 "" x86_64_defconfig bzImage arch/x86/boot/bzImage build-x86_64
    build_selftests x86_64 ""
    ;;
esac
echo "BUILD-DONE $what"
