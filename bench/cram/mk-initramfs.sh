#!/bin/bash
# Assemble a minimal initramfs (cpio.gz) for the CRAM test VM, from inside the
# cram-build container. Pulls bash + busybox + ndctl/daxctl + util-linux +
# e2fsprogs and their shared libs, plus the compiled dax/cram selftests.
#   Run in container:  bash /bench/mk-initramfs.sh /bench/initramfs.cpio.gz
set -euo pipefail
OUT=${1:-/bench/initramfs.cpio.gz}
R=$(mktemp -d)
trap 'rm -rf "$R"' EXIT   # clean the staging dir even if assembly fails partway
mkdir -p "$R"/{bin,sbin,usr/bin,usr/sbin,proc,sys,dev,tmp,opt,etc,lib,lib64}

# copy a binary (resolved via PATH) plus its shared-lib closure, preserving paths
copy() {
    local b p
    for b in "$@"; do
        p=$(command -v "$b" 2>/dev/null || echo "$b")
        [ -e "$p" ] || { echo "WARN missing $b" >&2; continue; }
        cp -aL --parents "$p" "$R" 2>/dev/null || true
        ldd "$p" 2>/dev/null | grep -oE '/[^ ]+\.so[0-9.]*' | sort -u | while read -r l; do
            [ -e "$l" ] && cp -aL --parents "$l" "$R" 2>/dev/null || true
        done || true
    done
}

copy bash busybox ndctl daxctl \
     mkswap swapon swapoff mount umount dmesg modprobe \
     mkfs.ext4 mke2fs \
     awk grep sed cut sort head tail cat ls mkdir sleep sync kill \
     numactl
# ELF interpreter
cp -aL --parents /lib64/ld-linux-x86-64.so.2 "$R" 2>/dev/null || true
cp -aL --parents /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 "$R" 2>/dev/null || true

# busybox applet symlinks for anything we didn't copy as a real binary
if [ -e "$R/usr/bin/busybox" ] || [ -e "$R/bin/busybox" ]; then
    [ -e "$R/usr/bin/busybox" ] && BB=usr/bin/busybox || BB=bin/busybox
    [ -e "$R/bin/busybox" ] || ln -sf "/$BB" "$R/bin/busybox"
    # every applet (real binaries live in /usr/* and win via PATH order)
    for a in $("$R/$BB" --list 2>/dev/null); do
        [ -e "$R/bin/$a" ] || ln -sf "/$BB" "$R/bin/$a"
    done
fi

# the upstream selftests (scripts + compiled tools + ktap helpers)
mkdir -p "$R/opt/dax" "$R/opt/kselftest"
cp -a /src/tools/testing/selftests/dax/. "$R/opt/dax/"
cp -a /src/tools/testing/selftests/kselftest/. "$R/opt/kselftest/" 2>/dev/null || true
# scripts source ../kselftest/ktap_helpers.sh (relative to /opt/dax = /opt/kselftest)

# the cram test scripts carry a #!/bin/bash shebang; bash lives at /usr/bin/bash
[ -e "$R/usr/bin/bash" ] && ln -sf /usr/bin/bash "$R/bin/bash"
cp /bench/guest-init.sh "$R/init"
chmod +x "$R/init"
( cd "$R" && find . | cpio -o -H newc 2>/dev/null | gzip -9 ) > "$OUT"
echo "initramfs: $(ls -lh "$OUT" | awk '{print $5}')  ($OUT)"
rm -rf "$R"
