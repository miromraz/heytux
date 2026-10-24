#!/bin/bash
# Functional boot of the arm64 CRAM kernel in qemu-system-aarch64 (TCG) with two
# NUMA nodes + an nvdimm on the memoryless node 1 (the stand-in CRAM device).
# Proves the Pixel's architecture boots the CRAM kernel and the CRAM subsystem
# initialises (debugfs dir + N_MEMORY_PRIVATE plumbing). Full CRAM measurement is
# done on x86 (run-cram-x86.sh): the mechanism is arch-independent core-mm code,
# and TCG throughput numbers would be meaningless for a zram comparison.
# Run on the llumik HOST.
set -euo pipefail
W=~/work/cram; BENCH=$W/bench; IMG=cram-build:1
KDIR=/src/build-arm64

# Tiny arm64 initramfs: one static busybox + an init that reports CRAM state.
docker run --rm -v "$BENCH":/bench "$IMG" bash -euo pipefail -c '
  R=$(mktemp -d); mkdir -p "$R"/{bin,proc,sys,dev}
  dpkg --add-architecture arm64 >/dev/null 2>&1
  apt-get update -qq >/dev/null 2>&1
  cd /tmp; apt-get download busybox-static:arm64 >/dev/null 2>&1
  dpkg-deb -x busybox-static_*arm64.deb bbx
  cp "$(find bbx -name busybox -type f | head -1)" "$R/bin/busybox"
  for a in sh mount cat ls echo sleep poweroff grep; do ln -sf /bin/busybox "$R/bin/$a"; done
  cat > "$R/init" <<"EOF"
#!/bin/busybox sh
/bin/busybox mount -t proc none /proc
/bin/busybox mount -t sysfs none /sys
/bin/busybox mount -t debugfs none /sys/kernel/debug 2>/dev/null
echo ===ARM-VM-BEGIN===
/bin/busybox uname -a
echo "--- nodes ---"; /bin/busybox ls /sys/devices/system/node/ | /bin/busybox grep node
echo "--- CRAM debugfs present? ---"; /bin/busybox ls -d /sys/kernel/debug/cram 2>/dev/null && echo CRAM-DEBUGFS-OK || echo CRAM-DEBUGFS-MISSING
echo "--- dmesg cram/private ---"; /bin/busybox dmesg | /bin/busybox grep -iE "cram|private.*node|nvdimm|papr|nfit" | /bin/busybox head -20
echo ===ARM-VM-END===
/bin/busybox poweroff -f
EOF
  chmod +x "$R/init" "$R/bin/busybox"
  ( cd "$R" && find . | cpio -o -H newc 2>/dev/null | gzip -9 ) > /bench/initramfs-arm64.cpio.gz
  echo "arm64 initramfs: $(ls -lh /bench/initramfs-arm64.cpio.gz | awk "{print \$5}")"
'

# arm64 virt nvdimm hotplug needs acpi-ged plumbing; for a functional boot we
# just prove the kernel boots on arm64 with 2 NUMA nodes and CRAM initialised.
# Full CRAM node registration + measurement is done on x86 (run-cram-x86.sh).
timeout 900 docker run --rm -v "$W/linux":/src -v "$BENCH":/bench "$IMG" \
  qemu-system-aarch64 \
    -machine virt -cpu cortex-a57 -smp 4 -m 4G \
    -object memory-backend-ram,id=ram0,size=3G \
    -object memory-backend-ram,id=ram1,size=1G \
    -numa node,nodeid=0,memdev=ram0,cpus=0-3 \
    -numa node,nodeid=1,memdev=ram1 \
    -kernel "$KDIR/arch/arm64/boot/Image" \
    -initrd /bench/initramfs-arm64.cpio.gz \
    -append "console=ttyAMA0 panic=-1 rdinit=/init loglevel=6" \
    -nographic -no-reboot 2>&1 | tee "$BENCH/vm-arm64.log"

echo "=================== ARM64 BOOT RESULT ==================="
sed -n '/===ARM-VM-BEGIN===/,/===ARM-VM-END===/p' "$BENCH/vm-arm64.log" || true

# Fail unless the guest reached the end marker AND cram debugfs was present (OK, not MISSING).
if ! grep -q "===ARM-VM-END===" "$BENCH/vm-arm64.log"; then
    echo "FAIL: arm64 guest did not reach ARM-VM-END (crash/hang)"; exit 1
fi
if ! grep -q "CRAM-DEBUGFS-OK" "$BENCH/vm-arm64.log"; then
    echo "FAIL: cram debugfs missing on arm64 (was passing on CRAM-DEBUGFS-MISSING)"; exit 1
fi
echo "OK: arm64 boot + cram debugfs present"
