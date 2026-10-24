#!/bin/bash
# Boot the x86_64 CRAM kernel under KVM with TWO NUMA nodes -- node 0 = DRAM,
# node 1 = a memoryless node carrying an emulated nvdimm that stands in for the
# CRAM compression device (exactly how CRAM's own tests fake the hardware with
# plain memory). Runs the upstream dax/cram selftests + a zstd-zram baseline and
# scrapes the RESULT lines. Run on the llumik HOST (issues docker commands);
# qemu stays inside the container per the no-host-install rule.
set -euo pipefail
W=~/work/cram
BENCH=$W/bench
IMG=cram-build:1
KDIR=/src/build-x86_64

# 1. initramfs (bash + ndctl + selftests)
docker run --rm -v "$W/linux":/src -v "$BENCH":/bench "$IMG" \
    bash /bench/mk-initramfs.sh /bench/initramfs.cpio.gz

# 2. two raw disks for the file-tier tests (pgcache=/dev/vda, file_perf=/dev/vdb)
[ -f "$BENCH/fsdisk.img" ]  || truncate -s 2G "$BENCH/fsdisk.img"
[ -f "$BENCH/fsdisk2.img" ] || truncate -s 2G "$BENCH/fsdisk2.img"

# 3. boot. node1 nvdimm (3G) = the stand-in CRAM device on a memoryless node.
docker run --rm --device /dev/kvm -v "$W/linux":/src -v "$BENCH":/bench "$IMG" \
  qemu-system-x86_64 \
    -enable-kvm -cpu host -smp 4 \
    -machine q35,nvdimm=on \
    -m 4G,slots=4,maxmem=16G \
    -object memory-backend-ram,id=ram0,size=4G \
    -numa node,nodeid=0,memdev=ram0,cpus=0-3 \
    -numa node,nodeid=1 \
    -object memory-backend-ram,id=cram0,size=3058M \
    -device nvdimm,memdev=cram0,id=nv0,node=1,label-size=2M \
    -kernel "$KDIR/arch/x86/boot/bzImage" \
    -initrd /bench/initramfs.cpio.gz \
    -drive file=/bench/fsdisk.img,if=virtio,format=raw \
    -drive file=/bench/fsdisk2.img,if=virtio,format=raw \
    -append "console=ttyS0 panic=-1 rdinit=/init loglevel=4" \
    -nographic -no-reboot -serial mon:stdio \
    2>&1 | tee "$BENCH/vm-x86.log"

echo "=================== SCRAPED RESULTS ==================="
sed -n '/===VM-BEGIN===/,/===VM-END===/p' "$BENCH/vm-x86.log" | \
    grep -E "RESULT |cram-fault|swap-fault|^## .* rc=|balloon|^node |zratio|comp=|Mem" || true

# Host must fail if the guest never reached the end marker, lacked cram debugfs, or reported a failure.
if ! grep -q "===VM-END===" "$BENCH/vm-x86.log"; then
    echo "FAIL: guest did not reach VM-END (crash/hang)"; exit 1
fi
if ! grep -q "CRAM-DEBUGFS-OK" "$BENCH/vm-x86.log"; then
    echo "FAIL: cram debugfs missing in guest (feature absent)"; exit 1
fi
fails="$(sed -n 's/.*===VM-STATUS fails=\([0-9]*\) .*/\1/p' "$BENCH/vm-x86.log" | tail -1)"
if [ -z "$fails" ]; then
    echo "FAIL: no VM-STATUS line from guest"; exit 1
fi
if [ "$fails" -ne 0 ]; then
    echo "FAIL: $fails cram guest test(s) failed"; exit 1
fi
echo "OK: all cram guest tests passed"
