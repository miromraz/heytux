#!/bin/busybox sh
# PID 1 inside the CRAM test VM. Sets up a zstd zram swap device (the baseline the
# CRAM promote path is measured against), then runs the upstream dax/cram
# selftests, which self-provision the CRAM node from the qemu nvdimm on the
# memoryless NUMA node 1. Output is fenced for host-side scraping.
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
mount -t proc none /proc 2>/dev/null
mount -t sysfs none /sys 2>/dev/null
mount -t devtmpfs none /dev 2>/dev/null
mount -t debugfs none /sys/kernel/debug 2>/dev/null
mount -t tmpfs none /tmp 2>/dev/null
mkdir -p /dev/pts && mount -t devpts none /dev/pts 2>/dev/null

echo "===VM-BEGIN==="
uname -r
echo "--- numa topology ---"
grep -H . /sys/devices/system/node/node*/meminfo 2>/dev/null | grep MemTotal
echo "--- nvdimm regions ---"
ndctl list -R 2>/dev/null || echo "(ndctl: no regions / not present)"
echo "--- node possible/has_memory ---"
cat /sys/devices/system/node/possible /sys/devices/system/node/has_memory 2>/dev/null
echo "--- region numa + create devdax ---"
ndctl list -R 2>/dev/null | grep -iE 'numa|region|target'
for r in $(ndctl list -R 2>/dev/null | grep -oE 'region[0-9]+' | sort -u); do
    ndctl create-namespace -m devdax -e "${r/region/namespace}.0" -f 2>&1 | grep -iE '"dev"|error|numa' || true
done
echo "--- dax devices + target_node ---"
for d in /sys/bus/dax/devices/dax*; do
    [ -e "$d/target_node" ] && echo "$(basename "$d") target_node=$(cat "$d/target_node")"
done

# zstd zram swap == the "plain zram" baseline (phone runs zram+zstd).
echo "--- zram(zstd) swap setup ---"
ZRAM=/sys/block/zram0
if [ -d "$ZRAM" ]; then
    echo zstd > "$ZRAM/comp_algorithm" 2>/dev/null
    echo 3G   > "$ZRAM/disksize"       2>/dev/null
    mkswap /dev/zram0 >/dev/null 2>&1
    swapon -p 100 /dev/zram0 2>/dev/null
    echo "comp=$(cat $ZRAM/comp_algorithm 2>/dev/null)"
fi
cat /proc/swaps

echo "--- block devices ---"
ls -l /dev/vd* 2>/dev/null

cd /opt/dax
# cram_file_perf defaults to /dev/vdb, cram_pgcache to /dev/vda; each mkfs+mounts
# its own disk (both raw virtio disks are supplied by the runner).
export CRAM_FILE_MB=${CRAM_FILE_MB:-128}

# debugfs must be present (a missing cram debugfs means the feature isn't there: a hard failure,
# not a pass). Emit an explicit marker the host checks for.
if [ -e /sys/kernel/debug/cram/nodes ]; then echo "CRAM-DEBUGFS-OK"; else echo "CRAM-DEBUGFS-MISSING"; fi

CRAM_FAILS=0; RAN=0
TESTS="cram_node_basic.sh cram_perf.sh cram_ratio.sh cram_readable.sh cram_file_perf.sh cram_pgcache.sh"
EXPECTED=$(echo $TESTS | wc -w)
run() { echo "########## $1 ##########"; bash "./$1" 2>&1; local rc=$?; echo "## $1 rc=$rc ##";
        RAN=$((RAN+1)); [ "$rc" -ne 0 ] && CRAM_FAILS=$((CRAM_FAILS+1)); }
for t in $TESTS; do
    if [ -f "./$t" ]; then run "$t"; else echo "## MISSING $t ##"; CRAM_FAILS=$((CRAM_FAILS+1)); fi
done
# a run with fewer than the expected number of tests is a failure, not a silent pass
[ "$RAN" -lt "$EXPECTED" ] && CRAM_FAILS=$((CRAM_FAILS+1))
echo "===VM-STATUS fails=$CRAM_FAILS ran=$RAN expected=$EXPECTED==="

echo "--- cram/nodes (balloon = simulated DRAM freed) ---"
cat /sys/kernel/debug/cram/nodes 2>/dev/null
echo "--- demote/promote counters ---"
for f in /sys/kernel/debug/cram/*count* /sys/kernel/debug/cram/*_pages; do
    [ -f "$f" ] && echo "$(basename "$f")=$(cat "$f" 2>/dev/null)"
done
echo "--- meminfo ---"; grep -E "MemTotal|MemFree|MemAvailable" /proc/meminfo
echo "===VM-END==="
sync
poweroff -f 2>/dev/null
echo o > /proc/sysrq-trigger 2>/dev/null
sleep 5
