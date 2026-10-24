# CRAM (Meta "Compressed RAM") evaluation for the Pixel 4a kernel

Goal: find out whether Gregory Price's CRAM (LPC 2026, Prague) can save RAM on a
Pixel 4a (SM7150) running our mainline-based 7.1 kernel. Nothing here touches a
phone. All builds/boots run on `llumik` inside the `cram-build` container; never
on the dev host.

## 1. The code (found, public)

- Tree: `github.com/gourryinverse/linux` (Gregory Price, `gourry@gourry.net`).
- Branch used: **`scratch/gourry/cramtest/cram_72`** — the full CRAM stack.
  Base kernel **Linux 7.2.0-rc2**, license **GPL-2.0**.
- Related branches: `private_compression{,_v2,_v3}`, `private_nodes_v7`.
  The "Private Memory NUMA Nodes" base series is on its way upstream; the
  compressed-RAM backend is deliberately kept out of it and posted separately.
- Mainline status: NOT merged. The private-node plumbing may land; the CRAM
  feature itself is a dev branch.

### What CRAM actually is (from the source, not press)
- `mm/cram.c` + `include/linux/cram.h`: a read-only memory *tier*. Clean anon and
  clean file folios are demoted onto an `N_MEMORY_PRIVATE` NUMA node by ordinary
  page **migration**, and mapped **present, read-only** so reads are zero-copy at
  cacheline granularity. A write **faults and promotes** the folio back to DRAM
  (COW for anon, drop-and-refault for file).
- **There is no compression codec anywhere in the core or the dax driver.** CRAM
  assumes *hardware* that compresses the backing while still allowing byte reads.
  A driver calls `cram_register(nid, ranges, zratio, ops, data)`; `zratio` is just
  a number the hardware reports. A **balloon** then reserves
  `perceived * (zratio-current)/zratio` pages so the node's *effective* capacity
  looks larger than its physical size. That balloon is the ONLY thing that models
  "saved RAM".
- Test path (`drivers/dax/kmem.c`, all marked `[TEST]`): bind a dax/kmem device on
  a memoryless node, `echo <zratio> > cram_zratio`, `echo 1 > cram`, online. The
  node is plain DRAM; the balloon simulates the savings. This is exactly how all
  published numbers (incl. the "452x") were produced — plain DRAM, no device.

## 2. Build + boot (done on llumik)

- `build-kernel.sh` cross-builds **arm64** (proves it compiles for the Pixel's
  arch) and native **x86_64** (what we boot; the test harness is x86-native:
  `nd_e820` / `memmap=` e820 pmem / ACPI NFIT).
- `cram.config` is the Kconfig fragment (CRAM + dax/kmem + NFIT + zram-zstd).
- `run-cram-x86.sh` boots the x86 kernel under **KVM** with **two NUMA nodes**:
  node 0 = DRAM, node 1 = memoryless carrying an emulated **nvdimm** that stands in
  for the CRAM device (same trick as upstream's tests). It runs the upstream
  `tools/testing/selftests/dax/cram_*` tests plus a **zstd zram** swap device as
  the baseline, and scrapes the `RESULT` lines.
- `run-arm64-boot.sh` boots the arm64 kernel in **qemu TCG** (functional only).
- `mk-initramfs.sh` / `guest-init.sh` build and drive the test VM.

Why the quantitative run is on x86, not TCG arm64: the CRAM mechanism is
arch-independent core-mm code, and TCG throughput numbers are meaningless for a
zram comparison. The arm64 build + boot prove portability; the x86 KVM run
produces the representative numbers.

### Results
- arm64: Linux **7.2.0-rc2 aarch64** boots under TCG, 2 NUMA nodes, CRAM
  subsystem initialised (`/sys/kernel/debug/cram` present). See `vm-arm64.log`.
- x86_64 CRAM-vs-zram numbers: see `vm-x86.log` and the "SCRAPED RESULTS"
  section it prints. (Summary filled from the run below.)

### Measured (x86_64, KVM, 4 GB DRAM node 0 + 2.75 GB emulated CRAM node 1, zram=zstd)

All upstream cram selftests pass (one flaky balloon-timing subtest in
`cram_node_basic`). Latencies in ns, `cram_perf_tool` / `cram_file_perf_tool`:

Anon tier (cram_perf) -- CRAM vs zram(zstd) swap, head to head:
| path | mean | p50 | p99 |
|-|-|-|-|
| cram-read (in-place, zero-copy) | 30 | 30 | 50 |
| cram-fault (write -> promote to DRAM) | ~3200 | 3130 | 4440-5050 |
| swap-fault (zram zstd swap-in) | ~7500 | 7400 | 11400-11900 |

File tier (cram_file_perf, 128 MB, the llama-like read-only mmap case):
| path | mean | p50 | p99 |
|-|-|-|-|
| file-read (in-place demoted page) | 19 | 20 | 40 |
| file-write (COW-promote off CRAM) | 769 | 20 | 4160 |

- Read of a demoted page is ~20-30 ns (zero-copy, no decompress) vs ~7500 ns for
  a zram swap-in -- the "reads at near-DRAM speed" claim, reproduced. Write-promote
  is ~2.3x faster than a zram swap-in.
- "DRAM freed" is **simulated**: cram_ratio showed zratio=2000 (2:1),
  perceived=720896 pages (2.75 GB); dropping the achieved ratio to 1500 inflated
  the balloon toward target=180224 pages. Node 1 is plain DRAM -- nothing is
  actually compressed; the balloon is the only thing modelling the saving.
- Evidence: `vm-x86.log`, `vm-arm64.log`.

## 3. Software backend? (the decisive question for us)

A software-compressed CRAM node is **not a small addition** to this code, and the
end product would be something we already run:

- CRAM's whole value is pages mapped **present, read-only** so reads are zero-copy
  byte-addressable. Software compression cannot serve a compressed page in place;
  you must **decompress on fault**. Decompress-on-fault behind a not-present page
  *is exactly zswap/zram* — which the Pixel already runs (zstd, 2.7 GB).
- The core never compresses; it only migrates + write-fences. Bolting zstd behind
  the mapping means either (a) keep pages present => you didn't compress => no
  saving, or (b) make them not-present + decompress => you rebuilt zswap. There is
  no middle path without the byte-addressable-compressed hardware CRAM targets.
- Effort to make it actually save RAM in software: that is a research project
  (a new not-present compressed tier with its own fault path), and its best case
  equals today's zram. Not worth writing.

## 4. Verdict

**(c) Not useful on the Pixel 4a now.** The SM7150 has no byte-addressable
compression hardware and no CXL. Porting CRAM would demote pages to *ordinary*
DRAM and report a fake ratio: zero real saving, strictly worse than the zram we
already run. The only software variant that saves RAM collapses into zswap/zram.
Revisit only if the phone ever gains CXL/compressed-memory hardware (it won't).

The private-NUMA-node plumbing underneath CRAM is independently interesting but
orthogonal to saving RAM on this device.
