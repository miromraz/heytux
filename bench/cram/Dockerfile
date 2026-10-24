FROM debian:trixie
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential bc bison flex libssl-dev libelf-dev libncurses-dev \
    crossbuild-essential-arm64 kmod cpio git wget ca-certificates xz-utils \
    qemu-system-x86 qemu-system-arm qemu-utils \
    python3 python3-pip python3-setuptools \
    busybox-static ndctl e2fsprogs numactl udev iproute2 util-linux fio \
 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages virtme-ng || pip3 install virtme-ng
