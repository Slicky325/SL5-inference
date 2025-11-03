# CAmkES-VM Setup Guide for seL4

## Running LLM Inference on seL4 using Linux VM

This guide covers **Option 1** from the seL4 Compatibility Report: running your LLM inference engines inside a Linux guest VM on top of seL4 using CAmkES-VM.

### Why This Approach?

- ✅ Maintains seL4's formal verification and security isolation
- ✅ Provides full Linux environment with all dependencies (libc, POSIX, filesystem, threads)
- ✅ No need to port inference engines - run them unchanged
- ✅ Supported by seL4 project with mature tooling
- ✅ Can communicate with native seL4 components via shared memory/RPC

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Building seL4 with CAmkES-VM](#building-sel4-with-camkes-vm)
4. [Creating a Linux Guest](#creating-a-linux-guest)
5. [Integrating Inference Engines](#integrating-inference-engines)
6. [Communication Between seL4 and Linux VM](#communication-between-sel4-and-linux-vm)
7. [Building and Running](#building-and-running)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **CPU**: x86_64 or ARM with virtualization extensions
  - Intel: VT-x (vmx)
  - AMD: AMD-V (svm)
  - ARM: ARM Virtualization Extensions (Cortex-A15 or later)
- **RAM**: Minimum 8GB (16GB+ recommended for LLM inference)
- **Storage**: 20GB+ free space

### Software Requirements

- Linux host system (Ubuntu 20.04/22.04 recommended)
- Python 3.6+
- Git
- Build tools and dependencies

---

## Environment Setup

### 1. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install seL4 build dependencies
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    gcc-multilib \
    g++-multilib \
    libxml2-utils \
    ncurses-dev \
    libssl-dev \
    libsqlite3-dev \
    libcunit1-dev \
    expect \
    qemu-system-x86 \
    qemu-system-arm \
    device-tree-compiler \
    u-boot-tools \
    curl \
    git \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv

# Install Python packages
pip3 install --user \
    sel4-deps \
    camkes-deps \
    ply \
    pyelftools \
    pyfdt
```

### 2. Install repo Tool

```bash
mkdir -p ~/.local/bin
curl https://storage.googleapis.com/git-repo-downloads/repo > ~/.local/bin/repo
chmod a+x ~/.local/bin/repo

# Add to PATH if not already
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Set Up Rust Toolchain (for Rust-based inference)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup default stable
```

---

## Building seL4 with CAmkES-VM

### 1. Initialize CAmkES-VM Project

```bash
# Create workspace directory
mkdir -p ~/seL4-vm-workspace
cd ~/seL4-vm-workspace

# Initialize repo with CAmkES VM manifest
repo init -u https://github.com/seL4/camkes-vm-manifest.git

# Sync all repositories
repo sync
```

This will download:
- seL4 microkernel
- CAmkES component framework
- VMM (Virtual Machine Monitor) libraries
- Example VM projects

### 2. Explore CAmkES-VM Examples

```bash
cd ~/seL4-vm-workspace
ls apps/

# You should see example VM applications:
# - vm_minimal      - Minimal VM example
# - vm_multi        - Multiple VMs example  
# - vm_virtio       - VM with VirtIO devices
```

### 3. Configure Build for x86_64

```bash
cd ~/seL4-vm-workspace

# Use the minimal VM example as starting point
mkdir build-x86
cd build-x86

# Configure CMake
../init-build.sh -DPLATFORM=x86_64 -DCAMKES_VM_APP=vm_minimal
```

For ARM (e.g., QEMU virt-a15):
```bash
mkdir build-arm
cd build-arm
../init-build.sh -DPLATFORM=qemu-arm-virt -DCAMKES_VM_APP=vm_minimal
```

### 4. Build the System

```bash
ninja
```

This produces:
- `images/capdl-loader-image-x86_64-pc99` (bootable image)
- VM configuration files
- seL4 kernel and CAmkES components

---

## Creating a Linux Guest

### 1. Choose a Linux Distribution

**Recommended options:**
- **Buildroot**: Minimal, customizable (best for embedded)
- **Alpine Linux**: Lightweight, fast boot
- **Debian/Ubuntu**: Full-featured, easier package management

### 2. Build Buildroot Linux Guest

```bash
cd ~/seL4-vm-workspace

# Clone Buildroot
git clone https://github.com/buildroot/buildroot.git
cd buildroot

# Use default x86_64 config
make qemu_x86_64_defconfig

# Customize configuration
make menuconfig
```

**Key configurations:**
- Target options → x86_64
- Filesystem images → ext2/3/4 root filesystem
- Kernel → Linux kernel version 5.x or 6.x
- System configuration → Enable root login
- Target packages → Enable networking, development tools

**For LLM inference, also enable:**
- Toolchain → Enable C++ support
- Toolchain → Enable WCHAR support
- Target packages → Libraries → Other → Enable libstdc++
- Target packages → Development tools → gcc, make, cmake
- Target packages → Interpreter languages → python3
- Target packages → Networking → openssh

```bash
# Build (this takes 20-60 minutes)
make -j$(nproc)
```

Output files:
- `output/images/bzImage` - Linux kernel
- `output/images/rootfs.ext4` - Root filesystem

### 3. Prepare Guest Images for CAmkES-VM

```bash
# Copy to CAmkES-VM project
cd ~/seL4-vm-workspace
mkdir -p apps/vm_minimal/overlay_files

# Copy kernel and rootfs
cp ~/seL4-vm-workspace/buildroot/output/images/bzImage \
   apps/vm_minimal/overlay_files/linux_kernel

cp ~/seL4-vm-workspace/buildroot/output/images/rootfs.ext4 \
   apps/vm_minimal/overlay_files/linux_rootfs.ext4
```

### 4. Configure CAmkES VM Component

Edit `apps/vm_minimal/vm.camkes`:

```c
import <std_connector.camkes>;
import <global-connectors.camkes>;
import <seL4VMDTBPassthrough.idl4>;
import <FileServerInterface.camkes>;
import <FileServer/FileServer.camkes>;
import <SerialServer/SerialServer.camkes>;
import <TimeServer/TimeServer.camkes>;
import <vm-connectors.camkes>;
import <devices.camkes>;

component VM {
    VM_INIT_DEF()
}

assembly {
    composition {
        VM_COMPOSITION_DEF()
        VM_PER_VM_COMP_DEF(0)
        
        component FileServer fserv;
        component SerialServer serial;
        component TimeServer time_server;
        
        // Connect VM to file server for rootfs
        connection seL4RPCDataport fs(from vm0.fs, to fserv.fs_ctrl);
    }
    
    configuration {
        VM_CONFIGURATION_DEF()
        VM_PER_VM_CONFIG_DEF(0)
        
        vm0.simple_untyped23_pool = 20;  // Adjust memory
        vm0.heap_size = 0x2000000;       // 32MB heap
        vm0.guest_ram_mb = 4096;         // 4GB guest RAM
        
        // Kernel and rootfs images
        vm0.kernel_cmdline = "console=ttyS0 root=/dev/vda rw";
        vm0.kernel_image = "linux_kernel";
        vm0.kernel_relocs = "linux_kernel";
        vm0.initrd_image = "linux_rootfs.ext4";
    }
}
```

### 5. Rebuild with Linux Guest

```bash
cd ~/seL4-vm-workspace/build-x86
ninja
```

---

## Integrating Inference Engines

### 1. Cross-Compile or Include in Guest

**Option A: Build in Guest** (easier, slower)
- Boot the Linux VM
- Install build tools inside guest
- Clone and build inference engines inside VM

**Option B: Cross-Compile** (faster runtime setup)
- Cross-compile on host for guest architecture
- Include binaries in rootfs overlay

### 2. Add Inference Files to Buildroot

```bash
cd ~/seL4-vm-workspace/buildroot

# Create overlay directory for custom files
mkdir -p overlay/root/inference

# Copy your inference code
cp -r /path/to/SL5-inference/* overlay/root/inference/

# Reconfigure Buildroot to use overlay
make menuconfig
# → System configuration → Root filesystem overlay directories
# → Add: $(TOPDIR)/overlay

# Rebuild
make -j$(nproc)
```

### 3. Model File Handling

**Challenge**: LLM models are large (2-10GB+)

**Solution options:**

**A. VirtIO Block Device** (recommended)
```bash
# Create a separate disk image for models
dd if=/dev/zero of=models.img bs=1M count=10240  # 10GB
mkfs.ext4 models.img

# Mount and copy models
mkdir /tmp/models_mount
sudo mount -o loop models.img /tmp/models_mount
sudo cp models/*.gguf /tmp/models_mount/
sudo umount /tmp/models_mount
```

Configure in `vm.camkes`:
```c
vm0.disk_images = ["linux_rootfs.ext4", "models.img"];
```

**B. 9P Filesystem Sharing** (share host directory)
```c
// In vm.camkes configuration
vm0.fs_passthrough = true;
vm0.host_share_path = "/path/to/models";
```

**C. Network Transfer** (if VM has network)
- Set up SSH/SCP into guest
- Transfer models after boot

---

## Communication Between seL4 and Linux VM

### Shared Memory Communication

For integrating inference with native seL4 components:

**1. Set up shared memory region in CAmkES:**

```c
// In your CAmkES assembly
component InferenceClient {
    dataport Buf(8192) inference_request;
    dataport Buf(8192) inference_response;
    emits InferenceEvent request_event;
    consumes InferenceEvent response_event;
}

connection seL4SharedData mem1(from inference_client.inference_request,
                                to vm0.shared_mem_request);
connection seL4SharedData mem2(from vm0.shared_mem_response,
                                to inference_client.inference_response);
```

**2. Inside Linux VM, access via `/dev/mem` or virtio-driver:**

```c
// Example: Linux userspace accessing shared memory
#include <sys/mman.h>
#include <fcntl.h>

int fd = open("/dev/mem", O_RDWR | O_SYNC);
void* shared = mmap(NULL, 8192, PROT_READ | PROT_WRITE, 
                     MAP_SHARED, fd, SHARED_MEM_PHYS_ADDR);

// Read request from seL4 component
struct inference_request* req = (struct inference_request*)shared;

// Run inference
run_inference(req->prompt, response_buffer);

// Write response back
memcpy(shared, response_buffer, response_size);
```

**3. Use VirtIO console for simple RPC:**

```c
// In VM: read from /dev/vport0p1
int fd = open("/dev/vport0p1", O_RDWR);
char request[256];
read(fd, request, sizeof(request));

// Process and respond
char response[256];
sprintf(response, "Result: %s", process(request));
write(fd, response, strlen(response));
```

---

## Building and Running

### 1. Final Build

```bash
cd ~/seL4-vm-workspace/build-x86
ninja
```

### 2. Run in QEMU

```bash
# Run the built image
./simulate

# Or manually with QEMU:
qemu-system-x86_64 \
    -m 8G \
    -cpu Broadwell \
    -serial mon:stdio \
    -nographic \
    -kernel images/kernel-x86_64-pc99 \
    -initrd images/capdl-loader-image-x86_64-pc99 \
    -append "console=ttyS0,115200"
```

**Expected boot sequence:**
1. seL4 kernel boots
2. CAmkES component system initializes
3. VMM starts Linux guest
4. Linux boots in guest
5. Login prompt appears

### 3. Access Linux Guest

```bash
# At Linux login prompt
login: root
# (no password if configured in Buildroot)

# Test inference
cd /root/inference
ls -la

# Run llama.cpp example
./llamacpp-inf/llama.cpp/build/bin/main \
    -m /mnt/models/llama-2-7b.Q4_K_M.gguf \
    -p "Hello, world!" \
    -n 50
```

### 4. Run on Real Hardware

**Create bootable USB/SD card:**

```bash
# For x86 PC
dd if=images/capdl-loader-image-x86_64-pc99 of=/dev/sdX bs=4M
sync

# Boot target machine from USB
```

**Configure BIOS/UEFI:**
- Enable Intel VT-x or AMD-V
- Disable Secure Boot (if conflicts)

---

## Troubleshooting

### VM Fails to Boot

**Check kernel command line:**
```c
vm0.kernel_cmdline = "console=ttyS0 root=/dev/vda rw debug";
```

**Verify initrd is included:**
```bash
ls -lh apps/vm_minimal/overlay_files/
```

**Check memory allocation:**
```c
vm0.guest_ram_mb = 4096;  // Increase if needed
```

### Inference Runs Slowly

- **Increase vCPUs**: Modify `vm0.num_vcpus = 4;` in configuration
- **Check CPU pinning**: Ensure guest isn't preempted
- **Disable debug logging**: Build release version
- **Profile**: Use `perf` inside guest to find bottlenecks

### Out of Memory in Guest

```c
// Increase guest RAM in vm.camkes
vm0.guest_ram_mb = 8192;  // 8GB

// Rebuild
ninja
```

### Model File Not Found

```bash
# Inside guest, check mount points
mount | grep vd
ls -la /dev/vd*

# Manually mount second disk
mkdir /mnt/models
mount /dev/vdb /mnt/models
```

### Cannot Access Shared Memory

**Check physical memory mappings:**
```bash
# On host, in seL4 build
cat build-x86/capdl/vm0.h | grep SHARED_MEM
```

**In guest, verify `/dev/mem` access:**
```bash
ls -la /dev/mem
# If missing, enable in Buildroot:
# Device Drivers → Character devices → /dev/mem virtual device support
```

### Networking Issues

**Add VirtIO network to CAmkES:**
```c
vm0.virtio_net = true;
vm0.virtio_net_mac = "02:00:00:00:00:01";
```

**Inside guest:**
```bash
ip link show
ip addr add 10.0.0.2/24 dev eth0
ip link set eth0 up
```

---

## Performance Optimization

### 1. CPU Optimization

- **Enable HugePages** in guest for large model allocations
- **Pin guest to physical cores** via CAmkES configuration
- **Use highest optimization level** for inference builds (`-O3 -march=native`)

### 2. Memory Optimization

- **Reduce guest overhead**: Minimal Linux distribution
- **Use memory balloon** for dynamic allocation
- **Configure swap** if needed (though slower)

### 3. I/O Optimization

- **Use VirtIO devices** (faster than emulated hardware)
- **Mount models.img with cache=writeback**
- **Consider tmpfs for temporary inference data**

---

## Next Steps

### Extend This Setup

1. **Add Multiple VMs**: Run different models in isolated VMs
2. **Implement RPC Layer**: Clean API between seL4 and inference
3. **Add GPU Passthrough**: Pass NVIDIA GPU to Linux VM (advanced)
4. **Resource Monitoring**: Track VM resource usage from seL4
5. **Secure Channels**: Encrypt communication between components

### Development Workflow

```bash
# Iterative development cycle
cd ~/seL4-vm-workspace/build-x86

# 1. Modify CAmkES components or VM config
vim ../apps/vm_minimal/vm.camkes

# 2. Rebuild
ninja

# 3. Test in QEMU
./simulate

# 4. Update guest rootfs if needed
cd ../buildroot
# ... make changes ...
make -j$(nproc)
cp output/images/rootfs.ext4 ../apps/vm_minimal/overlay_files/

# 5. Rebuild and test
cd ../build-x86
ninja
./simulate
```

---

## Resources

### Official Documentation
- [seL4 CAmkES-VM Documentation](https://docs.sel4.systems/projects/camkes-vm/)
- [seL4 Virtualization](https://docs.sel4.systems/projects/virtualization/)
- [CAmkES Manual](https://docs.sel4.systems/projects/camkes/)
- [seL4 Tutorial](https://docs.sel4.systems/Tutorials/)

### Community
- [seL4 Discourse Forum](https://sel4.discourse.group/)
- [seL4 GitHub](https://github.com/seL4)
- [seL4 Mattermost Chat](https://mattermost.trustworthy.systems/)

### Related Projects
- [Genode with seL4](https://genode.org/documentation/platforms/sel4)
- [CantripOS](https://github.com/AmbiML/sparrow-cantrip-full) - ML on seL4
- [seL4 Microkit](https://github.com/seL4/microkit) - Simplified framework

---

## Summary

This setup provides:
- ✅ **Security**: seL4's formal guarantees + VM isolation
- ✅ **Compatibility**: Run any Linux application unchanged
- ✅ **Performance**: Near-native speed with hardware virtualization
- ✅ **Flexibility**: Easy integration with seL4 components
- ✅ **Maintainability**: Standard Linux tooling and workflows

**Tradeoffs:**
- ❌ Slightly higher resource overhead (VM + hypervisor)
- ❌ More complex build process
- ❌ Guest OS attack surface (mitigated by seL4 isolation)

For LLM inference workloads, this is the most practical approach to leverage seL4's security properties while maintaining full compatibility with existing inference engines.

