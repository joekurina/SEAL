# FPGA CKKS Symmetric Encode/Encrypt Pipeline

This document describes the architecture and implementation of the FPGA-accelerated CKKS symmetric encode/encrypt pipeline in Microsoft SEAL. The pipeline targets Intel Agilex 7 FPGAs and is implemented using Intel oneAPI SYCL.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Overview](#overview)
3. [Build Instructions](#build-instructions)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Mathematical Background](#mathematical-background)
6. [Pipeline Stages](#pipeline-stages)
7. [Implementation Details](#implementation-details)
8. [RTL Replacement Guide](#rtl-replacement-guide)
9. [Testing](#testing)

---

## Quick Start

```bash
# Build with FPGA emulator (recommended for development)
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build_fpga_emu \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DSEAL_USE_FPGA=ON \
    -DFPGA_EMULATOR=ON \
    -DSEAL_BUILD_TESTS=ON \
    -DSEAL_USE_CXX17=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build_fpga_emu -j$(nproc)

# Run FPGA tests
./build_fpga_emu/bin/sealtest --gtest_filter="*FPGA*"
```

---

## Overview

The FPGA CKKS pipeline accelerates the symmetric encryption of CKKS plaintexts. Given a vector of complex numbers (slots), it produces a valid SEAL ciphertext that can be decrypted using SEAL's standard `Decryptor`.

### What This Pipeline Does

```
Input:  vector<complex<double>> slots    (up to N/2 complex values)
        SecretKey sk                      (SEAL secret key)
        double scale                      (encoding scale, e.g., 2^40)

Output: Ciphertext ct                     (SEAL-compatible ciphertext)
        where Decrypt(ct, sk) ≈ slots
```

### Pipeline Stages Overview

| Stage | Operation | Domain | Data Type |
|-------|-----------|--------|-----------|
| 1. Prepare | Slot permutation + conjugate pairing | Complex | `complex<double>[N]` |
| 2. DWT Inverse | Inverse Discrete Walsh Transform | Complex → Real | `complex<double>[N]` → `double[N]` |
| 3. Scale & Reduce | Round and reduce mod q | Real → Integer | `double[N]` → `uint64[N]` |
| 4. NTT Forward | Number Theoretic Transform | Coefficient → Evaluation | `uint64[N]` → `uint64[N]` |
| 5. Encrypt | Symmetric encryption formula | Evaluation | `uint64[N]` → `(uint64[N], uint64[N])` |

---

## Build Instructions

### Prerequisites

- **Intel oneAPI Base Toolkit** (2024.0 or later) with DPC++/C++ Compiler (`icpx`)
- **CMake** 3.16 or later
- **C++17** compatible standard library

### Build Modes

| Mode | Flag | Use Case | Compile Time |
|------|------|----------|--------------|
| **Emulator** | `-DFPGA_EMULATOR=ON` | Development, testing, CI | ~2 minutes |
| **Simulator** | `-DFPGA_SIMULATOR=ON` | RTL-level verification | ~30 minutes |
| **Hardware** | `-DFPGA_HARDWARE=ON` | Production FPGA bitstream | ~4-8 hours |

### Emulator Build (Recommended for Development)

The emulator runs SYCL kernels on the CPU, simulating FPGA behavior without requiring hardware.

```bash
# 1. Set up Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh

# 2. Configure CMake with Intel compilers
cmake -S . -B build_fpga_emu \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DSEAL_USE_FPGA=ON \
    -DFPGA_EMULATOR=ON \
    -DSEAL_BUILD_TESTS=ON \
    -DSEAL_USE_CXX17=ON \
    -DCMAKE_BUILD_TYPE=Release

# 3. Build
cmake --build build_fpga_emu -j$(nproc)

# 4. Run tests
./build_fpga_emu/bin/sealtest --gtest_filter="*FPGA*"
```

### Hardware Build (Production)

Generates an actual FPGA bitstream. Requires several hours and significant RAM (~32GB).

```bash
source /opt/intel/oneapi/setvars.sh

cmake -S . -B build_fpga_hw \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_C_COMPILER=icx \
    -DSEAL_USE_FPGA=ON \
    -DFPGA_EMULATOR=OFF \
    -DFPGA_HARDWARE=ON \
    -DSEAL_BUILD_TESTS=ON \
    -DSEAL_USE_CXX17=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build_fpga_hw -j$(nproc)
```

### Host-Only Build (No SYCL Required)

For testing the host reference implementation without Intel oneAPI:

```bash
cmake -S . -B build_fpga_host \
    -DSEAL_BUILD_TESTS=ON \
    -DSEAL_USE_CXX17=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build_fpga_host -j$(nproc)
./build_fpga_host/bin/sealtest --gtest_filter="*FPGA*"
```

---

## Pipeline Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOST (CPU)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  vector<complex<double>> slots     SecretKey sk      EncryptionParameters   │
│              │                          │                    │              │
│              ▼                          ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FPGACKKSEncoder::prepare_for_fpga()              │    │
│  │  • Permute slots according to CKKS index map                        │    │
│  │  • Pair with complex conjugates: v[i], conj(v[i])                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│              │                                                              │
│              ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       FPGAInputPacket                               │    │
│  │  • values[N]: prepared complex slots                                │    │
│  │  • dwt_inv_roots[N]: precomputed DWT twiddle factors                │    │
│  │  • ntt_root_powers[N]: precomputed NTT twiddle factors              │    │
│  │  • secret_key_ntt[N]: secret key in NTT form                        │    │
│  │  • uniform_poly_ntt[N]: random polynomial 'a' (c1)                  │    │
│  │  • error_samples[N]: CBD error samples                              │    │
│  │  • scale, modulus, n, log_n                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                      │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FPGA PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Entrance   │───▶│  DWT Inverse │───▶│Scale & Reduce│                   │
│  │    Kernel    │    │    Kernel    │    │    Kernel    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                           │
│    DWTPacket         ScaleReducePacket      NTTPacket                       │
│                                                 │                           │
│                                                 ▼                           │
│                      ┌──────────────┐    ┌──────────────┐                   │
│                      │     Exit     │◀───│   Encrypt    │                   │
│                      │    Kernel    │    │    Kernel    │                   │
│                      └──────────────┘    └──────────────┘                   │
│                             │                                               │
│                      CiphertextPacket                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOST (CPU)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       FPGAOutputPacket                              │    │
│  │  • c0[N]: first ciphertext component (NTT form)                     │    │
│  │  • c1[N]: second ciphertext component (NTT form)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│              │                                                              │
│              ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Construct SEAL Ciphertext                              │    │
│  │  • Copy c0, c1 to Ciphertext object                                 │    │
│  │  • Set is_ntt_form = true                                           │    │
│  │  • Set scale = input_scale                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│              │                                                              │
│              ▼                                                              │
│         Ciphertext ct   ───▶   SEAL Decryptor   ───▶   vector<double>       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SYCL Pipe Connections

Kernels communicate via Intel SYCL pipes, enabling concurrent execution:

| Pipe | Source | Destination | Data |
|------|--------|-------------|------|
| `EntranceToDWTPipe` | Entrance | DWT Inverse | Complex slots + roots + pass-through |
| `DWTToScaleReducePipe` | DWT Inverse | Scale & Reduce | Real coefficients + pass-through |
| `ScaleReduceToNTTPipe` | Scale & Reduce | NTT Forward | Integer coefficients + pass-through |
| `NTTToEncryptPipe` | NTT Forward | Encrypt | NTT-form plaintext + keys + error |
| `EncryptToExitPipe` | Encrypt | Exit | Ciphertext (c0, c1) |

---

## Mathematical Background

### CKKS Encoding

CKKS encodes a vector of complex numbers $\mathbf{z} \in \mathbb{C}^{N/2}$ into a polynomial $m(X) \in \mathbb{Z}_q[X]/(X^N+1)$:

1. **Canonical Embedding**: Map slots to polynomial coefficients via inverse DFT at primitive roots
2. **Scaling**: Multiply by scale factor $\Delta$ (e.g., $2^{40}$) to preserve precision
3. **Rounding**: Round to nearest integer
4. **Reduction**: Reduce modulo $q$

### Symmetric Encryption

Given plaintext polynomial $m$ in NTT form, secret key $s$, uniform random $a$, and error $e$:

$$c_1 = a$$
$$c_0 = m - a \cdot s + e \pmod{q}$$

**Decryption** recovers $m + e$:
$$c_0 + c_1 \cdot s = m - a \cdot s + e + a \cdot s = m + e$$

### DWT Inverse Roots Formula

The DWT inverse roots must match SEAL's internal representation:

$$\text{dwt\_inv\_roots}[i] = e^{-2\pi j \cdot (\text{reverse\_bits}(i-1, \log_2 N) + 1) / (2N)}$$

For $i = 1, 2, \ldots, N-1$. The root at index 0 is unused.

```cpp
for (size_t i = 1; i < n; i++) {
    size_t rev_idx = reverse_bits(i - 1, log_n) + 1;
    double angle = -2.0 * M_PI * rev_idx / (2.0 * n);
    dwt_inv_roots_[i] = complex<double>(cos(angle), sin(angle));
}
```

---

## Pipeline Stages

### Stage 1: Entrance Kernel

**Purpose**: Unpack host data and prepare for DWT

**Operations**:
1. Read `FPGAInputPacket` from host memory
2. Compute scale factor: `scale_factor = scale / N`
3. Package data into `DWTPacket` with all pass-through fields
4. Write to `EntranceToDWTPipe`

### Stage 2: DWT Inverse Kernel

**Purpose**: Transform complex slots to real polynomial coefficients

**Operations**:
1. Read `DWTPacket` from pipe
2. Execute Radix-2 DIF IFFT on complex values using provided roots
3. Multiply all values by `scale_factor`
4. Package into `ScaleReducePacket`
5. Write to `DWTToScaleReducePipe`

**Algorithm**: Decimation-in-Frequency (DIF) butterfly

```
for stage in 0..log_n:
    gap = 1 << stage
    for group in 0..(n / (2 * gap)):
        for j in 0..gap:
            a = values[group * 2 * gap + j]
            b = values[group * 2 * gap + j + gap]
            twiddle = roots[...]
            values[...] = a + b
            values[...] = (a - b) * twiddle
```

### Stage 3: Scale & Reduce Kernel

**Purpose**: Convert floating-point to modular integers

**Operations**:
1. Read `ScaleReducePacket` from pipe
2. For each coefficient:
   - Round real part to nearest integer
   - Apply Barrett reduction to get value mod q
   - Handle negative values (add q if negative)
3. Package into `NTTPacket`
4. Write to `ScaleReduceToNTTPipe`

**Barrett Reduction**: For 128-bit products, uses optimized reduction without overflow.

### Stage 4: NTT Forward Kernel

**Purpose**: Transform plaintext to evaluation (NTT) domain

**Operations**:
1. Read `NTTPacket` from pipe
2. Execute forward NTT using precomputed root powers
3. Package into `EncryptPacket` with plaintext_ntt
4. Write to `NTTToEncryptPipe`

**Algorithm**: Cooley-Tukey Radix-2 DIT NTT with Montgomery/Barrett modular multiplication.

### Stage 5: Symmetric Encryption Kernel

**Purpose**: Apply encryption formula to produce ciphertext

**Operations**:
1. Read `EncryptPacket` from pipe
2. Convert error samples to NTT form (in-kernel NTT)
3. Compute: `as = a * s` (component-wise mod q)
4. Compute: `neg_as = -as mod q`
5. Compute: `c0 = neg_as + e + m mod q`
6. Set: `c1 = a`
7. Package into `CiphertextPacket`
8. Write to `EncryptToExitPipe`

### Stage 6: Exit Kernel

**Purpose**: Write ciphertext back to host memory

**Operations**:
1. Read `CiphertextPacket` from pipe
2. Copy `c0` and `c1` to `FPGAOutputPacket`
3. Signal completion

---

## Implementation Details

### Packet Pass-Through Mechanism

Each packet carries data needed by downstream kernels:

```
FPGAInputPacket
├── values[N]           → Used by DWT
├── dwt_inv_roots[N]    → Used by DWT  
├── scale               → Used by DWT (as scale/N)
├── ntt_root_powers[N]  → Pass-through to NTT, Encrypt
├── secret_key_ntt[N]   → Pass-through to Encrypt
├── uniform_poly_ntt[N] → Pass-through to Encrypt (becomes c1)
├── error_samples[N]    → Pass-through to Encrypt
├── modulus             → Pass-through to all
├── n, log_n            → Pass-through to all
└── barrett_ratio[2]    → Pass-through to Scale&Reduce, NTT, Encrypt
```

### 128-bit Modular Arithmetic

The `mod_u128` function handles 128-bit values without overflow:

```cpp
inline uint64_t mod_u128(uint128_t val, uint64_t modulus) {
    if (val.hi == 0) return val.lo % modulus;
    
    uint64_t result = 0;
    uint64_t base = 1;
    
    for (int i = 0; i < 64; i++) {
        if ((val.lo >> i) & 1) {
            result += base;
            if (result >= modulus) result -= modulus;
        }
        base <<= 1;
        if (base >= modulus) base -= modulus;
    }
    
    for (int i = 0; i < 64; i++) {
        if ((val.hi >> i) & 1) {
            result += base;
            if (result >= modulus) result -= modulus;
        }
        base <<= 1;
        if (base >= modulus) base -= modulus;
    }
    
    return result;
}
```

### Floating-Point Precision

The DWT uses double-precision arithmetic. Due to rounding, FPGA encoding may differ from SEAL by ±1 in rare coefficients (typically 0-1 out of N for N ≤ 32768). This does not affect correctness—the encrypt/decrypt cycle produces identical results to SEAL.

### Memory Layout

Each kernel uses on-chip buffers sized for maximum polynomial degree:

```cpp
constexpr size_t MAX_POLY_DEGREE = 32768;

// In each kernel:
complex<double> local_values[MAX_POLY_DEGREE];
uint64_t local_coeffs[MAX_POLY_DEGREE];
```

This keeps high-bandwidth transformations entirely on-chip, avoiding DDR/HBM latency.

---

## RTL Replacement Guide

The DWT Inverse kernel is designed for RTL replacement of the IFFT core.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DWT Inverse Kernel                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. WRAPPER: Load data from pipe                                    │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  DWTPacket pkt = EntranceToDWTPipe::read()              │     │
│     │  Copy pkt.values → local_values[]                       │     │
│     │  Copy pkt.inv_roots → local_roots[]                     │     │
│     └─────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ╔═════════════════════════════════════════════════════════════╗    │
│  ║           IFFT CORE - RTL REPLACEMENT POINT                 ║    │
│  ╠═════════════════════════════════════════════════════════════╣    │
│  ║  ifft_dif_core(local_values, local_roots, n)                ║    │
│  ║                                                             ║    │
│  ║  Input:  N complex<double> (bit-reversed)                   ║    │
│  ║  Output: N complex<double> (natural order)                  ║    │
│  ╚═════════════════════════════════════════════════════════════╝    │
│                              │                                      │
│                              ▼                                      │
│  3. WRAPPER: Post-process and write to pipe                         │
│     ┌─────────────────────────────────────────────────────────┐     │
│     │  for i in 0..n: local_values[i] *= scale_factor         │     │
│     │  DWTToScaleReducePipe::write(out_pkt)                   │     │
│     └─────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### RTL Module Interface

```verilog
module ifft_core #(
    parameter N = 32768,
    parameter DATA_WIDTH = 128  // complex<double>
)(
    input  wire                   clk,
    input  wire                   rst_n,
    
    input  wire [DATA_WIDTH-1:0]  values_in_data,
    input  wire                   values_in_valid,
    output wire                   values_in_ready,
    
    input  wire [DATA_WIDTH-1:0]  roots_in_data,
    input  wire                   roots_in_valid,
    output wire                   roots_in_ready,
    
    output wire [DATA_WIDTH-1:0]  values_out_data,
    output wire                   values_out_valid,
    input  wire                   values_out_ready
);
```

### Integration Steps

1. Create RTL module with Avalon-ST interfaces
2. Create RTL spec file (`ifft_core_spec.xml`)
3. Replace `ifft_dif_core()` body with RTL call
4. Compile: `icpx -fsycl -fintelfpga -Xshardware -Xsrtl-spec=... -Xsrtl=...`

---

## Testing

### Test Suites

| Suite | Tests | Description |
|-------|-------|-------------|
| `FPGAEncoderTest` | 16 | Slot permutation, DWT roots, samplers |
| `FPGADeviceTest` | 2 | SYCL device/context initialization |
| `FPGAPipelineTest` | 24 | Individual stages + end-to-end validation |
| `FPGACKKSCompatibilityTest` | 40 | Full encrypt/decrypt with various inputs |

### Key Validation Tests

| Test | Validates |
|------|-----------|
| `FullPipelineHostMatchesSEAL` | Complete encode → encrypt → SEAL decrypt cycle |
| `EncryptionFormulaWithSEAL` | $c_0 + c_1 \cdot s = m$ (zero error case) |
| `DWTMatchesSEAL` | DWT output matches SEAL encoding (coefficient form) |
| `NTTMatchesSEAL` | NTT implementation matches SEAL |

### Running Tests

```bash
# All FPGA tests
./build_fpga_emu/bin/sealtest --gtest_filter="*FPGA*"

# Specific test suite
./build_fpga_emu/bin/sealtest --gtest_filter="*FPGAPipelineTest*"

# Key validation only
./build_fpga_emu/bin/sealtest --gtest_filter="*FullPipelineHostMatchesSEAL*"

# Single polynomial degree
./build_fpga_emu/bin/sealtest --gtest_filter="*N8192*"
```

### Current Status

All 82 tests pass on both host-only and FPGA emulator builds.

---

## Files Reference

| File | Purpose |
|------|---------|
| `Inc/fpga_pipeline.h` | Main pipeline class declaration |
| `Inc/fpga_packets.h` | Packet structure definitions |
| `Inc/fpga_pipes.h` | SYCL pipe declarations |
| `Inc/fpga_ifft_core.h` | IFFT core (RTL replacement point) |
| `Inc/fpga_arith.h` | Modular arithmetic primitives |
| `Src/fpga_pipeline.cpp` | Pipeline orchestration |
| `Src/fpga_entrance_kernel.cpp` | Entrance kernel |
| `Src/fpga_dwt_kernel.cpp` | DWT Inverse kernel |
| `Src/fpga_ntt_kernel.cpp` | NTT Forward kernel (placeholder) |
| `Src/fpga_encrypt_kernel.cpp` | Encryption kernel |
| `Src/fpga_exit_kernel.cpp` | Exit kernel |
| `Src/fpga_dwt.cpp` | Host DWT implementation |
| `Src/fpga_ntt.cpp` | Host NTT implementation |
| `Src/fpga_encrypt.cpp` | Host encryption implementation |
| `Src/fpga_ckks_encoder.cpp` | CKKS encoder (slot preparation) |
| `Src/fpga_ckks_context.cpp` | Context and parameter management |
