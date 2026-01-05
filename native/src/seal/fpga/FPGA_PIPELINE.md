# FPGA CKKS Symmetric Encode/Encrypt Pipeline

This document describes the architecture and implementation of the FPGA-accelerated CKKS symmetric encode/encrypt pipeline in Microsoft SEAL. The pipeline targets Intel Agilex 7 FPGAs and is implemented using Intel oneAPI SYCL.

## Overview

The FPGA CKKS pipeline accelerates the most computationally intensive parts of the CKKS encryption process:
1.  **Encoding**: Inverse Discrete Walsh-Hadamard-like Transform (DWT) and scaling/rounding.
2.  **NTT**: Forward Number Theoretic Transform on plaintext coefficients and error samples.
3.  **Symmetric Encryption**: Modular arithmetic to produce ciphertext components ($c_0, c_1$).

The design uses a modular, kernel-based architecture where data flows through a series of kernels connected by SYCL pipes. This approach allows for overlapping execution of different stages and enables the replacement of specific kernels with highly optimized RTL implementations (e.g., generated via DSP Builder) without re-architecting the entire pipeline.

## Architecture

The pipeline consists of six modular kernels running as independent `single_task` units. Data is passed between kernels using `sycl::ext::intel::pipe`.

### Data Flow Diagram

```text
       Host Memory
            |
    (FPGAInputPacket)
            |
            v
    +----------------+
    |    Entrance    |
    +----------------+
            | (DWTPacket)
            v
    +----------------+
    |  DWT Inverse   |
    +----------------+
            | (ScaleReducePacket)
            v
    +----------------+
    | Scale & Reduce |
    +----------------+
            | (NTTPacket)
            v
    +----------------+
    |  NTT Forward   |
    +----------------+
            | (EncryptPacket)
            v
    +----------------+
    |   Symmetric    |
    |   Encryption   |
    +----------------+
            | (CiphertextPacket)
            v
    +----------------+
    |      Exit      |
    +----------------+
            |
    (FPGAOutputPacket)
            |
            v
       Host Memory
```

### SYCL Pipe Connections

| Pipe Name | Source Kernel | Destination Kernel | Packet Type |
|-----------|---------------|--------------------|-------------|
| `EntranceToDWTPipe` | Entrance | DWT Inverse | `DWTPacket` |
| `DWTToScaleReducePipe` | DWT Inverse | Scale & Reduce | `ScaleReducePacket` |
| `ScaleReduceToNTTPipe` | Scale & Reduce | NTT Forward | `NTTPacket` |
| `NTTToEncryptPipe` | NTT Forward | Symmetric Encrypt | `EncryptPacket` |
| `EncryptToExitPipe` | Symmetric Encrypt | Exit | `CiphertextPacket` |

## Pipeline Stages

### 1. Entrance Kernel
The Entrance kernel acts as the gateway between the host and the FPGA pipeline. It reads the `FPGAInputPacket` from host memory and populates the first internal packet (`DWTPacket`). It also performs initial calculations such as the DWT scale factor ($scale / N$).

### 2. DWT Inverse Kernel
Performs the inverse DWT required for CKKS encoding. It transforms complex-valued slots into polynomial coefficients in the time domain. This kernel uses a Cooley-Tukey-like butterfly structure but operating on complex numbers with pre-computed roots of unity.

### 3. Scale & Reduce Kernel
Converts the complex-valued results from DWT into integer coefficients modulo $q$.
-   **Rounding**: Rounds real parts of complex numbers to the nearest integer.
-   **Reduction**: Performs modular reduction of the large integer values to fit within the ciphertext modulus $q$.

### 4. NTT Forward Kernel
Performs the forward NTT on the plaintext coefficients. This moves the plaintext from the time domain to the power-of-x domain (coefficient representation) to the Evaluation domain (point-value representation), which is required for efficient encryption.

### 5. Symmetric Encryption Kernel
Produces the two components of a CKKS symmetric ciphertext ($c_0, c_1$):
-   Generates $c_1$ as a uniform random polynomial $a$ (provided in the packet).
-   Computes an NTT of the error samples $e$.
-   Computes $c_0 = -(a \cdot s + e) + m \pmod q$, where $s$ is the secret key and $m$ is the plaintext.
All calculations are performed in the NTT domain.

### 6. Exit Kernel
The Exit kernel collects the final ciphertext components ($c_0, c_1$) from the `CiphertextPacket` and writes them back to the host-provided `FPGAOutputPacket` buffer.

## Packet Structures

Packets are designed to carry both the active data for the current stage and "pass-through" data required by downstream kernels.

### Common Packet Fields
-   `n`: Polynomial modulus degree (e.g., 8192, 16384, 32768).
-   `log_n`: Logarithm of $n$.
-   `modulus`: The ciphertext modulus $q$.

### Pass-through Mechanism
Each kernel is responsible for forwarding data it doesn't use but that subsequent kernels need. For example, `DWTPacket` carries `secret_key_ntt` and `error_samples` even though the DWT kernel only operates on `values`.

```text
FPGAInputPacket
  |
  +-- values -----------> Used by DWT
  +-- dwt_inv_roots ----> Used by DWT
  +-- scale ------------> Used by DWT (as scale/n)
  +-- ntt_roots --------> Pass-through to NTT/Encrypt
  +-- secret_key_ntt ----> Pass-through to Encrypt
  +-- uniform_poly_ntt --> Pass-through to Encrypt
  +-- error_samples -----> Pass-through to Encrypt
```

## Individual Kernel Descriptions

### DWT Inverse
Mathematically, this performs:
$$f = \text{IDWT}(v) \cdot \frac{scale}{N}$$
where $v$ is the vector of complex numbers. The implementation uses an in-place butterfly network.

### NTT Forward
Mathematically, this performs:
$$\hat{f} = \text{NTT}(f, \text{roots}, q)$$
It uses a standard Radix-2 NTT algorithm optimized for FPGA memory access patterns.

### Symmetric Encrypt
Encryption in the NTT domain:
$$c_1 = \hat{a}$$
$$c_0 = \hat{m} + \hat{e} - \hat{a} \cdot \hat{s} \pmod q$$
The kernel handles the conversion of error samples to the NTT domain before performing the component-wise modular arithmetic.

## Host vs FPGA Execution Paths

The `FPGAPipeline` class provides two execution paths:
1.  **Host Path (`execute_host_pipeline`)**: A reference implementation that runs on the CPU. It is used for verification and when FPGA hardware is not available.
2.  **FPGA Path (`execute_fpga_pipeline`)**: Submits the six SYCL kernels to the FPGA device queue.

The class abstracts these paths, providing a unified `encrypt` interface to the rest of the SEAL library.

## Build Configuration

FPGA support is enabled via CMake:
```bash
cmake -DSEAL_USE_FPGA=ON .
```
When `SEAL_USE_FPGA` is defined:
-   SYCL headers are included.
-   FPGA-specific kernels and pipe definitions are compiled.
-   `FPGAPipeline` will attempt to use the FPGA hardware if a compatible SYCL device is found.

## Design Rationale

### Modularity
By splitting the pipeline into distinct kernels connected by pipes, we achieve:
-   **RTL Replacement**: Performance-critical blocks like NTT and DWT can be replaced with specialized RTL without changing the control logic.
-   **Resource Scaling**: Kernels can be individually tuned for resource usage (ALMs, DSPs, Memory) to fit different FPGA sizes.
-   **Pipelined Execution**: Multiple packets can theoretically be in the pipeline at different stages simultaneously (Task Parallelism).

### Memory Locality
Each kernel uses local buffers (`MAX_POLY_DEGREE`) to store polynomial coefficients during computation. This minimizes expensive global memory (DDR/HBM) accesses, keeping the high-bandwidth transformations entirely on-chip.
