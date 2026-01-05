// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

// =============================================================================
// IFFT CORE - RTL REPLACEMENT INTERFACE
// =============================================================================
//
// This header defines the interface for the IFFT core computation.
// The SYCL implementation can be replaced with RTL by:
//
// 1. Creating an RTL module (Verilog/VHDL) with Avalon-ST interfaces
// 2. Registering the RTL with Intel's SYCL RTL integration
// 3. Replacing the function body with RTL invocation
//
// RTL Interface Requirements:
// ---------------------------
// Input Stream:  N complex<double> values (128 bits each: 64-bit real + 64-bit imag)
// Input Stream:  N complex<double> twiddle factors (same format)
// Output Stream: N complex<double> transformed values
// Parameters:    N (polynomial degree), log_N
//
// Algorithm: Radix-2 DIF (Decimation-in-Frequency) IFFT
// Butterfly: out[j] = in[j] + in[j+gap]
//            out[j+gap] = (in[j] - in[j+gap]) * twiddle
//
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>

namespace seal
{
    namespace fpga
    {
        // =====================================================================
        // RTL REPLACEMENT POINT - BEGIN
        // =====================================================================
        //
        // To replace with RTL:
        // 1. Keep this function signature
        // 2. Replace implementation with:
        //    - SYCL RTL library call, OR
        //    - sycl::ext::intel::experimental::task_sequence with RTL, OR
        //    - Platform Designer component invocation
        //
        // The function performs in-place Radix-2 DIF IFFT.
        // Input values and roots are in bit-reversed order.
        // Output is in natural order.
        //
        // =====================================================================

        inline void ifft_dif_core(
            std::complex<double>* values,
            const std::complex<double>* roots,
            std::size_t n,
            std::size_t root_offset = 0)
        {
            std::complex<double> r, u, v;
            std::size_t gap = 1;
            std::size_t m = n >> 1;
            std::size_t root_idx = root_offset;

            while (m >= 1)
            {
                std::size_t offset = 0;
                for (std::size_t i = 0; i < m; i++)
                {
                    root_idx++;
                    r = roots[root_idx];
                    for (std::size_t j = 0; j < gap; j++)
                    {
                        std::size_t x_idx = offset + j;
                        std::size_t y_idx = x_idx + gap;
                        u = values[x_idx];
                        v = values[y_idx];
                        values[x_idx] = u + v;
                        values[y_idx] = (u - v) * r;
                    }
                    offset += gap << 1;
                }
                gap <<= 1;
                m >>= 1;
            }
        }

        // =====================================================================
        // RTL REPLACEMENT POINT - END
        // =====================================================================

    } // namespace fpga
} // namespace seal
