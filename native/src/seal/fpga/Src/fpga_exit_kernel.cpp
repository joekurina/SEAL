// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_exit_kernel.h"

#ifdef SEAL_USE_FPGA

namespace seal
{
    namespace fpga
    {
        sycl::event submit_exit_kernel(
            sycl::queue& q,
            std::size_t* n_out,
            std::uint64_t* c0_out,
            std::uint64_t* c1_out)
        {
            // Capture USM pointers by value - this is valid for SYCL device kernels
            return q.submit([=](sycl::handler& h) {
                h.single_task<ExitKernel>([=]() {
                    CiphertextPacket pkt = pipes::EncryptToExitPipe::read();

                    *n_out = pkt.n;
                    for (std::size_t i = 0; i < pkt.n; i++)
                    {
                        c0_out[i] = pkt.c0[i];
                        c1_out[i] = pkt.c1[i];
                    }
                });
            });
        }
    }
}

#endif
