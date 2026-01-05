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
            FPGAOutputPacket& output)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task<ExitKernel>([&output]() {
                    CiphertextPacket pkt = pipes::EncryptToExitPipe::read();

                    output.poly_modulus_degree = pkt.n;
                    for (std::size_t i = 0; i < pkt.n; i++)
                    {
                        output.c0[i] = pkt.c0[i];
                        output.c1[i] = pkt.c1[i];
                    }
                });
            });
        }
    }
}

#endif
