// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_entrance_kernel.h"

#ifdef SEAL_USE_FPGA

namespace seal
{
    namespace fpga
    {
        sycl::event submit_entrance_kernel(
            sycl::queue& q,
            const FPGAInputPacket& input)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task<EntranceKernel>([=]() {
                    DWTPacket dwt_pkt;
                    dwt_pkt.n = input.poly_modulus_degree;
                    dwt_pkt.log_n = input.log_poly_modulus_degree;
                    dwt_pkt.scale_factor = input.scale / static_cast<double>(input.poly_modulus_degree);
                    dwt_pkt.modulus = input.modulus;
                    dwt_pkt.barrett_ratio[0] = input.barrett_ratio[0];
                    dwt_pkt.barrett_ratio[1] = input.barrett_ratio[1];

                    for (std::size_t i = 0; i < input.poly_modulus_degree; i++)
                    {
                        dwt_pkt.values[i] = input.prepared_values[i];
                        dwt_pkt.inv_roots[i] = input.dwt_inv_roots[i];
                        dwt_pkt.ntt_roots[i] = input.ntt_roots[i];
                        dwt_pkt.secret_key_ntt[i] = input.secret_key_ntt[i];
                        dwt_pkt.uniform_poly_ntt[i] = input.uniform_poly_ntt[i];
                        dwt_pkt.error_samples[i] = input.error_samples[i];
                    }

                    pipes::EntranceToDWTPipe::write(dwt_pkt);
                });
            });
        }
    }
}

#endif
