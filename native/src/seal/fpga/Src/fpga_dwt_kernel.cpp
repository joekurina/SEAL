// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_dwt_kernel.h"

#ifdef SEAL_USE_FPGA

namespace seal
{
    namespace fpga
    {
        sycl::event submit_dwt_inverse_kernel(sycl::queue& q)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task<DWTInverseKernel>([=]() {
                    DWTPacket pkt = pipes::EntranceToDWTPipe::read();

                    std::size_t n = pkt.n;
                    int log_n = pkt.log_n;
                    double scale_factor = pkt.scale_factor;

                    std::complex<double> local_values[MAX_POLY_DEGREE];
                    std::complex<double> local_roots[MAX_POLY_DEGREE];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] = pkt.values[i];
                        local_roots[i] = pkt.inv_roots[i];
                    }

                    std::complex<double> r, u, v;
                    std::size_t gap = 1;
                    std::size_t m = n >> 1;
                    std::size_t root_idx = 0;

                    while (m > 1)
                    {
                        std::size_t offset = 0;
                        for (std::size_t i = 0; i < m; i++)
                        {
                            root_idx++;
                            r = local_roots[root_idx];
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                std::size_t x_idx = offset + j;
                                std::size_t y_idx = x_idx + gap;
                                u = local_values[x_idx];
                                v = local_values[y_idx];
                                local_values[x_idx] = u + v;
                                local_values[y_idx] = (u - v) * r;
                            }
                            offset += gap << 1;
                        }
                        gap <<= 1;
                        m >>= 1;
                    }

                    root_idx++;
                    r = local_roots[root_idx];
                    std::complex<double> scaled_r = r * scale_factor;

                    for (std::size_t j = 0; j < gap; j++)
                    {
                        std::size_t x_idx = j;
                        std::size_t y_idx = j + gap;
                        u = local_values[x_idx];
                        v = local_values[y_idx];
                        local_values[x_idx] = (u + v) * scale_factor;
                        local_values[y_idx] = (u - v) * scaled_r;
                    }

                    ScaleReducePacket out_pkt;
                    out_pkt.n = n;
                    out_pkt.log_n = log_n;
                    out_pkt.modulus = pkt.modulus;
                    out_pkt.barrett_ratio[0] = pkt.barrett_ratio[0];
                    out_pkt.barrett_ratio[1] = pkt.barrett_ratio[1];
                    for (std::size_t i = 0; i < n; i++)
                    {
                        out_pkt.values[i] = local_values[i];
                        out_pkt.ntt_roots[i] = pkt.ntt_roots[i];
                        out_pkt.secret_key_ntt[i] = pkt.secret_key_ntt[i];
                        out_pkt.uniform_poly_ntt[i] = pkt.uniform_poly_ntt[i];
                        out_pkt.error_samples[i] = pkt.error_samples[i];
                    }

                    pipes::DWTToScaleReducePipe::write(out_pkt);
                });
            });
        }
    }
}

#endif
