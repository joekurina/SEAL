// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_ntt_kernel.h"
#include "../Inc/fpga_arith.h"

#ifdef SEAL_USE_FPGA

namespace seal
{
    namespace fpga
    {
        namespace
        {
            using arith::mod_reduce;
            using arith::mul_mod_fpga;
        }

        sycl::event submit_scale_reduce_kernel(sycl::queue& q)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task<ScaleReduceKernel>([=]() {
                    ScaleReducePacket pkt = pipes::DWTToScaleReducePipe::read();

                    std::size_t n = pkt.n;
                    std::uint64_t modulus = pkt.modulus;

                    NTTPacket out_pkt;
                    out_pkt.n = n;
                    out_pkt.log_n = pkt.log_n;
                    out_pkt.modulus = modulus;

                    for (std::size_t i = 0; i < n; i++)
                    {
                        out_pkt.roots[i] = pkt.ntt_roots[i];
                        out_pkt.secret_key_ntt[i] = pkt.secret_key_ntt[i];
                        out_pkt.uniform_poly_ntt[i] = pkt.uniform_poly_ntt[i];
                        out_pkt.error_samples[i] = pkt.error_samples[i];
                        double val = pkt.values[i].real();
                        double rounded = (val >= 0) ? (val + 0.5) : (val - 0.5);
                        rounded = (rounded >= 0) ? static_cast<double>(static_cast<std::int64_t>(rounded))
                                                 : static_cast<double>(static_cast<std::int64_t>(rounded));

                        std::uint64_t result;
                        if (rounded >= 0)
                        {
                            double mod_val = rounded;
                            while (mod_val >= static_cast<double>(modulus))
                            {
                                mod_val -= static_cast<double>(modulus);
                            }
                            result = static_cast<std::uint64_t>(mod_val);
                        }
                        else
                        {
                            double abs_val = -rounded;
                            while (abs_val >= static_cast<double>(modulus))
                            {
                                abs_val -= static_cast<double>(modulus);
                            }
                            std::uint64_t abs_mod = static_cast<std::uint64_t>(abs_val);
                            result = (abs_mod == 0) ? 0 : modulus - abs_mod;
                        }
                        out_pkt.coeffs[i] = result;
                    }

                    pipes::ScaleReduceToNTTPipe::write(out_pkt);
                });
            });
        }

        sycl::event submit_ntt_forward_kernel(sycl::queue& q)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task<NTTForwardKernel>([=]() {
                    NTTPacket pkt = pipes::ScaleReduceToNTTPipe::read();

                    std::size_t n = pkt.n;
                    int log_n = pkt.log_n;
                    std::uint64_t modulus = pkt.modulus;

                    std::uint64_t local_values[MAX_POLY_DEGREE];
                    std::uint64_t local_roots[MAX_POLY_DEGREE];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] = pkt.coeffs[i];
                        local_roots[i] = pkt.roots[i];
                    }

                    std::size_t t = n >> 1;
                    std::size_t m = 1;
                    std::size_t root_idx = 0;

                    while (m < n)
                    {
                        std::size_t j1 = 0;
                        for (std::size_t i = 0; i < m; i++)
                        {
                            root_idx++;
                            std::uint64_t w = local_roots[root_idx];
                            std::size_t j2 = j1 + t;

                            for (std::size_t j = j1; j < j2; j++)
                            {
                                std::uint64_t u = local_values[j];
                                std::uint64_t v = mul_mod_fpga(local_values[j + t], w, modulus);

                                local_values[j] = mod_reduce(u + v, modulus);
                                local_values[j + t] = mod_reduce(u + modulus - v, modulus);
                            }
                            j1 += (t << 1);
                        }
                        t >>= 1;
                        m <<= 1;
                    }

                    EncryptPacket out_pkt;
                    out_pkt.n = n;
                    out_pkt.log_n = log_n;
                    out_pkt.modulus = modulus;
                    for (std::size_t i = 0; i < n; i++)
                    {
                        out_pkt.plaintext_ntt[i] = local_values[i];
                        out_pkt.secret_key_ntt[i] = pkt.secret_key_ntt[i];
                        out_pkt.uniform_poly_ntt[i] = pkt.uniform_poly_ntt[i];
                        out_pkt.error_samples[i] = pkt.error_samples[i];
                        out_pkt.ntt_roots[i] = pkt.roots[i];
                    }

                    pipes::NTTToEncryptPipe::write(out_pkt);
                });
            });
        }
    }
}

#endif
