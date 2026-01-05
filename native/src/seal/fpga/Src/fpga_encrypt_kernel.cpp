// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_encrypt_kernel.h"

#ifdef SEAL_USE_FPGA

namespace seal
{
    namespace fpga
    {
        namespace
        {
            inline std::uint64_t mod_reduce(std::uint64_t value, std::uint64_t modulus)
            {
                return value >= modulus ? value - modulus : value;
            }

            inline std::uint64_t mul_mod(std::uint64_t a, std::uint64_t b, std::uint64_t modulus)
            {
                __uint128_t product = static_cast<__uint128_t>(a) * b;
                return static_cast<std::uint64_t>(product % modulus);
            }
        }

        sycl::event submit_encrypt_kernel(sycl::queue& q)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task<EncryptKernel>([=]() {
                    EncryptPacket pkt = pipes::NTTToEncryptPipe::read();

                    std::size_t n = pkt.n;
                    int log_n = pkt.log_n;
                    std::uint64_t modulus = pkt.modulus;

                    std::uint64_t c0[MAX_POLY_DEGREE];
                    std::uint64_t c1[MAX_POLY_DEGREE];
                    std::uint64_t error_ntt[MAX_POLY_DEGREE];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        std::int64_t e = pkt.error_samples[i];
                        std::uint64_t e_mod = (e >= 0) ? static_cast<std::uint64_t>(e)
                                                       : modulus - static_cast<std::uint64_t>(-e);
                        error_ntt[i] = e_mod;
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
                            std::uint64_t w = pkt.ntt_roots[root_idx];
                            std::size_t j2 = j1 + t;

                            for (std::size_t j = j1; j < j2; j++)
                            {
                                std::uint64_t u = error_ntt[j];
                                std::uint64_t v = mul_mod(error_ntt[j + t], w, modulus);
                                error_ntt[j] = mod_reduce(u + v, modulus);
                                error_ntt[j + t] = mod_reduce(u + modulus - v, modulus);
                            }
                            j1 += (t << 1);
                        }
                        t >>= 1;
                        m <<= 1;
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        std::uint64_t as = mul_mod(pkt.uniform_poly_ntt[i], pkt.secret_key_ntt[i], modulus);
                        std::uint64_t neg_as = (as == 0) ? 0 : modulus - as;
                        std::uint64_t m_plus_e = mod_reduce(pkt.plaintext_ntt[i] + error_ntt[i], modulus);
                        c0[i] = mod_reduce(neg_as + m_plus_e, modulus);
                        c1[i] = pkt.uniform_poly_ntt[i];
                    }

                    CiphertextPacket out_pkt;
                    out_pkt.n = n;
                    for (std::size_t i = 0; i < n; i++)
                    {
                        out_pkt.c0[i] = c0[i];
                        out_pkt.c1[i] = c1[i];
                    }

                    pipes::EncryptToExitPipe::write(out_pkt);
                });
            });
        }
    }
}

#endif
