// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_encrypt.h"
#include "../Inc/fpga_ntt.h"
#include "../Inc/fpga_arith.h"
#include <cstring>

namespace seal
{
    namespace fpga
    {
        namespace
        {
            inline std::uint64_t multiply_mod(std::uint64_t a, std::uint64_t b, std::uint64_t modulus)
            {
                return arith::mul_mod_fpga(a, b, modulus);
            }
        }

        void add_poly_mod_host(
            const std::uint64_t* a,
            const std::uint64_t* b,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus)
        {
            for (std::size_t i = 0; i < n; i++)
            {
                std::uint64_t sum = a[i] + b[i];
                dest[i] = sum >= modulus ? sum - modulus : sum;
            }
        }

        void negate_poly_mod_host(
            const std::uint64_t* src,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus)
        {
            for (std::size_t i = 0; i < n; i++)
            {
                dest[i] = src[i] == 0 ? 0 : modulus - src[i];
            }
        }

        void dyadic_product_mod_host(
            const std::uint64_t* a,
            const std::uint64_t* b,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus)
        {
            for (std::size_t i = 0; i < n; i++)
            {
                dest[i] = multiply_mod(a[i], b[i], modulus);
            }
        }

        void error_to_ntt_host(
            const std::int64_t* error_samples,
            std::uint64_t* dest,
            std::size_t n,
            int log_n,
            std::uint64_t modulus,
            const std::uint64_t* ntt_root_powers)
        {
            for (std::size_t i = 0; i < n; i++)
            {
                std::int64_t e = error_samples[i];
                if (e >= 0)
                {
                    dest[i] = static_cast<std::uint64_t>(e);
                }
                else
                {
                    dest[i] = modulus - static_cast<std::uint64_t>(-e);
                }
            }

            ntt_forward_host(dest, n, log_n, ntt_root_powers, modulus);
        }

        void encrypt_symmetric_host(
            const std::uint64_t* plaintext_ntt,
            const std::uint64_t* secret_key_ntt,
            const std::uint64_t* uniform_poly_ntt,
            const std::int64_t* error_samples,
            std::uint64_t* c0_out,
            std::uint64_t* c1_out,
            std::size_t n,
            int log_n,
            std::uint64_t modulus,
            const std::uint64_t* ntt_root_powers)
        {
            std::vector<std::uint64_t> error_ntt(n);
            error_to_ntt_host(error_samples, error_ntt.data(), n, log_n, modulus, ntt_root_powers);

            std::vector<std::uint64_t> as(n);
            dyadic_product_mod_host(uniform_poly_ntt, secret_key_ntt, as.data(), n, modulus);

            std::vector<std::uint64_t> neg_as(n);
            negate_poly_mod_host(as.data(), neg_as.data(), n, modulus);

            std::vector<std::uint64_t> neg_as_plus_e(n);
            add_poly_mod_host(neg_as.data(), error_ntt.data(), neg_as_plus_e.data(), n, modulus);

            add_poly_mod_host(neg_as_plus_e.data(), plaintext_ntt, c0_out, n, modulus);

            std::memcpy(c1_out, uniform_poly_ntt, n * sizeof(std::uint64_t));
        }

#ifdef SEAL_USE_FPGA

        sycl::event encrypt_symmetric(
            sycl::queue& q,
            const std::uint64_t* plaintext_ntt,
            const std::uint64_t* secret_key_ntt,
            const std::uint64_t* uniform_poly_ntt,
            const std::int64_t* error_samples,
            std::uint64_t* c0_out,
            std::uint64_t* c1_out,
            std::size_t n,
            int log_n,
            std::uint64_t modulus,
            const std::uint64_t* ntt_root_powers)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    std::uint64_t local_error[32768];
                    std::uint64_t local_roots[32768];
                    std::uint64_t local_as[32768];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        std::int64_t e = error_samples[i];
                        if (e >= 0)
                        {
                            local_error[i] = static_cast<std::uint64_t>(e);
                        }
                        else
                        {
                            local_error[i] = modulus - static_cast<std::uint64_t>(-e);
                        }
                        local_roots[i] = ntt_root_powers[i];
                    }

                    std::uint64_t two_times_modulus = modulus << 1;
                    std::size_t gap = n >> 1;
                    std::size_t m = 1;
                    std::size_t root_idx = 0;

                    while (m < n)
                    {
                        std::size_t offset = 0;
                        for (std::size_t i = 0; i < m; i++)
                        {
                            root_idx++;
                            std::uint64_t w = local_roots[root_idx];

                            for (std::size_t j = 0; j < gap; j++)
                            {
                                std::size_t x_idx = offset + j;
                                std::size_t y_idx = x_idx + gap;

                                std::uint64_t u = local_error[x_idx];
                                if (u >= two_times_modulus) u -= two_times_modulus;

                                std::uint64_t v = arith::mul_mod_fpga(local_error[y_idx], w, modulus);

                                local_error[x_idx] = u + v;
                                local_error[y_idx] = u + two_times_modulus - v;
                            }
                            offset += gap << 1;
                        }
                        gap >>= 1;
                        m <<= 1;
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        std::uint64_t val = local_error[i];
                        if (val >= two_times_modulus) val -= two_times_modulus;
                        if (val >= modulus) val -= modulus;
                        local_error[i] = val;
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_as[i] = arith::mul_mod_fpga(uniform_poly_ntt[i], secret_key_ntt[i], modulus);
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        std::uint64_t neg_as = (local_as[i] == 0) ? 0 : modulus - local_as[i];

                        std::uint64_t neg_as_plus_e = neg_as + local_error[i];
                        if (neg_as_plus_e >= modulus) neg_as_plus_e -= modulus;

                        std::uint64_t c0_val = neg_as_plus_e + plaintext_ntt[i];
                        if (c0_val >= modulus) c0_val -= modulus;

                        c0_out[i] = c0_val;
                        c1_out[i] = uniform_poly_ntt[i];
                    }
                });
            });
        }

#endif

    } // namespace fpga
} // namespace seal
