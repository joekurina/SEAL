// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_ntt.h"
#include "../Inc/fpga_arith.h"
#include <cmath>
#include <cstring>

namespace seal
{
    namespace fpga
    {
        namespace
        {
            inline std::uint64_t multiply_uint64_hw64(std::uint64_t a, std::uint64_t b)
            {
                __uint128_t product = static_cast<__uint128_t>(a) * b;
                return static_cast<std::uint64_t>(product >> 64);
            }

            inline std::uint64_t barrett_reduce_64_internal(
                std::uint64_t x, std::uint64_t modulus, const std::uint64_t* ratio)
            {
                std::uint64_t tmp = multiply_uint64_hw64(x, ratio[1]);
                tmp = x - tmp * modulus;
                return tmp >= modulus ? tmp - modulus : tmp;
            }

            inline std::uint64_t multiply_mod_lazy(
                std::uint64_t a, std::uint64_t b, std::uint64_t modulus)
            {
                __uint128_t product = static_cast<__uint128_t>(a) * b;
                return static_cast<std::uint64_t>(product % modulus);
            }

            inline std::uint64_t add_mod(std::uint64_t a, std::uint64_t b, std::uint64_t modulus)
            {
                std::uint64_t sum = a + b;
                return sum >= modulus ? sum - modulus : sum;
            }

            inline std::uint64_t sub_mod(std::uint64_t a, std::uint64_t b, std::uint64_t modulus)
            {
                return a >= b ? a - b : a + modulus - b;
            }
        }

        void scale_and_reduce_host(
            const std::complex<double>* src,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus,
            const std::uint64_t* barrett_ratio)
        {
            for (std::size_t i = 0; i < n; i++)
            {
                double coeff_d = std::round(src[i].real());
                bool is_negative = std::signbit(coeff_d);
                coeff_d = std::fabs(coeff_d);

                std::uint64_t coeff_u;
                if (coeff_d < static_cast<double>(UINT64_MAX))
                {
                    coeff_u = static_cast<std::uint64_t>(coeff_d);
                }
                else
                {
                    coeff_u = static_cast<std::uint64_t>(std::fmod(coeff_d, static_cast<double>(modulus)));
                }

                coeff_u = barrett_reduce_64_internal(coeff_u, modulus, barrett_ratio);

                if (is_negative && coeff_u != 0)
                {
                    coeff_u = modulus - coeff_u;
                }

                dest[i] = coeff_u;
            }
        }

        void ntt_forward_host(
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* root_powers,
            std::uint64_t modulus)
        {
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
                    std::uint64_t w = root_powers[root_idx];

                    for (std::size_t j = 0; j < gap; j++)
                    {
                        std::size_t x_idx = offset + j;
                        std::size_t y_idx = x_idx + gap;

                        std::uint64_t u = values[x_idx];
                        if (u >= two_times_modulus)
                        {
                            u -= two_times_modulus;
                        }

                        std::uint64_t v = multiply_mod_lazy(values[y_idx], w, modulus);

                        values[x_idx] = u + v;
                        values[y_idx] = u + two_times_modulus - v;
                    }
                    offset += gap << 1;
                }
                gap >>= 1;
                m <<= 1;
            }

            for (std::size_t i = 0; i < n; i++)
            {
                if (values[i] >= two_times_modulus)
                {
                    values[i] -= two_times_modulus;
                }
                if (values[i] >= modulus)
                {
                    values[i] -= modulus;
                }
            }
        }

        void ntt_inverse_host(
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* inv_root_powers,
            std::uint64_t modulus,
            std::uint64_t inv_n)
        {
            std::uint64_t two_times_modulus = modulus << 1;

            std::size_t gap = 1;
            std::size_t m = n >> 1;
            std::size_t root_idx = 0;

            while (m > 1)
            {
                std::size_t offset = 0;
                for (std::size_t i = 0; i < m; i++)
                {
                    root_idx++;
                    std::uint64_t w = inv_root_powers[root_idx];

                    for (std::size_t j = 0; j < gap; j++)
                    {
                        std::size_t x_idx = offset + j;
                        std::size_t y_idx = x_idx + gap;

                        std::uint64_t u = values[x_idx];
                        std::uint64_t v = values[y_idx];

                        std::uint64_t sum = u + v;
                        if (sum >= two_times_modulus)
                        {
                            sum -= two_times_modulus;
                        }
                        values[x_idx] = sum;

                        std::uint64_t diff = u + two_times_modulus - v;
                        values[y_idx] = multiply_mod_lazy(diff, w, modulus);
                    }
                    offset += gap << 1;
                }
                gap <<= 1;
                m >>= 1;
            }

            root_idx++;
            std::uint64_t final_w = inv_root_powers[root_idx];
            std::uint64_t scaled_w = multiply_mod_lazy(final_w, inv_n, modulus);

            for (std::size_t j = 0; j < gap; j++)
            {
                std::size_t x_idx = j;
                std::size_t y_idx = j + gap;

                std::uint64_t u = values[x_idx];
                std::uint64_t v = values[y_idx];

                std::uint64_t sum = u + v;
                if (sum >= two_times_modulus)
                {
                    sum -= two_times_modulus;
                }
                values[x_idx] = multiply_mod_lazy(sum, inv_n, modulus);

                std::uint64_t diff = u + two_times_modulus - v;
                values[y_idx] = multiply_mod_lazy(diff, scaled_w, modulus);
            }

            for (std::size_t i = 0; i < n; i++)
            {
                if (values[i] >= modulus)
                {
                    values[i] -= modulus;
                }
            }
        }

#ifdef SEAL_USE_FPGA

        sycl::event scale_and_reduce(
            sycl::queue& q,
            const std::complex<double>* src,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus,
            const std::uint64_t* barrett_ratio)
        {
            std::uint64_t ratio0 = barrett_ratio[0];
            std::uint64_t ratio1 = barrett_ratio[1];

            return q.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    for (std::size_t i = 0; i < n; i++)
                    {
                        double coeff_d = src[i].real();
                        coeff_d = coeff_d >= 0 ? coeff_d + 0.5 : coeff_d - 0.5;
                        coeff_d = coeff_d >= 0 ? static_cast<std::int64_t>(coeff_d) 
                                               : static_cast<std::int64_t>(coeff_d);
                        
                        bool is_negative = coeff_d < 0;
                        if (is_negative) coeff_d = -coeff_d;

                        std::uint64_t coeff_u = static_cast<std::uint64_t>(coeff_d);
                        
                        arith::uint128_t product = arith::mul_u64(coeff_u, ratio1);
                        std::uint64_t tmp = product.hi;
                        tmp = coeff_u - tmp * modulus;
                        if (tmp >= modulus) tmp -= modulus;
                        coeff_u = tmp;

                        if (is_negative && coeff_u != 0)
                        {
                            coeff_u = modulus - coeff_u;
                        }

                        dest[i] = coeff_u;
                    }
                });
            });
        }

        sycl::event ntt_forward(
            sycl::queue& q,
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* root_powers,
            std::uint64_t modulus)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    std::uint64_t local_values[32768];
                    std::uint64_t local_roots[32768];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] = values[i];
                        local_roots[i] = root_powers[i];
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

                                std::uint64_t u = local_values[x_idx];
                                if (u >= two_times_modulus)
                                {
                                    u -= two_times_modulus;
                                }

                                std::uint64_t v = arith::mul_mod_fpga(local_values[y_idx], w, modulus);

                                local_values[x_idx] = u + v;
                                local_values[y_idx] = u + two_times_modulus - v;
                            }
                            offset += gap << 1;
                        }
                        gap >>= 1;
                        m <<= 1;
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        std::uint64_t val = local_values[i];
                        if (val >= two_times_modulus) val -= two_times_modulus;
                        if (val >= modulus) val -= modulus;
                        values[i] = val;
                    }
                });
            });
        }

        sycl::event ntt_inverse(
            sycl::queue& q,
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* inv_root_powers,
            std::uint64_t modulus,
            std::uint64_t inv_n)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    std::uint64_t local_values[32768];
                    std::uint64_t local_roots[32768];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] = values[i];
                        local_roots[i] = inv_root_powers[i];
                    }

                    std::uint64_t two_times_modulus = modulus << 1;

                    std::size_t gap = 1;
                    std::size_t m = n >> 1;
                    std::size_t root_idx = 0;

                    while (m > 1)
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

                                std::uint64_t u = local_values[x_idx];
                                std::uint64_t v = local_values[y_idx];

                                std::uint64_t sum = u + v;
                                if (sum >= two_times_modulus) sum -= two_times_modulus;
                                local_values[x_idx] = sum;

                                std::uint64_t diff = u + two_times_modulus - v;
                                local_values[y_idx] = arith::mul_mod_fpga(diff, w, modulus);
                            }
                            offset += gap << 1;
                        }
                        gap <<= 1;
                        m >>= 1;
                    }

                    root_idx++;
                    std::uint64_t final_w = local_roots[root_idx];
                    std::uint64_t scaled_w = arith::mul_mod_fpga(final_w, inv_n, modulus);

                    for (std::size_t j = 0; j < gap; j++)
                    {
                        std::size_t x_idx = j;
                        std::size_t y_idx = j + gap;

                        std::uint64_t u = local_values[x_idx];
                        std::uint64_t v = local_values[y_idx];

                        std::uint64_t sum = u + v;
                        if (sum >= two_times_modulus) sum -= two_times_modulus;
                        local_values[x_idx] = arith::mul_mod_fpga(sum, inv_n, modulus);

                        std::uint64_t diff = u + two_times_modulus - v;
                        local_values[y_idx] = arith::mul_mod_fpga(diff, scaled_w, modulus);
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        std::uint64_t val = local_values[i];
                        if (val >= modulus) val -= modulus;
                        values[i] = val;
                    }
                });
            });
        }

#endif

    } // namespace fpga
} // namespace seal
