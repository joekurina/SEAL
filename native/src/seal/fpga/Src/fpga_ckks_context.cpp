// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_ckks_context.h"
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace seal
{
    namespace fpga
    {
        void FPGACKKSParams::validate() const
        {
            if (poly_modulus_degree != 4096 && poly_modulus_degree != 8192 &&
                poly_modulus_degree != 16384 && poly_modulus_degree != 32768)
            {
                throw std::invalid_argument("poly_modulus_degree must be 4096, 8192, 16384, or 32768");
            }

            std::size_t expected_log = 0;
            std::size_t n = poly_modulus_degree;
            while (n > 1)
            {
                n >>= 1;
                expected_log++;
            }
            if (log_poly_modulus_degree != expected_log)
            {
                throw std::invalid_argument("log_poly_modulus_degree mismatch");
            }

            if (modulus < 2)
            {
                throw std::invalid_argument("modulus must be at least 2");
            }

            if (scale <= 0)
            {
                throw std::invalid_argument("scale must be positive");
            }
        }

        void FPGACKKSParams::compute_derived_constants()
        {
            compute_barrett_ratio(modulus, barrett_ratio);
            two_times_modulus = modulus << 1;

            std::uint64_t n = poly_modulus_degree;
            std::uint64_t mod = modulus;
            std::uint64_t phi_n = mod - 1;
            std::uint64_t exp = phi_n - 1;

            std::uint64_t result = 1;
            std::uint64_t base = n % mod;

            while (exp > 0)
            {
                if (exp & 1)
                {
                    __uint128_t temp = static_cast<__uint128_t>(result) * base;
                    result = temp % mod;
                }
                __uint128_t temp = static_cast<__uint128_t>(base) * base;
                base = temp % mod;
                exp >>= 1;
            }
            inv_n_mod_q = result;
        }

        void compute_barrett_ratio(std::uint64_t modulus, std::uint64_t* ratio)
        {
            __uint128_t numerator = static_cast<__uint128_t>(1) << 127;
            numerator <<= 1;
            __uint128_t quotient = numerator / modulus;

            ratio[0] = static_cast<std::uint64_t>(quotient);
            ratio[1] = static_cast<std::uint64_t>(quotient >> 64);
        }

        std::uint64_t barrett_reduce_128(const std::uint64_t* x, std::uint64_t modulus, const std::uint64_t* ratio)
        {
            __uint128_t x128 = (static_cast<__uint128_t>(x[1]) << 64) | x[0];
            __uint128_t ratio128 = (static_cast<__uint128_t>(ratio[1]) << 64) | ratio[0];

            __uint128_t q = (x128 * ratio128) >> 128;
            __uint128_t r = x128 - q * modulus;

            while (r >= modulus)
            {
                r -= modulus;
            }
            return static_cast<std::uint64_t>(r);
        }

        std::uint64_t barrett_reduce_64(std::uint64_t x, std::uint64_t modulus, const std::uint64_t* ratio)
        {
            std::uint64_t arr[2] = {x, 0};
            return barrett_reduce_128(arr, modulus, ratio);
        }

#ifdef SEAL_USE_FPGA

        FPGACKKSContext::FPGACKKSContext(const FPGACKKSParams& params, bool use_emulator)
            : params_(params)
        {
            params_.validate();
            params_.compute_derived_constants();
            init_queue(use_emulator);
            compute_index_map();
            compute_dwt_roots();
            compute_ntt_roots();
        }

        FPGACKKSContext::~FPGACKKSContext() = default;

        FPGACKKSContext::FPGACKKSContext(FPGACKKSContext&&) noexcept = default;
        FPGACKKSContext& FPGACKKSContext::operator=(FPGACKKSContext&&) noexcept = default;

        void FPGACKKSContext::init_queue(bool use_emulator)
        {
            auto exception_handler = [](sycl::exception_list exceptions) {
                for (const auto& e : exceptions)
                {
                    try
                    {
                        std::rethrow_exception(e);
                    }
                    catch (const sycl::exception& ex)
                    {
                        throw std::runtime_error(std::string("SYCL exception: ") + ex.what());
                    }
                }
            };

            if (use_emulator)
            {
                queue_ = sycl::queue(sycl::ext::intel::fpga_emulator_selector_v,
                                     exception_handler,
                                     sycl::property::queue::enable_profiling());
            }
            else
            {
                queue_ = sycl::queue(sycl::ext::intel::fpga_selector_v,
                                     exception_handler,
                                     sycl::property::queue::enable_profiling());
            }
        }

        void FPGACKKSContext::compute_index_map()
        {
            std::size_t n = params_.poly_modulus_degree;
            std::size_t slots = n >> 1;
            int logn = static_cast<int>(params_.log_poly_modulus_degree);

            matrix_reps_index_map_.resize(n);

            std::uint64_t gen = 3;
            std::uint64_t pos = 1;
            std::uint64_t m = static_cast<std::uint64_t>(n) << 1;

            for (std::size_t i = 0; i < slots; i++)
            {
                std::uint64_t index1 = (pos - 1) >> 1;
                std::uint64_t index2 = (m - pos - 1) >> 1;

                matrix_reps_index_map_[i] = static_cast<std::size_t>(reverse_bits(index1, logn));
                matrix_reps_index_map_[slots | i] = static_cast<std::size_t>(reverse_bits(index2, logn));

                pos *= gen;
                pos &= (m - 1);
            }
        }

        void FPGACKKSContext::compute_dwt_roots()
        {
            std::size_t n = params_.poly_modulus_degree;
            int logn = static_cast<int>(params_.log_poly_modulus_degree);
            std::size_t m = n << 1;

            dwt_root_powers_.resize(n);
            dwt_inv_root_powers_.resize(n);

            double angle_base = 2.0 * M_PI / static_cast<double>(m);

            for (std::size_t i = 1; i < n; i++)
            {
                std::size_t reversed = reverse_bits(i, logn);
                double angle = angle_base * static_cast<double>(reversed);
                dwt_root_powers_[i] = std::complex<double>(std::cos(angle), std::sin(angle));

                std::size_t inv_idx = reverse_bits(i - 1, logn) + 1;
                double inv_angle = -angle_base * static_cast<double>(inv_idx);
                dwt_inv_root_powers_[i] = std::complex<double>(std::cos(inv_angle), std::sin(inv_angle));
            }
        }

        void FPGACKKSContext::compute_ntt_roots()
        {
            std::size_t n = params_.poly_modulus_degree;
            int logn = static_cast<int>(params_.log_poly_modulus_degree);
            std::uint64_t modulus = params_.modulus;

            ntt_root_powers_.resize(n);
            ntt_inv_root_powers_.resize(n);

            std::uint64_t root = find_primitive_root(n << 1, modulus);
            std::uint64_t inv_root = mod_inverse(root, modulus);

            for (std::size_t i = 1; i < n; i++)
            {
                std::size_t reversed = reverse_bits(i, logn);
                ntt_root_powers_[i] = mod_exp(root, reversed, modulus);

                std::size_t inv_idx = reverse_bits(i - 1, logn) + 1;
                ntt_inv_root_powers_[i] = mod_exp(inv_root, inv_idx, modulus);
            }
        }

        std::uint64_t FPGACKKSContext::find_primitive_root(std::size_t degree, std::uint64_t modulus)
        {
            std::uint64_t phi_n = modulus - 1;

            if (phi_n % degree != 0)
            {
                throw std::invalid_argument("degree does not divide phi(modulus)");
            }

            std::uint64_t exp = phi_n / degree;

            for (std::uint64_t candidate = 2; candidate < modulus; candidate++)
            {
                std::uint64_t root = mod_exp(candidate, exp, modulus);

                if (mod_exp(root, degree / 2, modulus) == modulus - 1)
                {
                    return root;
                }
            }

            throw std::runtime_error("failed to find primitive root");
        }

        std::uint64_t FPGACKKSContext::mod_exp(std::uint64_t base, std::uint64_t exp, std::uint64_t mod)
        {
            std::uint64_t result = 1;
            base %= mod;

            while (exp > 0)
            {
                if (exp & 1)
                {
                    __uint128_t temp = static_cast<__uint128_t>(result) * base;
                    result = temp % mod;
                }
                __uint128_t temp = static_cast<__uint128_t>(base) * base;
                base = temp % mod;
                exp >>= 1;
            }
            return result;
        }

        std::uint64_t FPGACKKSContext::mod_inverse(std::uint64_t a, std::uint64_t mod)
        {
            std::int64_t t = 0, newt = 1;
            std::int64_t r = static_cast<std::int64_t>(mod);
            std::int64_t newr = static_cast<std::int64_t>(a);

            while (newr != 0)
            {
                std::int64_t quotient = r / newr;

                std::int64_t temp_t = t;
                t = newt;
                newt = temp_t - quotient * newt;

                std::int64_t temp_r = r;
                r = newr;
                newr = temp_r - quotient * newr;
            }

            if (r > 1)
            {
                throw std::invalid_argument("modular inverse does not exist");
            }

            if (t < 0)
            {
                t += static_cast<std::int64_t>(mod);
            }

            return static_cast<std::uint64_t>(t);
        }

        std::uint64_t FPGACKKSContext::reverse_bits(std::uint64_t value, int bit_count)
        {
            std::uint64_t result = 0;
            for (int i = 0; i < bit_count; i++)
            {
                result = (result << 1) | (value & 1);
                value >>= 1;
            }
            return result;
        }

#endif

    } // namespace fpga
} // namespace seal
