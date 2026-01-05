// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_ckks_encryptor.h"
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace seal
{
    namespace fpga
    {
        namespace
        {
            std::size_t reverse_bits(std::size_t value, int bit_count)
            {
                std::size_t result = 0;
                for (int i = 0; i < bit_count; i++)
                {
                    result = (result << 1) | (value & 1);
                    value >>= 1;
                }
                return result;
            }

            std::uint64_t mod_exp(std::uint64_t base, std::uint64_t exp, std::uint64_t mod)
            {
                std::uint64_t result = 1;
                base %= mod;
                while (exp > 0)
                {
                    if (exp & 1)
                    {
                        __uint128_t temp = static_cast<__uint128_t>(result) * base;
                        result = static_cast<std::uint64_t>(temp % mod);
                    }
                    exp >>= 1;
                    __uint128_t temp = static_cast<__uint128_t>(base) * base;
                    base = static_cast<std::uint64_t>(temp % mod);
                }
                return result;
            }

            std::uint64_t multiply_uint_mod(std::uint64_t a, std::uint64_t b, std::uint64_t mod)
            {
                __uint128_t temp = static_cast<__uint128_t>(a) * b;
                return static_cast<std::uint64_t>(temp % mod);
            }

            bool is_primitive_root(std::uint64_t root, std::uint64_t degree, std::uint64_t mod)
            {
                if (root == 0)
                {
                    return false;
                }
                return mod_exp(root, degree >> 1, mod) == (mod - 1);
            }

            bool try_primitive_root(std::uint64_t degree, std::uint64_t mod, std::uint64_t& dest)
            {
                std::uint64_t size_entire_group = mod - 1;
                std::uint64_t size_quotient_group = size_entire_group / degree;

                for (std::uint64_t attempt = 0; attempt < 100; attempt++)
                {
                    std::uint64_t root = mod_exp(attempt + 2, size_quotient_group, mod);
                    if (is_primitive_root(root, degree, mod))
                    {
                        dest = root;
                        return true;
                    }
                }
                return false;
            }

            bool try_minimal_primitive_root(std::uint64_t degree, std::uint64_t mod, std::uint64_t& dest)
            {
                std::uint64_t root;
                if (!try_primitive_root(degree, mod, root))
                {
                    return false;
                }
                std::uint64_t generator_sq = multiply_uint_mod(root, root, mod);
                std::uint64_t current_generator = root;

                for (std::size_t i = 0; i < degree; i += 2)
                {
                    if (current_generator < root)
                    {
                        root = current_generator;
                    }
                    current_generator = multiply_uint_mod(current_generator, generator_sq, mod);
                }

                dest = root;
                return true;
            }

            std::uint64_t mod_inverse(std::uint64_t a, std::uint64_t mod)
            {
                return mod_exp(a, mod - 2, mod);
            }
        }

        void FPGACKKSEncryptor::init_dwt_roots()
        {
            std::size_t n = params_.poly_modulus_degree;
            int log_n = static_cast<int>(params_.log_poly_modulus_degree);

            dwt_inv_roots_.resize(n);
            double m = static_cast<double>(n << 1);
            double angle_base = 2.0 * M_PI / m;

            for (std::size_t i = 1; i < n; i++)
            {
                std::size_t inv_idx = reverse_bits(i - 1, log_n) + 1;
                double inv_angle = -angle_base * static_cast<double>(inv_idx);
                dwt_inv_roots_[i] = std::complex<double>(std::cos(inv_angle), std::sin(inv_angle));
            }
        }

        FPGACKKSEncryptor::FPGACKKSEncryptor(
            const FPGACKKSParams& params,
            const std::uint64_t* secret_key_ntt)
            : params_(params)
            , encoder_(params.poly_modulus_degree)
        {
            std::size_t n = params_.poly_modulus_degree;
            int log_n = static_cast<int>(params_.log_poly_modulus_degree);
            std::uint64_t modulus = params_.modulus;

            secret_key_ntt_.assign(secret_key_ntt, secret_key_ntt + n);

            init_dwt_roots();

            std::uint64_t root;
            try_minimal_primitive_root(n << 1, modulus, root);

            ntt_roots_.resize(n);
            ntt_roots_[0] = 1;
            std::uint64_t power = root;
            for (std::size_t i = 1; i < n; i++)
            {
                ntt_roots_[reverse_bits(i, log_n)] = power;
                power = multiply_uint_mod(power, root, modulus);
            }

            inv_n_ = mod_inverse(static_cast<std::uint64_t>(n), modulus);
        }

        FPGACKKSEncryptor::FPGACKKSEncryptor(
            const FPGACKKSParams& params,
            const std::uint64_t* secret_key_ntt,
            const std::uint64_t* ntt_root_powers,
            std::uint64_t inv_n)
            : params_(params)
            , encoder_(params.poly_modulus_degree)
            , inv_n_(inv_n)
        {
            std::size_t n = params_.poly_modulus_degree;

            secret_key_ntt_.assign(secret_key_ntt, secret_key_ntt + n);
            ntt_roots_.assign(ntt_root_powers, ntt_root_powers + n);

            init_dwt_roots();
        }

        void FPGACKKSEncryptor::encrypt_host(
            const std::vector<std::complex<double>>& values,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out,
            std::uint64_t seed) const
        {
            std::vector<std::complex<double>> prepared;
            encoder_.prepare_for_fpga(values, prepared);
            encrypt_internal_host(prepared, c0_out, c1_out, seed);
        }

        void FPGACKKSEncryptor::encrypt_host(
            const std::vector<double>& values,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out,
            std::uint64_t seed) const
        {
            std::vector<std::complex<double>> prepared;
            encoder_.prepare_for_fpga(values, prepared);
            encrypt_internal_host(prepared, c0_out, c1_out, seed);
        }

        void FPGACKKSEncryptor::encrypt_internal_host(
            const std::vector<std::complex<double>>& prepared_values,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out,
            std::uint64_t seed) const
        {
            std::size_t n = params_.poly_modulus_degree;
            int log_n = static_cast<int>(params_.log_poly_modulus_degree);
            std::uint64_t modulus = params_.modulus;
            double scale = params_.scale;

            if (seed == 0)
            {
                seed = static_cast<std::uint64_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count());
            }

            std::vector<std::complex<double>> dwt_values = prepared_values;

            double scale_factor = scale / static_cast<double>(n);
            dwt_inverse_host(dwt_values.data(), n, log_n, dwt_inv_roots_.data(), scale_factor);

            std::vector<std::uint64_t> plaintext_coeffs(n);
            scale_and_reduce_host(dwt_values.data(), plaintext_coeffs.data(), n, modulus, params_.barrett_ratio);

            std::vector<std::uint64_t> plaintext_ntt = plaintext_coeffs;
            ntt_forward_host(plaintext_ntt.data(), n, log_n, ntt_roots_.data(), modulus);

            FPGACBDSampler cbd_sampler(seed);
            std::vector<std::int64_t> error_samples(n);
            cbd_sampler.sample(n, error_samples.data());

            FPGAUniformSampler uniform_sampler(seed ^ 0xDEADBEEF);
            std::vector<std::uint64_t> uniform_poly_ntt(n);
            uniform_sampler.sample(n, modulus, uniform_poly_ntt.data());

            c0_out.resize(n);
            c1_out.resize(n);

            encrypt_symmetric_host(
                plaintext_ntt.data(),
                secret_key_ntt_.data(),
                uniform_poly_ntt.data(),
                error_samples.data(),
                c0_out.data(),
                c1_out.data(),
                n,
                log_n,
                modulus,
                ntt_roots_.data());
        }

#ifdef SEAL_USE_FPGA

        void FPGACKKSEncryptor::encrypt(
            sycl::queue& q,
            const std::vector<std::complex<double>>& values,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out,
            std::uint64_t seed) const
        {
            std::vector<std::complex<double>> prepared;
            encoder_.prepare_for_fpga(values, prepared);
            encrypt_internal(q, prepared, c0_out, c1_out, seed);
        }

        void FPGACKKSEncryptor::encrypt(
            sycl::queue& q,
            const std::vector<double>& values,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out,
            std::uint64_t seed) const
        {
            std::vector<std::complex<double>> prepared;
            encoder_.prepare_for_fpga(values, prepared);
            encrypt_internal(q, prepared, c0_out, c1_out, seed);
        }

        void FPGACKKSEncryptor::encrypt_internal(
            sycl::queue& q,
            const std::vector<std::complex<double>>& prepared_values,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out,
            std::uint64_t seed) const
        {
            std::size_t n = params_.poly_modulus_degree;
            int log_n = static_cast<int>(params_.log_poly_modulus_degree);
            std::uint64_t modulus = params_.modulus;
            double scale = params_.scale;

            if (seed == 0)
            {
                seed = static_cast<std::uint64_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count());
            }

            std::vector<std::complex<double>> dwt_values = prepared_values;

            double scale_factor = scale / static_cast<double>(n);
            auto dwt_event = dwt_inverse(q, dwt_values.data(), n, log_n, dwt_inv_roots_.data(), scale_factor);
            dwt_event.wait();

            std::vector<std::uint64_t> plaintext_coeffs(n);
            auto scale_event = scale_and_reduce(q, dwt_values.data(), plaintext_coeffs.data(), n, modulus, params_.barrett_ratio);
            scale_event.wait();

            std::vector<std::uint64_t> plaintext_ntt = plaintext_coeffs;
            auto ntt_event = ntt_forward(q, plaintext_ntt.data(), n, log_n, ntt_roots_.data(), modulus);
            ntt_event.wait();

            FPGACBDSampler cbd_sampler(seed);
            std::vector<std::int64_t> error_samples(n);
            cbd_sampler.sample(n, error_samples.data());

            FPGAUniformSampler uniform_sampler(seed ^ 0xDEADBEEF);
            std::vector<std::uint64_t> uniform_poly_ntt(n);
            uniform_sampler.sample(n, modulus, uniform_poly_ntt.data());

            c0_out.resize(n);
            c1_out.resize(n);

            auto enc_event = encrypt_symmetric(
                q,
                plaintext_ntt.data(),
                secret_key_ntt_.data(),
                uniform_poly_ntt.data(),
                error_samples.data(),
                c0_out.data(),
                c1_out.data(),
                n,
                log_n,
                modulus,
                ntt_roots_.data());
            enc_event.wait();
        }

#endif

    } // namespace fpga
} // namespace seal
