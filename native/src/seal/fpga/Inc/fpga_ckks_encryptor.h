// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>
#include <vector>

#include "fpga_ckks_context.h"
#include "fpga_ckks_encoder.h"
#include "fpga_dwt.h"
#include "fpga_ntt.h"
#include "fpga_encrypt.h"

namespace seal
{
    namespace fpga
    {
        /**
         * FPGACKKSEncryptor orchestrates the full CKKS symmetric encode/encrypt pipeline.
         *
         * Pipeline:
         *   1. prepare_for_fpga() - Index mapping and conjugate embedding
         *   2. dwt_inverse()      - Discrete Weighted Transform (inverse FFT)
         *   3. scale_and_reduce() - Scale by encryption scale and reduce mod q
         *   4. ntt_forward()      - Number Theoretic Transform
         *   5. encrypt_symmetric()- c0 = -a*s + e + m, c1 = a
         *
         * Output: (c0, c1) ciphertext polynomials in NTT form
         */
        class FPGACKKSEncryptor
        {
        public:
            /**
             * Construct encryptor with FPGA context and compute roots internally.
             * @param params FPGA CKKS parameters (poly_modulus_degree, modulus, scale)
             * @param secret_key_ntt Secret key polynomial in NTT form (size = poly_modulus_degree)
             */
            FPGACKKSEncryptor(
                const FPGACKKSParams& params,
                const std::uint64_t* secret_key_ntt);

            /**
             * Construct encryptor with externally provided NTT roots (for SEAL compatibility).
             * @param params FPGA CKKS parameters
             * @param secret_key_ntt Secret key polynomial in NTT form
             * @param ntt_root_powers NTT forward roots in bit-reversed order (size = poly_modulus_degree)
             * @param inv_n Inverse of N modulo the modulus
             */
            FPGACKKSEncryptor(
                const FPGACKKSParams& params,
                const std::uint64_t* secret_key_ntt,
                const std::uint64_t* ntt_root_powers,
                std::uint64_t inv_n);

            /**
             * Encrypt complex values using the full FPGA pipeline (host version).
             *
             * @param values Input complex values (size = slot_count = poly_modulus_degree/2)
             * @param c0_out Output ciphertext component c0 in NTT form (size = poly_modulus_degree)
             * @param c1_out Output ciphertext component c1 in NTT form (size = poly_modulus_degree)
             * @param seed Optional seed for random sampling (default = 0 for time-based)
             */
            void encrypt_host(
                const std::vector<std::complex<double>>& values,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out,
                std::uint64_t seed = 0) const;

            /**
             * Encrypt real values using the full FPGA pipeline (host version).
             *
             * @param values Input real values (size = slot_count = poly_modulus_degree/2)
             * @param c0_out Output ciphertext component c0 in NTT form (size = poly_modulus_degree)
             * @param c1_out Output ciphertext component c1 in NTT form (size = poly_modulus_degree)
             * @param seed Optional seed for random sampling (default = 0 for time-based)
             */
            void encrypt_host(
                const std::vector<double>& values,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out,
                std::uint64_t seed = 0) const;

#ifdef SEAL_USE_FPGA
            /**
             * Encrypt complex values using the full FPGA pipeline (SYCL device version).
             *
             * @param q SYCL queue for device execution
             * @param values Input complex values (size = slot_count)
             * @param c0_out Output ciphertext component c0 in NTT form (size = poly_modulus_degree)
             * @param c1_out Output ciphertext component c1 in NTT form (size = poly_modulus_degree)
             * @param seed Optional seed for random sampling
             */
            void encrypt(
                sycl::queue& q,
                const std::vector<std::complex<double>>& values,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out,
                std::uint64_t seed = 0) const;

            /**
             * Encrypt real values using the full FPGA pipeline (SYCL device version).
             *
             * @param q SYCL queue for device execution
             * @param values Input real values (size = slot_count)
             * @param c0_out Output ciphertext component c0 in NTT form (size = poly_modulus_degree)
             * @param c1_out Output ciphertext component c1 in NTT form (size = poly_modulus_degree)
             * @param seed Optional seed for random sampling
             */
            void encrypt(
                sycl::queue& q,
                const std::vector<double>& values,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out,
                std::uint64_t seed = 0) const;
#endif

            std::size_t slot_count() const { return params_.poly_modulus_degree / 2; }
            std::size_t poly_modulus_degree() const { return params_.poly_modulus_degree; }
            const FPGACKKSParams& params() const { return params_; }

        private:
            void encrypt_internal_host(
                const std::vector<std::complex<double>>& prepared_values,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out,
                std::uint64_t seed) const;

#ifdef SEAL_USE_FPGA
            void encrypt_internal(
                sycl::queue& q,
                const std::vector<std::complex<double>>& prepared_values,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out,
                std::uint64_t seed) const;
#endif

            void init_dwt_roots();

            FPGACKKSParams params_;
            FPGACKKSEncoder encoder_;
            std::vector<std::uint64_t> secret_key_ntt_;

            std::vector<std::complex<double>> dwt_inv_roots_;
            std::vector<std::uint64_t> ntt_roots_;
            std::uint64_t inv_n_;
        };

    } // namespace fpga
} // namespace seal
