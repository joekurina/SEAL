// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <cstddef>
#include <complex>

namespace seal
{
    namespace fpga
    {
        constexpr std::size_t MAX_POLY_DEGREE = 32768;
        constexpr std::size_t MAX_SLOTS = MAX_POLY_DEGREE / 2;

        struct FPGAInputPacket
        {
            std::size_t poly_modulus_degree;
            int log_poly_modulus_degree;
            std::uint64_t modulus;
            double scale;
            std::uint64_t barrett_ratio[2];

            std::complex<double> prepared_values[MAX_POLY_DEGREE];
            std::complex<double> dwt_inv_roots[MAX_POLY_DEGREE];
            std::uint64_t ntt_roots[MAX_POLY_DEGREE];
            std::uint64_t secret_key_ntt[MAX_POLY_DEGREE];

            std::int64_t error_samples[MAX_POLY_DEGREE];
            std::uint64_t uniform_poly_ntt[MAX_POLY_DEGREE];
        };

        struct FPGAOutputPacket
        {
            std::size_t poly_modulus_degree;
            std::uint64_t c0[MAX_POLY_DEGREE];
            std::uint64_t c1[MAX_POLY_DEGREE];
        };

        struct DWTPacket
        {
            std::size_t n;
            int log_n;
            double scale_factor;
            std::complex<double> values[MAX_POLY_DEGREE];
            std::complex<double> inv_roots[MAX_POLY_DEGREE];

            // Pass-through data for downstream kernels
            std::uint64_t modulus;
            std::uint64_t barrett_ratio[2];
            std::uint64_t ntt_roots[MAX_POLY_DEGREE];
            std::uint64_t secret_key_ntt[MAX_POLY_DEGREE];
            std::uint64_t uniform_poly_ntt[MAX_POLY_DEGREE];
            std::int64_t error_samples[MAX_POLY_DEGREE];
        };

        struct ScaleReducePacket
        {
            std::size_t n;
            int log_n;
            std::uint64_t modulus;
            std::uint64_t barrett_ratio[2];
            std::complex<double> values[MAX_POLY_DEGREE];

            // Pass-through data for downstream kernels
            std::uint64_t ntt_roots[MAX_POLY_DEGREE];
            std::uint64_t secret_key_ntt[MAX_POLY_DEGREE];
            std::uint64_t uniform_poly_ntt[MAX_POLY_DEGREE];
            std::int64_t error_samples[MAX_POLY_DEGREE];
        };

        struct NTTPacket
        {
            std::size_t n;
            int log_n;
            std::uint64_t modulus;
            std::uint64_t coeffs[MAX_POLY_DEGREE];
            std::uint64_t roots[MAX_POLY_DEGREE];

            // Pass-through data for encrypt kernel
            std::uint64_t secret_key_ntt[MAX_POLY_DEGREE];
            std::uint64_t uniform_poly_ntt[MAX_POLY_DEGREE];
            std::int64_t error_samples[MAX_POLY_DEGREE];
        };

        struct EncryptPacket
        {
            std::size_t n;
            int log_n;
            std::uint64_t modulus;
            std::uint64_t plaintext_ntt[MAX_POLY_DEGREE];
            std::uint64_t secret_key_ntt[MAX_POLY_DEGREE];
            std::uint64_t uniform_poly_ntt[MAX_POLY_DEGREE];
            std::int64_t error_samples[MAX_POLY_DEGREE];
            std::uint64_t ntt_roots[MAX_POLY_DEGREE];
        };

        struct CiphertextPacket
        {
            std::size_t n;
            std::uint64_t c0[MAX_POLY_DEGREE];
            std::uint64_t c1[MAX_POLY_DEGREE];
        };

    } // namespace fpga
} // namespace seal
