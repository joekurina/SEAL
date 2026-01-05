// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef SEAL_USE_FPGA
#include <sycl/sycl.hpp>
#endif

namespace seal
{
    namespace fpga
    {
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
            const std::uint64_t* ntt_root_powers);

        void add_poly_mod_host(
            const std::uint64_t* a,
            const std::uint64_t* b,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus);

        void negate_poly_mod_host(
            const std::uint64_t* src,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus);

        void dyadic_product_mod_host(
            const std::uint64_t* a,
            const std::uint64_t* b,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus);

        void error_to_ntt_host(
            const std::int64_t* error_samples,
            std::uint64_t* dest,
            std::size_t n,
            int log_n,
            std::uint64_t modulus,
            const std::uint64_t* ntt_root_powers);

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
            const std::uint64_t* ntt_root_powers);

#endif

    } // namespace fpga
} // namespace seal
