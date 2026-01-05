// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>
#include <vector>

#ifdef SEAL_USE_FPGA
#include <sycl/sycl.hpp>
#endif

namespace seal
{
    namespace fpga
    {
        void scale_and_reduce_host(
            const std::complex<double>* src,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus,
            const std::uint64_t* barrett_ratio);

        void ntt_forward_host(
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* root_powers,
            std::uint64_t modulus);

        void ntt_inverse_host(
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* inv_root_powers,
            std::uint64_t modulus,
            std::uint64_t inv_n);

#ifdef SEAL_USE_FPGA

        sycl::event scale_and_reduce(
            sycl::queue& q,
            const std::complex<double>* src,
            std::uint64_t* dest,
            std::size_t n,
            std::uint64_t modulus,
            const std::uint64_t* barrett_ratio);

        sycl::event ntt_forward(
            sycl::queue& q,
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* root_powers,
            std::uint64_t modulus);

        sycl::event ntt_inverse(
            sycl::queue& q,
            std::uint64_t* values,
            std::size_t n,
            int log_n,
            const std::uint64_t* inv_root_powers,
            std::uint64_t modulus,
            std::uint64_t inv_n);

#endif

    } // namespace fpga
} // namespace seal
