// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <complex>
#include <vector>

#ifdef SEAL_USE_FPGA
#include <sycl/sycl.hpp>
#endif

namespace seal
{
    namespace fpga
    {
        void dwt_inverse_host(
            std::complex<double>* values,
            std::size_t n,
            int log_n,
            const std::complex<double>* inv_roots,
            double scale_factor);

#ifdef SEAL_USE_FPGA

        sycl::event dwt_inverse(
            sycl::queue& q,
            std::complex<double>* values,
            std::size_t n,
            int log_n,
            const std::complex<double>* inv_roots,
            double scale_factor);

        sycl::event dwt_inverse_batched(
            sycl::queue& q,
            sycl::buffer<std::complex<double>, 1>& values,
            int log_n,
            sycl::buffer<std::complex<double>, 1>& inv_roots,
            double scale_factor);

#endif

    } // namespace fpga
} // namespace seal
