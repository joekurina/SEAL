// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_dwt.h"
#include "../Inc/fpga_ifft_core.h"
#include <cmath>

namespace seal
{
    namespace fpga
    {
        void dwt_inverse_host(
            std::complex<double>* values,
            std::size_t n,
            int log_n,
            const std::complex<double>* inv_roots,
            double scale_factor)
        {
            // IFFT CORE - RTL REPLACEMENT POINT
            ifft_dif_core(values, inv_roots, n);

            for (std::size_t i = 0; i < n; i++)
            {
                values[i] *= scale_factor;
            }
        }

#ifdef SEAL_USE_FPGA

        sycl::event dwt_inverse(
            sycl::queue& q,
            std::complex<double>* values,
            std::size_t n,
            int log_n,
            const std::complex<double>* inv_roots,
            double scale_factor)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task([=]() {
                    std::complex<double> local_values[32768];
                    std::complex<double> local_roots[32768];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] = values[i];
                        local_roots[i] = inv_roots[i];
                    }

                    // IFFT CORE - RTL REPLACEMENT POINT
                    ifft_dif_core(local_values, local_roots, n);

                    for (std::size_t i = 0; i < n; i++)
                    {
                        values[i] = local_values[i] * scale_factor;
                    }
                });
            });
        }

        sycl::event dwt_inverse_batched(
            sycl::queue& q,
            sycl::buffer<std::complex<double>, 1>& values,
            int log_n,
            sycl::buffer<std::complex<double>, 1>& inv_roots,
            double scale_factor)
        {
            std::size_t n = std::size_t(1) << log_n;

            return q.submit([&](sycl::handler& h) {
                auto val_acc = values.get_access<sycl::access::mode::read_write>(h);
                auto root_acc = inv_roots.get_access<sycl::access::mode::read>(h);

                h.single_task([=]() {
                    std::complex<double> local_values[32768];
                    std::complex<double> local_roots[32768];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] = val_acc[i];
                        local_roots[i] = root_acc[i];
                    }

                    // IFFT CORE - RTL REPLACEMENT POINT
                    ifft_dif_core(local_values, local_roots, n);

                    for (std::size_t i = 0; i < n; i++)
                    {
                        val_acc[i] = local_values[i] * scale_factor;
                    }
                });
            });
        }

#endif

    } // namespace fpga
} // namespace seal
