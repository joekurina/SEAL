// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_dwt.h"
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
            std::complex<double> r, u, v;
            std::complex<double>* x = nullptr;
            std::complex<double>* y = nullptr;

            std::size_t gap = 1;
            std::size_t m = n >> 1;
            std::size_t root_idx = 0;

            while (m > 1)
            {
                std::size_t offset = 0;
                if (gap < 4)
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        root_idx++;
                        r = inv_roots[root_idx];
                        x = values + offset;
                        y = x + gap;
                        for (std::size_t j = 0; j < gap; j++)
                        {
                            u = *x;
                            v = *y;
                            *x++ = u + v;
                            *y++ = (u - v) * r;
                        }
                        offset += gap << 1;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < m; i++)
                    {
                        root_idx++;
                        r = inv_roots[root_idx];
                        x = values + offset;
                        y = x + gap;
                        for (std::size_t j = 0; j < gap; j += 4)
                        {
                            u = x[0]; v = y[0];
                            x[0] = u + v;
                            y[0] = (u - v) * r;

                            u = x[1]; v = y[1];
                            x[1] = u + v;
                            y[1] = (u - v) * r;

                            u = x[2]; v = y[2];
                            x[2] = u + v;
                            y[2] = (u - v) * r;

                            u = x[3]; v = y[3];
                            x[3] = u + v;
                            y[3] = (u - v) * r;

                            x += 4;
                            y += 4;
                        }
                        offset += gap << 1;
                    }
                }
                gap <<= 1;
                m >>= 1;
            }

            root_idx++;
            r = inv_roots[root_idx];
            std::complex<double> scaled_r = r * scale_factor;
            x = values;
            y = x + gap;

            if (gap < 4)
            {
                for (std::size_t j = 0; j < gap; j++)
                {
                    u = *x;
                    v = *y;
                    *x++ = (u + v) * scale_factor;
                    *y++ = (u - v) * scaled_r;
                }
            }
            else
            {
                for (std::size_t j = 0; j < gap; j += 4)
                {
                    u = x[0]; v = y[0];
                    x[0] = (u + v) * scale_factor;
                    y[0] = (u - v) * scaled_r;

                    u = x[1]; v = y[1];
                    x[1] = (u + v) * scale_factor;
                    y[1] = (u - v) * scaled_r;

                    u = x[2]; v = y[2];
                    x[2] = (u + v) * scale_factor;
                    y[2] = (u - v) * scaled_r;

                    u = x[3]; v = y[3];
                    x[3] = (u + v) * scale_factor;
                    y[3] = (u - v) * scaled_r;

                    x += 4;
                    y += 4;
                }
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

                    std::complex<double> r, u, v;
                    std::size_t gap = 1;
                    std::size_t m = n >> 1;
                    std::size_t root_idx = 0;

                    while (m > 1)
                    {
                        std::size_t offset = 0;
                        for (std::size_t i = 0; i < m; i++)
                        {
                            root_idx++;
                            r = local_roots[root_idx];
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                std::size_t x_idx = offset + j;
                                std::size_t y_idx = x_idx + gap;
                                u = local_values[x_idx];
                                v = local_values[y_idx];
                                local_values[x_idx] = u + v;
                                local_values[y_idx] = (u - v) * r;
                            }
                            offset += gap << 1;
                        }
                        gap <<= 1;
                        m >>= 1;
                    }

                    root_idx++;
                    r = local_roots[root_idx];
                    std::complex<double> scaled_r = r * scale_factor;

                    for (std::size_t j = 0; j < gap; j++)
                    {
                        std::size_t x_idx = j;
                        std::size_t y_idx = j + gap;
                        u = local_values[x_idx];
                        v = local_values[y_idx];
                        local_values[x_idx] = (u + v) * scale_factor;
                        local_values[y_idx] = (u - v) * scaled_r;
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        values[i] = local_values[i];
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

                    std::complex<double> r, u, v;
                    std::size_t gap = 1;
                    std::size_t m = n >> 1;
                    std::size_t root_idx = 0;

                    while (m > 1)
                    {
                        std::size_t offset = 0;
                        for (std::size_t i = 0; i < m; i++)
                        {
                            root_idx++;
                            r = local_roots[root_idx];
                            for (std::size_t j = 0; j < gap; j++)
                            {
                                std::size_t x_idx = offset + j;
                                std::size_t y_idx = x_idx + gap;
                                u = local_values[x_idx];
                                v = local_values[y_idx];
                                local_values[x_idx] = u + v;
                                local_values[y_idx] = (u - v) * r;
                            }
                            offset += gap << 1;
                        }
                        gap <<= 1;
                        m >>= 1;
                    }

                    root_idx++;
                    r = local_roots[root_idx];
                    std::complex<double> scaled_r = r * scale_factor;

                    for (std::size_t j = 0; j < gap; j++)
                    {
                        std::size_t x_idx = j;
                        std::size_t y_idx = j + gap;
                        u = local_values[x_idx];
                        v = local_values[y_idx];
                        local_values[x_idx] = (u + v) * scale_factor;
                        local_values[y_idx] = (u - v) * scaled_r;
                    }

                    for (std::size_t i = 0; i < n; i++)
                    {
                        val_acc[i] = local_values[i];
                    }
                });
            });
        }

#endif

    } // namespace fpga
} // namespace seal
