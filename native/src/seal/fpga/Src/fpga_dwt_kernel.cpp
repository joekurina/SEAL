// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_dwt_kernel.h"
#include "../Inc/fpga_ifft_core.h"

#ifdef SEAL_USE_FPGA

namespace seal
{
    namespace fpga
    {
        sycl::event submit_dwt_inverse_kernel(sycl::queue& q)
        {
            return q.submit([&](sycl::handler& h) {
                h.single_task<DWTInverseKernel>([=]() {
                    DWTPacket pkt = pipes::EntranceToDWTPipe::read();

                    std::size_t n = pkt.n;
                    int log_n = pkt.log_n;
                    double scale_factor = pkt.scale_factor;

                    std::complex<double> local_values[MAX_POLY_DEGREE];
                    std::complex<double> local_roots[MAX_POLY_DEGREE];

                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] = pkt.values[i];
                        local_roots[i] = pkt.inv_roots[i];
                    }

                    // ==========================================================
                    // IFFT CORE - RTL REPLACEMENT POINT
                    // Replace this call with RTL invocation when available.
                    // Input:  local_values (N complex doubles, bit-reversed)
                    //         local_roots (N complex twiddles, bit-reversed)
                    // Output: local_values (N complex doubles, natural order)
                    // ==========================================================
                    ifft_dif_core(local_values, local_roots, n);
                    // ==========================================================
                    // END IFFT CORE
                    // ==========================================================

                    // Post-processing: Apply CKKS scaling factor
                    for (std::size_t i = 0; i < n; i++)
                    {
                        local_values[i] *= scale_factor;
                    }

                    ScaleReducePacket out_pkt;
                    out_pkt.n = n;
                    out_pkt.log_n = log_n;
                    out_pkt.modulus = pkt.modulus;
                    out_pkt.barrett_ratio[0] = pkt.barrett_ratio[0];
                    out_pkt.barrett_ratio[1] = pkt.barrett_ratio[1];
                    for (std::size_t i = 0; i < n; i++)
                    {
                        out_pkt.values[i] = local_values[i];
                        out_pkt.ntt_roots[i] = pkt.ntt_roots[i];
                        out_pkt.secret_key_ntt[i] = pkt.secret_key_ntt[i];
                        out_pkt.uniform_poly_ntt[i] = pkt.uniform_poly_ntt[i];
                        out_pkt.error_samples[i] = pkt.error_samples[i];
                    }

                    pipes::DWTToScaleReducePipe::write(out_pkt);
                });
            });
        }
    }
}

#endif
