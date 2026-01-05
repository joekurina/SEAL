// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "fpga_kernel_types.h"
#include "fpga_ckks_context.h"
#include "fpga_ckks_encoder.h"

#ifdef SEAL_USE_FPGA
#include <sycl/sycl.hpp>
#endif

#include <vector>
#include <complex>

namespace seal
{
    namespace fpga
    {
        class FPGAPipeline
        {
        public:
            FPGAPipeline(const FPGACKKSParams& params);

            void encrypt(
                const std::vector<double>& values,
                const std::uint64_t* secret_key_ntt,
                const std::uint64_t* ntt_roots,
                const std::complex<double>* dwt_inv_roots,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out);

            void encrypt(
                const std::vector<std::complex<double>>& values,
                const std::uint64_t* secret_key_ntt,
                const std::uint64_t* ntt_roots,
                const std::complex<double>* dwt_inv_roots,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out);

#ifdef SEAL_USE_FPGA
            void encrypt_fpga(
                sycl::queue& q,
                const std::vector<double>& values,
                const std::uint64_t* secret_key_ntt,
                const std::uint64_t* ntt_roots,
                const std::complex<double>* dwt_inv_roots,
                std::vector<std::uint64_t>& c0_out,
                std::vector<std::uint64_t>& c1_out);
#endif

        private:
            FPGACKKSParams params_;
            FPGACKKSEncoder encoder_;

            void prepare_input_packet(
                const std::vector<std::complex<double>>& prepared_values,
                const std::uint64_t* secret_key_ntt,
                const std::uint64_t* ntt_roots,
                const std::complex<double>* dwt_inv_roots,
                FPGAInputPacket& packet);

            void execute_host_pipeline(
                const FPGAInputPacket& input,
                FPGAOutputPacket& output);

#ifdef SEAL_USE_FPGA
            void execute_fpga_pipeline(
                sycl::queue& q,
                const FPGAInputPacket& input,
                FPGAOutputPacket& output);
#endif
        };
    }
}
