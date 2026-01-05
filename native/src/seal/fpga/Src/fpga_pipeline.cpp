// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_pipeline.h"
#include "../Inc/fpga_dwt.h"
#include "../Inc/fpga_ntt.h"
#include "../Inc/fpga_encrypt.h"
#include <chrono>
#include <random>

#ifdef SEAL_USE_FPGA
#include "../Inc/fpga_entrance_kernel.h"
#include "../Inc/fpga_dwt_kernel.h"
#include "../Inc/fpga_ntt_kernel.h"
#include "../Inc/fpga_encrypt_kernel.h"
#include "../Inc/fpga_exit_kernel.h"
#endif

namespace seal
{
    namespace fpga
    {
        FPGAPipeline::FPGAPipeline(const FPGACKKSParams& params)
            : params_(params)
            , encoder_(params.poly_modulus_degree)
        {
        }

        void FPGAPipeline::prepare_input_packet(
            const std::vector<std::complex<double>>& prepared_values,
            const std::uint64_t* secret_key_ntt,
            const std::uint64_t* ntt_roots,
            const std::complex<double>* dwt_inv_roots,
            FPGAInputPacket& packet)
        {
            std::size_t n = params_.poly_modulus_degree;

            packet.poly_modulus_degree = n;
            packet.log_poly_modulus_degree = params_.log_poly_modulus_degree;
            packet.modulus = params_.modulus;
            packet.scale = params_.scale;
            packet.barrett_ratio[0] = params_.barrett_ratio[0];
            packet.barrett_ratio[1] = params_.barrett_ratio[1];

            for (std::size_t i = 0; i < n; i++)
            {
                packet.prepared_values[i] = prepared_values[i];
                packet.dwt_inv_roots[i] = dwt_inv_roots[i];
                packet.ntt_roots[i] = ntt_roots[i];
                packet.secret_key_ntt[i] = secret_key_ntt[i];
            }

            std::uint64_t seed = static_cast<std::uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());

            FPGACBDSampler cbd_sampler(seed);
            std::vector<std::int64_t> error_samples(n);
            cbd_sampler.sample(n, error_samples.data());

            FPGAUniformSampler uniform_sampler(seed ^ 0xDEADBEEF);
            std::vector<std::uint64_t> uniform_poly(n);
            uniform_sampler.sample(n, params_.modulus, uniform_poly.data());

            for (std::size_t i = 0; i < n; i++)
            {
                packet.error_samples[i] = error_samples[i];
                packet.uniform_poly_ntt[i] = uniform_poly[i];
            }
        }

        void FPGAPipeline::execute_host_pipeline(
            const FPGAInputPacket& input,
            FPGAOutputPacket& output)
        {
            std::size_t n = input.poly_modulus_degree;
            int log_n = input.log_poly_modulus_degree;
            std::uint64_t modulus = input.modulus;

            std::vector<std::complex<double>> dwt_values(n);
            for (std::size_t i = 0; i < n; i++)
            {
                dwt_values[i] = input.prepared_values[i];
            }

            double scale_factor = input.scale / static_cast<double>(n);
            dwt_inverse_host(dwt_values.data(), n, log_n, input.dwt_inv_roots, scale_factor);

            std::vector<std::uint64_t> plaintext_coeffs(n);
            scale_and_reduce_host(dwt_values.data(), plaintext_coeffs.data(), n, modulus, input.barrett_ratio);

            std::vector<std::uint64_t> plaintext_ntt = plaintext_coeffs;
            ntt_forward_host(plaintext_ntt.data(), n, log_n, input.ntt_roots, modulus);

            std::vector<std::uint64_t> c0(n), c1(n);
            encrypt_symmetric_host(
                plaintext_ntt.data(),
                input.secret_key_ntt,
                input.uniform_poly_ntt,
                input.error_samples,
                c0.data(),
                c1.data(),
                n,
                log_n,
                modulus,
                input.ntt_roots);

            output.poly_modulus_degree = n;
            for (std::size_t i = 0; i < n; i++)
            {
                output.c0[i] = c0[i];
                output.c1[i] = c1[i];
            }
        }

        void FPGAPipeline::encrypt(
            const std::vector<double>& values,
            const std::uint64_t* secret_key_ntt,
            const std::uint64_t* ntt_roots,
            const std::complex<double>* dwt_inv_roots,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out)
        {
            std::vector<std::complex<double>> prepared;
            encoder_.prepare_for_fpga(values, prepared);

            FPGAInputPacket input;
            prepare_input_packet(prepared, secret_key_ntt, ntt_roots, dwt_inv_roots, input);

            FPGAOutputPacket output;
            execute_host_pipeline(input, output);

            std::size_t n = output.poly_modulus_degree;
            c0_out.resize(n);
            c1_out.resize(n);
            for (std::size_t i = 0; i < n; i++)
            {
                c0_out[i] = output.c0[i];
                c1_out[i] = output.c1[i];
            }
        }

        void FPGAPipeline::encrypt(
            const std::vector<std::complex<double>>& values,
            const std::uint64_t* secret_key_ntt,
            const std::uint64_t* ntt_roots,
            const std::complex<double>* dwt_inv_roots,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out)
        {
            std::vector<std::complex<double>> prepared;
            encoder_.prepare_for_fpga(values, prepared);

            FPGAInputPacket input;
            prepare_input_packet(prepared, secret_key_ntt, ntt_roots, dwt_inv_roots, input);

            FPGAOutputPacket output;
            execute_host_pipeline(input, output);

            std::size_t n = output.poly_modulus_degree;
            c0_out.resize(n);
            c1_out.resize(n);
            for (std::size_t i = 0; i < n; i++)
            {
                c0_out[i] = output.c0[i];
                c1_out[i] = output.c1[i];
            }
        }

#ifdef SEAL_USE_FPGA

        void FPGAPipeline::execute_fpga_pipeline(
            sycl::queue& q,
            const FPGAInputPacket& input,
            FPGAOutputPacket& output)
        {
            auto entrance_event = submit_entrance_kernel(q, input);
            auto dwt_event = submit_dwt_inverse_kernel(q);
            auto scale_reduce_event = submit_scale_reduce_kernel(q);
            auto ntt_event = submit_ntt_forward_kernel(q);
            auto encrypt_event = submit_encrypt_kernel(q);
            auto exit_event = submit_exit_kernel(q, output);

            exit_event.wait();
        }

        void FPGAPipeline::encrypt_fpga(
            sycl::queue& q,
            const std::vector<double>& values,
            const std::uint64_t* secret_key_ntt,
            const std::uint64_t* ntt_roots,
            const std::complex<double>* dwt_inv_roots,
            std::vector<std::uint64_t>& c0_out,
            std::vector<std::uint64_t>& c1_out)
        {
            std::vector<std::complex<double>> prepared;
            encoder_.prepare_for_fpga(values, prepared);

            FPGAInputPacket input;
            prepare_input_packet(prepared, secret_key_ntt, ntt_roots, dwt_inv_roots, input);

            FPGAOutputPacket output;
            execute_fpga_pipeline(q, input, output);

            std::size_t n = output.poly_modulus_degree;
            c0_out.resize(n);
            c1_out.resize(n);
            for (std::size_t i = 0; i < n; i++)
            {
                c0_out[i] = output.c0[i];
                c1_out[i] = output.c1[i];
            }
        }

#endif

    }
}
