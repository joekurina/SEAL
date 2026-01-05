// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>
#include <memory>
#include <vector>
#include <stdexcept>

#ifdef SEAL_USE_FPGA
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

namespace seal
{
    namespace fpga
    {
        struct FPGACKKSParams
        {
            std::size_t poly_modulus_degree;
            std::size_t log_poly_modulus_degree;
            std::uint64_t modulus;
            double scale;
            std::uint64_t barrett_ratio[2];
            std::uint64_t inv_n_mod_q;
            std::uint64_t two_times_modulus;

            void validate() const;
            void compute_derived_constants();
        };

#ifdef SEAL_USE_FPGA

        class FPGACKKSContext
        {
        public:
            explicit FPGACKKSContext(const FPGACKKSParams& params, bool use_emulator = true);
            ~FPGACKKSContext();

            FPGACKKSContext(const FPGACKKSContext&) = delete;
            FPGACKKSContext& operator=(const FPGACKKSContext&) = delete;
            FPGACKKSContext(FPGACKKSContext&&) noexcept;
            FPGACKKSContext& operator=(FPGACKKSContext&&) noexcept;

            sycl::queue& queue() { return queue_; }
            const sycl::queue& queue() const { return queue_; }
            const FPGACKKSParams& params() const { return params_; }

            const std::complex<double>* dwt_inv_root_powers() const { return dwt_inv_root_powers_.data(); }
            const std::complex<double>* dwt_root_powers() const { return dwt_root_powers_.data(); }
            const std::uint64_t* ntt_root_powers() const { return ntt_root_powers_.data(); }
            const std::uint64_t* ntt_inv_root_powers() const { return ntt_inv_root_powers_.data(); }
            const std::size_t* matrix_reps_index_map() const { return matrix_reps_index_map_.data(); }
            std::size_t slot_count() const { return params_.poly_modulus_degree / 2; }

        private:
            void init_queue(bool use_emulator);
            void compute_dwt_roots();
            void compute_ntt_roots();
            void compute_index_map();

            static std::uint64_t find_primitive_root(std::size_t degree, std::uint64_t modulus);
            static std::uint64_t mod_exp(std::uint64_t base, std::uint64_t exp, std::uint64_t mod);
            static std::uint64_t mod_inverse(std::uint64_t a, std::uint64_t mod);
            static std::uint64_t reverse_bits(std::uint64_t value, int bit_count);

            FPGACKKSParams params_;
            sycl::queue queue_;

            std::vector<std::complex<double>> dwt_root_powers_;
            std::vector<std::complex<double>> dwt_inv_root_powers_;
            std::vector<std::uint64_t> ntt_root_powers_;
            std::vector<std::uint64_t> ntt_inv_root_powers_;
            std::vector<std::size_t> matrix_reps_index_map_;
        };

#endif

        void compute_barrett_ratio(std::uint64_t modulus, std::uint64_t* ratio);
        std::uint64_t barrett_reduce_128(const std::uint64_t* x, std::uint64_t modulus, const std::uint64_t* ratio);
        std::uint64_t barrett_reduce_64(std::uint64_t x, std::uint64_t modulus, const std::uint64_t* ratio);

    } // namespace fpga
} // namespace seal
