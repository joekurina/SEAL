// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <complex>
#include <vector>
#include <random>

namespace seal
{
    namespace fpga
    {
        class FPGACKKSEncoder
        {
        public:
            FPGACKKSEncoder(std::size_t poly_modulus_degree);

            void prepare_for_fpga(
                const std::vector<std::complex<double>>& values,
                std::vector<std::complex<double>>& destination) const;

            void prepare_for_fpga(
                const std::vector<double>& values,
                std::vector<std::complex<double>>& destination) const;

            std::size_t slot_count() const { return poly_modulus_degree_ / 2; }
            std::size_t poly_modulus_degree() const { return poly_modulus_degree_; }
            const std::vector<std::size_t>& index_map() const { return matrix_reps_index_map_; }

        private:
            void compute_index_map();
            static std::size_t reverse_bits(std::size_t value, int bit_count);

            std::size_t poly_modulus_degree_;
            int log_poly_modulus_degree_;
            std::vector<std::size_t> matrix_reps_index_map_;
        };

        class FPGACBDSampler
        {
        public:
            explicit FPGACBDSampler(std::uint64_t seed = 12345);

            void sample(std::size_t count, std::vector<std::int64_t>& destination);
            void sample(std::size_t count, std::int64_t* destination);

        private:
            static int hamming_weight(std::uint8_t byte);

            std::mt19937_64 rng_;
        };

        class FPGAUniformSampler
        {
        public:
            explicit FPGAUniformSampler(std::uint64_t seed = 54321);

            void sample(std::size_t count, std::uint64_t modulus, std::vector<std::uint64_t>& destination);
            void sample(std::size_t count, std::uint64_t modulus, std::uint64_t* destination);

        private:
            std::mt19937_64 rng_;
        };

    } // namespace fpga
} // namespace seal
