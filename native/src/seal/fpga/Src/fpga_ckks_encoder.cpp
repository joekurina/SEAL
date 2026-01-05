// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../Inc/fpga_ckks_encoder.h"
#include <stdexcept>
#include <cstring>

namespace seal
{
    namespace fpga
    {
        FPGACKKSEncoder::FPGACKKSEncoder(std::size_t poly_modulus_degree)
            : poly_modulus_degree_(poly_modulus_degree)
        {
            if (poly_modulus_degree != 4096 && poly_modulus_degree != 8192 &&
                poly_modulus_degree != 16384 && poly_modulus_degree != 32768)
            {
                throw std::invalid_argument("poly_modulus_degree must be 4096, 8192, 16384, or 32768");
            }

            log_poly_modulus_degree_ = 0;
            std::size_t n = poly_modulus_degree;
            while (n > 1)
            {
                n >>= 1;
                log_poly_modulus_degree_++;
            }

            compute_index_map();
        }

        void FPGACKKSEncoder::compute_index_map()
        {
            std::size_t n = poly_modulus_degree_;
            std::size_t slots = n >> 1;

            matrix_reps_index_map_.resize(n);

            std::uint64_t gen = 3;
            std::uint64_t pos = 1;
            std::uint64_t m = static_cast<std::uint64_t>(n) << 1;

            for (std::size_t i = 0; i < slots; i++)
            {
                std::uint64_t index1 = (pos - 1) >> 1;
                std::uint64_t index2 = (m - pos - 1) >> 1;

                matrix_reps_index_map_[i] = reverse_bits(index1, log_poly_modulus_degree_);
                matrix_reps_index_map_[slots | i] = reverse_bits(index2, log_poly_modulus_degree_);

                pos *= gen;
                pos &= (m - 1);
            }
        }

        std::size_t FPGACKKSEncoder::reverse_bits(std::size_t value, int bit_count)
        {
            std::size_t result = 0;
            for (int i = 0; i < bit_count; i++)
            {
                result = (result << 1) | (value & 1);
                value >>= 1;
            }
            return result;
        }

        void FPGACKKSEncoder::prepare_for_fpga(
            const std::vector<std::complex<double>>& values,
            std::vector<std::complex<double>>& destination) const
        {
            std::size_t slots = slot_count();

            if (values.size() > slots)
            {
                throw std::invalid_argument("values size exceeds slot count");
            }

            destination.resize(poly_modulus_degree_);
            std::fill(destination.begin(), destination.end(), std::complex<double>(0.0, 0.0));

            for (std::size_t i = 0; i < values.size(); i++)
            {
                destination[matrix_reps_index_map_[i]] = values[i];
                destination[matrix_reps_index_map_[i + slots]] = std::conj(values[i]);
            }
        }

        void FPGACKKSEncoder::prepare_for_fpga(
            const std::vector<double>& values,
            std::vector<std::complex<double>>& destination) const
        {
            std::size_t slots = slot_count();

            if (values.size() > slots)
            {
                throw std::invalid_argument("values size exceeds slot count");
            }

            destination.resize(poly_modulus_degree_);
            std::fill(destination.begin(), destination.end(), std::complex<double>(0.0, 0.0));

            for (std::size_t i = 0; i < values.size(); i++)
            {
                std::complex<double> val(values[i], 0.0);
                destination[matrix_reps_index_map_[i]] = val;
                destination[matrix_reps_index_map_[i + slots]] = val;
            }
        }

        FPGACBDSampler::FPGACBDSampler(std::uint64_t seed)
            : rng_(seed)
        {
        }

        int FPGACBDSampler::hamming_weight(std::uint8_t byte)
        {
            int count = 0;
            while (byte)
            {
                count += byte & 1;
                byte >>= 1;
            }
            return count;
        }

        void FPGACBDSampler::sample(std::size_t count, std::vector<std::int64_t>& destination)
        {
            destination.resize(count);
            sample(count, destination.data());
        }

        void FPGACBDSampler::sample(std::size_t count, std::int64_t* destination)
        {
            std::uniform_int_distribution<std::uint64_t> dist(0, UINT64_MAX);

            for (std::size_t i = 0; i < count; i++)
            {
                std::uint64_t rand1 = dist(rng_);
                std::uint64_t rand2 = dist(rng_);

                std::uint8_t x0 = static_cast<std::uint8_t>(rand1);
                std::uint8_t x1 = static_cast<std::uint8_t>(rand1 >> 8);
                std::uint8_t x2 = static_cast<std::uint8_t>(rand1 >> 16) & 0x1F;
                std::uint8_t x3 = static_cast<std::uint8_t>(rand2);
                std::uint8_t x4 = static_cast<std::uint8_t>(rand2 >> 8);
                std::uint8_t x5 = static_cast<std::uint8_t>(rand2 >> 16) & 0x1F;

                int positive = hamming_weight(x0) + hamming_weight(x1) + hamming_weight(x2);
                int negative = hamming_weight(x3) + hamming_weight(x4) + hamming_weight(x5);

                destination[i] = static_cast<std::int64_t>(positive - negative);
            }
        }

        FPGAUniformSampler::FPGAUniformSampler(std::uint64_t seed)
            : rng_(seed)
        {
        }

        void FPGAUniformSampler::sample(std::size_t count, std::uint64_t modulus, std::vector<std::uint64_t>& destination)
        {
            destination.resize(count);
            sample(count, modulus, destination.data());
        }

        void FPGAUniformSampler::sample(std::size_t count, std::uint64_t modulus, std::uint64_t* destination)
        {
            std::uniform_int_distribution<std::uint64_t> dist(0, UINT64_MAX);
            std::uint64_t max_random = UINT64_MAX;
            std::uint64_t max_multiple = max_random - (max_random % modulus) - 1;

            for (std::size_t i = 0; i < count; i++)
            {
                std::uint64_t rand;
                do
                {
                    rand = dist(rng_);
                } while (rand >= max_multiple);

                destination[i] = rand % modulus;
            }
        }

    } // namespace fpga
} // namespace seal
