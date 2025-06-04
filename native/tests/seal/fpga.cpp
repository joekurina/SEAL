// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "gtest/gtest.h"
#include "seal/seal.h"
#include "seal/fpga/fpga_encoder.h" // Include the FPGAEncoder header
#include <vector>
#include <complex>
#include <cmath>   // For std::abs, std::pow
#include <cstdlib> // For srand, rand
#include <ctime>   // For time

// Namespace for SEAL tests
namespace sealtest
{
    // Test case for encoding a vector of complex numbers and then decoding.
    // This test verifies the basic encode-decode pipeline for various parameters.
    TEST(FPGAEncoderTest, FPGAEncoderEncodeVectorDecodeTest)
    {
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        // Test case 1: Small number of slots, specific coefficient moduli and scale, zero input
        {
            std::size_t slots = 32;
            parms.set_poly_modulus_degree(slots << 1); // N = 2 * slots
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slots << 1, { 40, 40, 40, 40 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {0.0, 0.0}; // Initialize with complex zero
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 16); // Scale
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); // Use FPGAEncoder for decoding

            // Check if decoded values are close to original values
            for (std::size_t i = 0; i < slots; ++i)
            {
                // Check real part
                auto tmp_real = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp_real < 0.5);
                // Check imaginary part
                auto tmp_imag = std::abs(values[i].imag() - result[i].imag());
                ASSERT_TRUE(tmp_imag < 0.5);
            }
        }

        // Test case 2: Small number of slots, larger coefficient moduli and scale, random real values
        {
            std::size_t slots = 32;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slots << 1, { 60, 60, 60, 60 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);

            for (std::size_t i = 0; i < slots; i++)
            {
                // Encoding real values as complex numbers with zero imaginary part
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 40);
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); // Use FPGAEncoder for decoding

            for (std::size_t i = 0; i < slots; ++i)
            {
                auto tmp = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
        }

        // Test case 3: More slots, different coefficient moduli configuration
        {
            std::size_t slots = 64;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slots << 1, { 60, 60, 60 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            int data_bound = (1 << 30);

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); 
            double delta = (1ULL << 40);
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); 

            for (std::size_t i = 0; i < slots; ++i)
            {
                auto tmp = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
        }

        // Test case 4: More slots, more smaller primes
        {
            std::size_t slots = 64;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slots << 1, { 30, 30, 30, 30, 30 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            int data_bound = (1 << 20); 

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); 
            double delta = (1ULL << 25); 
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); 

            for (std::size_t i = 0; i < slots; ++i)
            {
                auto tmp = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
        }

        // Test case 5: Smaller poly_modulus_degree (N=128, slots=64)
        {
            std::size_t poly_degree = 128;
            std::size_t slots = poly_degree / 2; 
            parms.set_poly_modulus_degree(poly_degree);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_degree, { 30, 30, 30, 30, 30 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots); 
            int data_bound = (1 << 20); 

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); 
            double delta = (1ULL << 25); 
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); 

            for (std::size_t i = 0; i < slots; ++i)
            {
                auto tmp = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
        }

        // Test case 6: Many primes
        {
            std::size_t poly_degree = 128;
            std::size_t slots = poly_degree / 2;
            parms.set_poly_modulus_degree(poly_degree);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(
                poly_degree, { 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            int data_bound = (1 << 10); 

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); 
            double delta = (1ULL << 20); 
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); 

            for (std::size_t i = 0; i < slots; ++i)
            {
                auto tmp = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
        }

        // Test case 7: Test with very large scale values
        {
            std::size_t slots = 64;
            parms.set_poly_modulus_degree(slots << 1); // N = 128
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slots << 1, { 40, 40, 40, 40, 40 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            int data_bound = (1 << 20);

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); 

            // Sub-case 7a: Very large scale (pow(2.0, 110))
            {
                double delta = std::pow(2.0, 110);
                seal::Plaintext plain;
                encoder.encode(values, context.first_parms_id(), delta, plain);
                std::vector<std::complex<double>> result;
                encoder.decode(plain, result); 

                for (std::size_t i = 0; i < slots; ++i)
                {
                    auto tmp = std::abs(values[i].real() - result[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                    ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                }
            }
            // Sub-case 7b: Even larger scale (pow(2.0, 130))
            {
                double delta = std::pow(2.0, 130);
                seal::Plaintext plain;
                encoder.encode(values, context.first_parms_id(), delta, plain);
                std::vector<std::complex<double>> result;
                encoder.decode(plain, result); 

                for (std::size_t i = 0; i < slots; ++i)
                {
                    auto tmp = std::abs(values[i].real() - result[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                    ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                }
            }
        }
    }

    // Test case for encoding a single double or int64_t value and then decoding.
    // This verifies the non-FFT paths and replication across slots.
    TEST(FPGAEncoderTest, FPGAEncoderEncodeSingleDecodeTest)
    {
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        // Test case 1: Single double value
        {
            std::size_t poly_degree = 64; // N=64
            std::size_t slots = poly_degree / 2; // slots = 32
            parms.set_poly_modulus_degree(poly_degree);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_degree, { 40, 40, 40, 40 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);
            double delta = (1ULL << 16);
            seal::Plaintext plain;
            std::vector<std::complex<double>> result;

            for (int iRun = 0; iRun < 50; iRun++)
            {
                double value = static_cast<double>(rand() % data_bound);
                encoder.encode(value, context.first_parms_id(), delta, plain);
                encoder.decode(plain, result); // Use FPGAEncoder for decoding

                for (std::size_t i = 0; i < slots; ++i)
                {
                    auto tmp = std::abs(value - result[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                    ASSERT_NEAR(result[i].imag(), 0.0, 0.5); 
                }
            }
        }

        // Test case 2: Single int64_t value (scale is implicitly 1.0)
        {
            std::size_t poly_degree = 64; // N=64
            std::size_t slots = poly_degree / 2; // slots = 32
            parms.set_poly_modulus_degree(poly_degree);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_degree, { 40, 40, 40, 40 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            
            srand(static_cast<unsigned>(time(NULL))); 
            
            // Sub-case 2a: Moderate data bound
            {
                int data_bound = (1 << 30);
                seal::Plaintext plain;
                std::vector<std::complex<double>> result;

                for (int iRun = 0; iRun < 50; iRun++)
                {
                    std::int64_t value = static_cast<std::int64_t>(rand() % data_bound);
                    if (rand() % 2) value = -value; 
                    encoder.encode(value, context.first_parms_id(), plain);
                    encoder.decode(plain, result); // Use FPGAEncoder for decoding

                    for (std::size_t i = 0; i < slots; ++i)
                    {
                        auto tmp = std::abs(static_cast<double>(value) - result[i].real());
                        ASSERT_TRUE(tmp < 0.5);
                        ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                    }
                }
            }
            // Sub-case 2b: Smaller data bound
            {
                int data_bound = (1 << 20);
                seal::Plaintext plain;
                std::vector<std::complex<double>> result;

                for (int iRun = 0; iRun < 50; iRun++)
                {
                    std::int64_t value = static_cast<std::int64_t>(rand() % data_bound);
                    if (rand() % 2) value = -value;
                    encoder.encode(value, context.first_parms_id(), plain);
                    encoder.decode(plain, result); // Use FPGAEncoder for decoding

                    for (std::size_t i = 0; i < slots; ++i)
                    {
                        auto tmp = std::abs(static_cast<double>(value) - result[i].real());
                        ASSERT_TRUE(tmp < 0.5);
                        ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                    }
                }
            }
        }
    }
} // namespace sealtest
