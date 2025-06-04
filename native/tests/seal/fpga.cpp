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
#include <iostream> // For debugging
#include <iomanip>  // For std::fixed, std::setprecision

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
            std::cout << "--- FPGAEncoderEncodeVectorDecodeTest: Test Case 1 (All Zeros) ---" << std::endl;
            std::size_t slots = 32;
            parms.set_poly_modulus_degree(slots << 1); // N = 2 * slots
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slots << 1, { 40, 40, 40, 40 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            std::cout << "Input 'values' (first 5 elements):" << std::endl;
            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {0.0, 0.0}; // Initialize with complex zero
                if (i < 5) {
                    std::cout << "  values[" << i << "] = " << std::fixed << std::setprecision(5) << values[i] << std::endl;
                }
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 16); // Scale
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::cout << "Plaintext scale after encode: " << plain.scale() << std::endl;
            std::cout << "Plaintext data (first 5 uint64_t from first RNS component):" << std::endl;
            if (plain.coeff_count() > 0) { // Changed from plain.data_size()
                 for(size_t i = 0; i < std::min(plain.coeff_count(), static_cast<std::size_t>(5)); ++i) {
                    std::cout << "  plain.data()[" << i << "] = " << plain.data()[i] << std::endl;
                }
            }


            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); // Use FPGAEncoder for decoding

            std::cout << "Decoded 'result' (first 5 elements):" << std::endl;
            for (std::size_t i = 0; i < std::min(slots, static_cast<std::size_t>(5)); ++i) {
                 std::cout << "  result[" << i << "] = " << std::fixed << std::setprecision(5) << result[i] << std::endl;
            }

            // Check if decoded values are close to original values
            for (std::size_t i = 0; i < slots; ++i)
            {
                // Check real part
                auto tmp_real = std::abs(values[i].real() - result[i].real());
                if (!(tmp_real < 0.5)) {
                    std::cout << "FAIL at index " << i << " (real part): values.real=" << values[i].real()
                              << ", result.real=" << result[i].real() << ", diff=" << tmp_real << std::endl;
                }
                ASSERT_TRUE(tmp_real < 0.5);
                // Check imaginary part
                auto tmp_imag = std::abs(values[i].imag() - result[i].imag());
                 if (!(tmp_imag < 0.5)) {
                    std::cout << "FAIL at index " << i << " (imag part): values.imag=" << values[i].imag()
                              << ", result.imag=" << result[i].imag() << ", diff=" << tmp_imag << std::endl;
                }
                ASSERT_TRUE(tmp_imag < 0.5);
            }
            std::cout << "--- Test Case 1 (All Zeros) Finished ---" << std::endl;
        }

        // Test case 2: Small number of slots, larger coefficient moduli and scale, random real values
        {
            std::cout << "\n--- FPGAEncoderEncodeVectorDecodeTest: Test Case 2 (Random Reals) ---" << std::endl;
            std::size_t slots = 32;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(slots << 1, { 60, 60, 60, 60 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);

            std::vector<std::complex<double>> values(slots);
            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);

            std::cout << "Input 'values' (first 5 elements, random real):" << std::endl;
            for (std::size_t i = 0; i < slots; i++)
            {
                // Encoding real values as complex numbers with zero imaginary part
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
                 if (i < 5) {
                    std::cout << "  values[" << i << "] = " << std::fixed << std::setprecision(1) << values[i] << std::endl;
                }
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 40);
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);
            
            std::cout << "Plaintext scale after encode (random): " << plain.scale() << std::endl;
            std::cout << "Plaintext data (first 5 uint64_t from first RNS component, random):" << std::endl;
             if (plain.coeff_count() > 0) { // Changed from plain.data_size()
                 for(size_t i = 0; i < std::min(plain.coeff_count(), static_cast<std::size_t>(5)); ++i) {
                    std::cout << "  plain.data()[" << i << "] = " << plain.data()[i] << std::endl;
                }
            }

            std::vector<std::complex<double>> result;
            encoder.decode(plain, result); // Use FPGAEncoder for decoding

            std::cout << "Decoded 'result' (first 5 elements, random real):" << std::endl;
            for (std::size_t i = 0; i < std::min(slots, static_cast<std::size_t>(5)); ++i) {
                 std::cout << "  result[" << i << "] = " << std::fixed << std::setprecision(1) << result[i] << std::endl;
            }

            for (std::size_t i = 0; i < slots; ++i)
            {
                auto tmp = std::abs(values[i].real() - result[i].real());
                 if (!(tmp < 0.5)) {
                    std::cout << "FAIL at index " << i << " (real part, random): values.real=" << values[i].real()
                              << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                }
                ASSERT_TRUE(tmp < 0.5);
                // Imaginary part should be close to zero if original was zero
                 if (!(std::abs(result[i].imag()) < 0.5)) {
                     std::cout << "FAIL at index " << i << " (imag part, random): result.imag=" << result[i].imag() << std::endl;
                 }
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
             std::cout << "--- Test Case 2 (Random Reals) Finished ---" << std::endl;
        }

        // Test case 3: More slots, different coefficient moduli configuration
        {
            std::cout << "\n--- FPGAEncoderEncodeVectorDecodeTest: Test Case 3 ---" << std::endl;
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
                 if (!(tmp < 0.5)) {
                    std::cout << "TC3 FAIL at index " << i << " (real part): values.real=" << values[i].real()
                              << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                }
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
            std::cout << "--- Test Case 3 Finished ---" << std::endl;
        }

        // Test case 4: More slots, more smaller primes
        {
            std::cout << "\n--- FPGAEncoderEncodeVectorDecodeTest: Test Case 4 ---" << std::endl;
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
                if (!(tmp < 0.5)) {
                    std::cout << "TC4 FAIL at index " << i << " (real part): values.real=" << values[i].real()
                              << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                }
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
             std::cout << "--- Test Case 4 Finished ---" << std::endl;
        }

        // Test case 5: Smaller poly_modulus_degree (N=128, slots=64)
        {
            std::cout << "\n--- FPGAEncoderEncodeVectorDecodeTest: Test Case 5 ---" << std::endl;
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
                 if (!(tmp < 0.5)) {
                    std::cout << "TC5 FAIL at index " << i << " (real part): values.real=" << values[i].real()
                              << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                }
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
            std::cout << "--- Test Case 5 Finished ---" << std::endl;
        }

        // Test case 6: Many primes
        {
            std::cout << "\n--- FPGAEncoderEncodeVectorDecodeTest: Test Case 6 ---" << std::endl;
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
                if (!(tmp < 0.5)) {
                    std::cout << "TC6 FAIL at index " << i << " (real part): values.real=" << values[i].real()
                              << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                }
                ASSERT_TRUE(tmp < 0.5);
                ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
            }
            std::cout << "--- Test Case 6 Finished ---" << std::endl;
        }

        // Test case 7: Test with very large scale values
        {
            std::cout << "\n--- FPGAEncoderEncodeVectorDecodeTest: Test Case 7 ---" << std::endl;
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
                std::cout << "--- Test Case 7a (Large Scale) ---" << std::endl;
                double delta = std::pow(2.0, 110);
                seal::Plaintext plain;
                encoder.encode(values, context.first_parms_id(), delta, plain);
                std::vector<std::complex<double>> result;
                encoder.decode(plain, result); 

                for (std::size_t i = 0; i < slots; ++i)
                {
                    auto tmp = std::abs(values[i].real() - result[i].real());
                     if (!(tmp < 0.5)) {
                        std::cout << "TC7a FAIL at index " << i << " (real part): values.real=" << values[i].real()
                                  << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                    }
                    ASSERT_TRUE(tmp < 0.5);
                    ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                }
            }
            // Sub-case 7b: Even larger scale (pow(2.0, 130))
            {
                 std::cout << "--- Test Case 7b (Very Large Scale) ---" << std::endl;
                double delta = std::pow(2.0, 130);
                seal::Plaintext plain;
                encoder.encode(values, context.first_parms_id(), delta, plain);
                std::vector<std::complex<double>> result;
                encoder.decode(plain, result); 

                for (std::size_t i = 0; i < slots; ++i)
                {
                    auto tmp = std::abs(values[i].real() - result[i].real());
                    if (!(tmp < 0.5)) {
                        std::cout << "TC7b FAIL at index " << i << " (real part): values.real=" << values[i].real()
                                  << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                    }
                    ASSERT_TRUE(tmp < 0.5);
                    ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                }
            }
            std::cout << "--- Test Case 7 Finished ---" << std::endl;
        }
    }

    // Test case for encoding a single double or int64_t value and then decoding.
    // This verifies the non-FFT paths and replication across slots.
    TEST(FPGAEncoderTest, FPGAEncoderEncodeSingleDecodeTest)
    {
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        // Test case 1: Single double value
        {
            std::cout << "\n--- FPGAEncoderEncodeSingleDecodeTest: Test Case 1 (Single Double) ---" << std::endl;
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
                    if (!(tmp < 0.5)) {
                        std::cout << "SingleDouble FAIL at run " << iRun << ", index " << i << ": value=" << value
                                  << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                    }
                    ASSERT_TRUE(tmp < 0.5);
                    ASSERT_NEAR(result[i].imag(), 0.0, 0.5); 
                }
            }
            std::cout << "--- Test Case 1 (Single Double) Finished ---" << std::endl;
        }

        // Test case 2: Single int64_t value (scale is implicitly 1.0)
        {
            std::cout << "\n--- FPGAEncoderEncodeSingleDecodeTest: Test Case 2 (Single Int64) ---" << std::endl;
            std::size_t poly_degree = 64; // N=64
            std::size_t slots = poly_degree / 2; // slots = 32
            parms.set_poly_modulus_degree(poly_degree);
            parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_degree, { 40, 40, 40, 40 }));
            seal::SEALContext context(parms, false, seal::sec_level_type::none);
            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            
            srand(static_cast<unsigned>(time(NULL))); 
            
            // Sub-case 2a: Moderate data bound
            {
                std::cout << "--- Test Case 2a (Single Int64, Moderate Bound) ---" << std::endl;
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
                        if (!(tmp < 0.5)) {
                            std::cout << "SingleInt64 (2a) FAIL at run " << iRun << ", index " << i << ": value=" << value
                                      << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                        }
                        ASSERT_TRUE(tmp < 0.5);
                        ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                    }
                }
            }
            // Sub-case 2b: Smaller data bound
            {
                std::cout << "--- Test Case 2b (Single Int64, Smaller Bound) ---" << std::endl;
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
                        if (!(tmp < 0.5)) {
                            std::cout << "SingleInt64 (2b) FAIL at run " << iRun << ", index " << i << ": value=" << value
                                      << ", result.real=" << result[i].real() << ", diff=" << tmp << std::endl;
                        }
                        ASSERT_TRUE(tmp < 0.5);
                        ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                    }
                }
            }
            std::cout << "--- Test Case 2 (Single Int64) Finished ---" << std::endl;
        }
    }
} // namespace sealtest
