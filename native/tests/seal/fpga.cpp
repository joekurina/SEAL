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
            seal::CKKSEncoder ckks_decoder(context); // Use CKKSEncoder for decoding
            ckks_decoder.decode(plain, result);

            // Check if decoded values are close to original values
            for (std::size_t i = 0; i < slots; ++i)
            {
                // Check real part
                auto tmp_real = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp_real < 0.5);
                // Check imaginary part (should be close to zero for this specific FPGAEncoder path if it zeros out imag parts)
                // If FPGAEncoder is expected to preserve imag parts, this check needs to be more robust.
                // Based on current FPGAEncoder impl, imag part of output from IFFT is zeroed.
                auto tmp_imag = std::abs(values[i].imag() - result[i].imag());
                 // If original imag was 0, decoded should also be close to 0.
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
            seal::CKKSEncoder ckks_decoder(context); // Use CKKSEncoder for decoding
            ckks_decoder.decode(plain, result);

            for (std::size_t i = 0; i < slots; ++i)
            {
                auto tmp = std::abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
                 // Imaginary part should be close to zero if original was zero
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
            // srand already called
            int data_bound = (1 << 30);

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 40);
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            seal::CKKSEncoder ckks_decoder(context); // Use CKKSEncoder for decoding
            ckks_decoder.decode(plain, result);

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
            // srand already called
            int data_bound = (1 << 20); // Reduced bound for smaller primes

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 25); // Adjusted scale
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            seal::CKKSEncoder ckks_decoder(context); // Use CKKSEncoder for decoding
            ckks_decoder.decode(plain, result);

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
            // srand already called
            int data_bound = (1 << 20); 

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 25); 
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            seal::CKKSEncoder ckks_decoder(context); // Use CKKSEncoder for decoding
            ckks_decoder.decode(plain, result);

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
            // srand already called
            int data_bound = (1 << 10); 

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            double delta = (1ULL << 20); 
            seal::Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);

            std::vector<std::complex<double>> result;
            seal::CKKSEncoder ckks_decoder(context); // Use CKKSEncoder for decoding
            ckks_decoder.decode(plain, result);

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
            // srand already called
            int data_bound = (1 << 20);

            for (std::size_t i = 0; i < slots; i++)
            {
                values[i] = {static_cast<double>(rand() % data_bound), 0.0};
            }

            seal::fpga::FPGAEncoder encoder(context); // Use FPGAEncoder
            seal::CKKSEncoder ckks_decoder(context); // Decoder for verification

            // Sub-case 7a: Very large scale (pow(2.0, 110))
            {
                double delta = std::pow(2.0, 110);
                seal::Plaintext plain;
                encoder.encode(values, context.first_parms_id(), delta, plain);
                std::vector<std::complex<double>> result;
                ckks_decoder.decode(plain, result);

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
                ckks_decoder.decode(plain, result);

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
            seal::CKKSEncoder ckks_decoder(context); // For decoding

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);
            double delta = (1ULL << 16);
            seal::Plaintext plain;
            std::vector<std::complex<double>> result;

            for (int iRun = 0; iRun < 50; iRun++)
            {
                double value = static_cast<double>(rand() % data_bound);
                encoder.encode(value, context.first_parms_id(), delta, plain);
                ckks_decoder.decode(plain, result);

                for (std::size_t i = 0; i < slots; ++i)
                {
                    auto tmp = std::abs(value - result[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                    ASSERT_NEAR(result[i].imag(), 0.0, 0.5); // Imaginary part should be zero
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
            seal::CKKSEncoder ckks_decoder(context); // For decoding

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
                    ckks_decoder.decode(plain, result);

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
                    ckks_decoder.decode(plain, result);

                    for (std::size_t i = 0; i < slots; ++i)
                    {
                        auto tmp = std::abs(static_cast<double>(value) - result[i].real());
                        ASSERT_TRUE(tmp < 0.5);
                        ASSERT_NEAR(result[i].imag(), 0.0, 0.5);
                    }
                }
            }
            // Sub-case 2c: Original test had large scale for int64_t, which is not how encode(int64_t,...) works.
            // The CKKSEncoder::encode(int64_t, ...) and FPGAEncoder::encode(int64_t, ...)
            // methods do not take a `scale` parameter and implicitly use a scale of 1.0
            // for the purpose of encoding the integer as a polynomial.
            // The original test's sub-cases for large scales with int64_t input
            // are effectively testing the encode(double, scale, ...) path after casting int to double.
            // If the intent was to test large integer values that might require larger scales
            // to maintain precision *after* encoding, they should be cast to double first and
            // encoded using the double overload.
            // The FPGAEncoder::encode(int64_t, ...) is designed for encoding integers directly
            // without an explicit scale parameter, similar to CKKSEncoder.
        }
    }
} // namespace sealtest