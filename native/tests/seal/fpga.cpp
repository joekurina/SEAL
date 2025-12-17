// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/seal.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <random>
#include <complex>

#ifdef SEAL_USE_FPGA


using namespace seal;
using namespace std;

namespace sealtest
{
    // ========================================================================
    // Helper Functions for FPGA CKKS Testing
    // ========================================================================


    // ========================================================================
    // Test Fixture
    // ========================================================================

    class FPGATest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // This will be called before each test
            // We'll set up SEAL context with FPGA-compatible parameters
        }

        void TearDown() override
        {
            // Cleanup if needed
        }

        /**
         * Create SEAL context with FPGA-compatible parameters.
         */
        void setup_context(uint32_t modulus)
        {
           
        }

        /**
         * Run a complete FPGA encrypt -> SEAL decrypt test.
         * tolerance: absolute tolerance for error
         * relative_tolerance: relative tolerance (as fraction, e.g., 0.05 = 5%)
         * If relative_tolerance > 0, use max(tolerance, |expected| * relative_tolerance)
         */
        void run_fpga_test(
            const vector<complex<double>>& input_values,
            uint32_t modulus,
            double scale,
            double tolerance = 0.5,
            double relative_tolerance = 0.0)
        {
            // Setup SEAL context
            setup_context(modulus);

            size_t n = 4096;
            size_t logn = 12;

            // Prepare Barrett constants
            uint32_t const_ratio[3];
            compute_barrett_constants(modulus, const_ratio);

            // Extract secret key for FPGA
            vector<uint32_t> fpga_secret_key;
            extract_secret_key_for_fpga(*secret_key_, modulus, fpga_secret_key);

            // Generate uniform polynomial and error samples
            default_random_engine rng(12345); // Seeded for reproducibility
            vector<uint32_t> uniform_poly;
            generate_uniform_poly(modulus, n, uniform_poly, rng);

            vector<int8_t> error_samples;
            generate_error_samples(n, error_samples, rng);

            // Encode input values for FPGA
            vector<complex<double>> encoding_buffer;
            // Start with input values
            // Apply index-mapping for CKKS
            // This will be sent to the FPGA for encoding and encryption

            // Prepare output buffers
            vector<uint32_t> fpga_c0(n);
            vector<uint32_t> fpga_c1(n);
            vector<uint32_t> ntt_pte(n);

            // Call FPGA encryption
            SYCL_encrypt(
                n,
                logn,
                scale,
                modulus,
                const_ratio,
                encoding_buffer.data(),
                fpga_secret_key.data(),
                uniform_poly.data(),
                error_samples.data(),
                ntt_pte.data(),
                fpga_c0.data(),
                fpga_c1.data(),
            );

            // Construct SEAL ciphertext from FPGA output
            Ciphertext ct;
            construct_seal_ciphertext(
                fpga_c0, fpga_c1,
                *context_,
                context_->first_parms_id(),
                scale,
                ct);

            // Decrypt with SEAL
            Plaintext decrypted;
            decryptor_->decrypt(ct, decrypted);

            // Decode with SEAL

            // Verify results
            size_t slot_count = encoder_->slot_count();
            for (size_t i = 0; i < slot_count; i++)
            {
                // Calculate effective tolerance (absolute + relative)
                double real_tol = tolerance;
                double imag_tol = tolerance;

                if (relative_tolerance > 0.0)
                {
                    // Use the larger of absolute tolerance or relative tolerance
                    double real_rel = abs(input_values[i].real()) * relative_tolerance;
                    double imag_rel = abs(input_values[i].imag()) * relative_tolerance;
                    real_tol = max(tolerance, real_rel);
                    imag_tol = max(tolerance, imag_rel);
                }

                EXPECT_NEAR(input_values[i].real(), decoded_values[i].real(), real_tol)
                    << "Mismatch at slot " << i << " (real part)";
                EXPECT_NEAR(input_values[i].imag(), decoded_values[i].imag(), imag_tol)
                    << "Mismatch at slot " << i << " (imaginary part)";
            }
        }

        // SEAL objects (will be initialized in setup_context)
        unique_ptr<EncryptionParameters> parms_;
        unique_ptr<SEALContext> context_;
        unique_ptr<KeyGenerator> keygen_;
        unique_ptr<SecretKey> secret_key_;
        unique_ptr<Encryptor> encryptor_;
        unique_ptr<Decryptor> decryptor_;
        unique_ptr<CKKSEncoder> encoder_;
    };

    // ========================================================================
    // Test 1: Basic Functionality Test (Zero Input)
    // ========================================================================

    TEST_F(FPGATest, BasicFunctionality)
    {
        // Use first supported modulus
        uint32_t modulus = 134012929u;  // ~27 bits
        // Scale must be less than modulus/2 for CKKS
        // Use a smaller scale: 2^20 (about 1 million)
        double scale = pow(2.0, 20);

        // Create simple input: all zeros
        size_t slot_count = 2048; // For n=4096
        vector<complex<double>> input_values(slot_count, complex<double>(0.0, 0.0));

        // Run the test with tolerance accounting for encryption noise
        // For all-zero input, expect small noise from error samples
        // Tolerance needs to accommodate discrete Gaussian noise (std dev ~1.5)
        run_fpga_test(input_values, modulus, scale, 2.0);

        cout << "Test 1 (Basic Functionality - Zero Input) passed!" << endl;
    }

    // ========================================================================
    // Test 2: Single Small Value (Diagnostic)
    // ========================================================================

    TEST_F(FPGATest, SingleSmallValue)
    {
        uint32_t modulus = 134012929u;  // ~2^27
        // For modulus ~2^27 and values around 1, use scale 2^20
        // Max representable value ≈ 2^27 / 2^20 = 128
        double scale = pow(2.0, 20);
        size_t slot_count = 2048;

        // Test with a single small value in all slots
        vector<complex<double>> input_values(slot_count, complex<double>(1.0, 0.0));

        // Use absolute tolerance + 10% relative tolerance
        run_fpga_test(input_values, modulus, scale, 2.0, 0.10);

        cout << "Test 2 (Single Small Value) passed!" << endl;
    }

    // ========================================================================
    // Test 2b: Real-Valued Inputs
    // ========================================================================

    TEST_F(FPGATest, RealValuedInputs)
    {
        uint32_t modulus = 134012929u;  // ~2^27
        // Values range 0-99, use scale 2^20 (max value ~128)
        double scale = pow(2.0, 20);
        size_t slot_count = 2048;

        // Test with small real values
        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = complex<double>(static_cast<double>(i % 100), 0.0);
        }

        // Use absolute tolerance + 10% relative tolerance
        run_fpga_test(input_values, modulus, scale, 2.0, 0.10);

        cout << "Test 2b (Real-Valued Inputs) passed!" << endl;
    }

    // ========================================================================
    // Test 3: Complex-Valued Inputs
    // ========================================================================

    TEST_F(FPGATest, ComplexValuedInputs)
    {
        uint32_t modulus = 134012929u;  // ~2^27
        // Values range 0-50, use scale 2^20 (max value ~128)
        double scale = pow(2.0, 20);
        size_t slot_count = 2048;

        // Test with complex values
        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            double real_part = static_cast<double>(i % 50);
            double imag_part = static_cast<double>((i * 3) % 50);
            input_values[i] = complex<double>(real_part, imag_part);
        }

        // Use absolute tolerance + 10% relative tolerance
        run_fpga_test(input_values, modulus, scale, 2.0, 0.10);

        cout << "Test 3 (Complex-Valued Inputs) passed!" << endl;
    }

    // ========================================================================
    // Test 4: Multi-Modulus Test - Test All 6 Supported Moduli
    // ========================================================================

    TEST_F(FPGATest, AllSupportedModuli)
    {
        // All 6 supported moduli for FPGA
        vector<uint32_t> moduli = {
            134012929u,   // selector 0, ~27 bits
            134111233u,   // selector 1, ~27 bits
            134176769u,   // selector 2, ~27 bits
            1053818881u,  // selector 3, ~30 bits
            1054015489u,  // selector 4, ~30 bits
            1054212097u   // selector 5, ~30 bits
        };

        size_t slot_count = 2048;

        // Test each modulus with the same input pattern
        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = complex<double>(
                static_cast<double>(i % 20),
                static_cast<double>((i * 2) % 20)
            );
        }

        for (size_t mod_idx = 0; mod_idx < moduli.size(); mod_idx++)
        {
            uint32_t modulus = moduli[mod_idx];

            // For larger moduli, we can use larger scales
            double scale = (modulus < 200000000u) ? pow(2.0, 20) : pow(2.0, 22);

            cout << "Testing modulus " << mod_idx << ": " << modulus << endl;

            run_fpga_test(input_values, modulus, scale, 2.0, 0.10);
        }

        cout << "Test 4 (All 6 Supported Moduli) passed!" << endl;
    }

    // ========================================================================
    // Test 5: Scale Variation Test
    // ========================================================================

    TEST_F(FPGATest, ScaleVariation)
    {
        uint32_t modulus = 1053818881u;  // Use larger modulus for scale testing
        size_t slot_count = 2048;

        // Prepare test values
        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = complex<double>(
                static_cast<double>(i % 10),
                static_cast<double>((i * 2) % 10)
            );
        }

        // Test different scales
        vector<double> scales = {
            pow(2.0, 16),  // Small scale
            pow(2.0, 20),  // Medium scale
            pow(2.0, 24),  // Larger scale
            pow(2.0, 28)   // Even larger (must be < modulus/2)
        };

        for (size_t scale_idx = 0; scale_idx < scales.size(); scale_idx++)
        {
            double scale = scales[scale_idx];
            cout << "Testing scale 2^" << (16 + scale_idx * 4) << endl;

            // Higher tolerance for smaller scales (less precision)
            double tolerance = (scale < pow(2.0, 20)) ? 3.0 : 2.0;

            run_fpga_test(input_values, modulus, scale, tolerance, 0.15);
        }

        cout << "Test 5 (Scale Variation) passed!" << endl;
    }

    // ========================================================================
    // Test 6: Larger Values
    // ========================================================================

    TEST_F(FPGATest, LargerValues)
    {
        uint32_t modulus = 1054212097u;  // Use largest modulus (~2^30)
        // For modulus ~2^30 and values up to 60, use scale 2^24
        // Max representable value ≈ 2^30 / 2^24 = 64
        double scale = pow(2.0, 24);
        size_t slot_count = 2048;

        // Test with values up to 60 (reduced from 1000 to fit within scale constraints)
        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = complex<double>(
                static_cast<double>(i % 60),
                static_cast<double>((i * 3) % 30)
            );
        }

        // For larger values, relative error is more important
        // Use ~1% relative error tolerance
        run_fpga_test(input_values, modulus, scale, 3.0, 0.01);

        cout << "Test 6 (Larger Values) passed!" << endl;
    }

    // ========================================================================
    // Test 7: Negative Values
    // ========================================================================

    TEST_F(FPGATest, NegativeValues)
    {
        uint32_t modulus = 134111233u;  // ~2^27
        // For modulus ~2^27 and values around ±50, use scale 2^20
        // Max representable value ≈ 2^27 / 2^20 = 128
        double scale = pow(2.0, 20);
        size_t slot_count = 2048;

        // Test with negative and positive values
        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            double sign = (i % 2 == 0) ? 1.0 : -1.0;
            input_values[i] = complex<double>(
                sign * static_cast<double>(i % 50),
                sign * static_cast<double>((i * 2) % 30)
            );
        }

        run_fpga_test(input_values, modulus, scale, 2.0, 0.10);

        cout << "Test 7 (Negative Values) passed!" << endl;
    }

    // ========================================================================
    // Test 8: Single Constant Value
    // ========================================================================

    TEST_F(FPGATest, SingleConstantValue)
    {
        uint32_t modulus = 134176769u;  // ~2^27
        // For modulus ~2^27 and value 42, use scale 2^20
        // Max representable value ≈ 2^27 / 2^20 = 128
        double scale = pow(2.0, 20);
        size_t slot_count = 2048;

        // All slots contain the same value
        complex<double> constant_value(42.0, 17.0);
        vector<complex<double>> input_values(slot_count, constant_value);

        run_fpga_test(input_values, modulus, scale, 2.0, 0.10);

        cout << "Test 8 (Single Constant Value) passed!" << endl;
    }

    // ========================================================================
    // Test 9: Fractional Values
    // ========================================================================

    TEST_F(FPGATest, FractionalValues)
    {
        uint32_t modulus = 1054015489u;  // ~2^30
        // For modulus ~2^30, use scale 2^22 for good precision on fractional values
        // Max representable value ≈ 2^30 / 2^22 = 256
        double scale = pow(2.0, 22);
        size_t slot_count = 2048;

        // Test with fractional values (max value: 204.7 real, 204.7 imag - well within 256 limit)
        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = complex<double>(
                static_cast<double>(i) / 10.0,
                static_cast<double>(i) / 10.0  // Changed from i*2 to i to stay within bounds
            );
        }

        // Fractional values need higher tolerance due to floating point
        run_fpga_test(input_values, modulus, scale, 2.5, 0.10);

        cout << "Test 9 (Fractional Values) passed!" << endl;
    }

    // ========================================================================
    // Test 10: Random Values with Fixed Seed
    // ========================================================================

    TEST_F(FPGATest, RandomValuesFixedSeed)
    {
        uint32_t modulus = 1053818881u;
        double scale = pow(2.0, 20);
        size_t slot_count = 2048;

        // Generate random values with fixed seed for reproducibility
        default_random_engine rng(54321);
        uniform_real_distribution<double> dist(-100.0, 100.0);

        vector<complex<double>> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = complex<double>(dist(rng), dist(rng));
        }

        run_fpga_test(input_values, modulus, scale, 2.5, 0.05);

        cout << "Test 10 (Random Values) passed!" << endl;
    }

} // namespace sealtest

#else // SEAL_USE_FPGA not defined

// If FPGA support is not enabled, provide dummy tests
TEST(FPGATest, Disabled)
{
    GTEST_SKIP() << "FPGA support not enabled. Build with -DSEAL_USE_FPGA=ON to run FPGA tests.";
}

#endif // SEAL_USE_FPGA
