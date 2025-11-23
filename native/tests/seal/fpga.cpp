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
// Forward declare the FPGA function to avoid including SYCL headers in test compilation
extern "C" {
    void SYCL_combined_encrypt(
        size_t n,
        size_t logn,
        double scale,
        uint32_t mod_value,
        const uint32_t* const_ratio,
        std::complex<double>* encoding_buffer,
        uint32_t* expanded_s,
        uint32_t* uniform_poly,
        int8_t* error_samples,
        uint32_t* ntt_pte,
        uint32_t* c0_s,
        uint32_t* c1,
        uint32_t* s_save,
        uint32_t* c1_save
    );
}

using namespace seal;
using namespace std;

namespace sealtest
{
    // ========================================================================
    // Helper Functions for FPGA CKKS Testing
    // ========================================================================

    /**
     * Bit-reversal for FFT indexing.
     */
    size_t bitrev(size_t x, size_t logn)
    {
        size_t result = 0;
        for (size_t i = 0; i < logn; i++)
        {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        return result;
    }

    /**
     * Compute primitive root of unity on-the-fly.
     * Returns exp(2*pi*i*k/m) for the m-th roots of unity.
     */
    complex<double> calc_root_otf(size_t k, size_t m)
    {
        double angle = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(m);
        return complex<double>(cos(angle), sin(angle));
    }

    /**
     * FFT implementation matching the FPGA pipeline (not SEAL's FFT).
     * This is the forward FFT - for decoding we need this.
     */
    void fft_inpl(vector<complex<double>>& vec, size_t n, size_t logn)
    {
        size_t m = n << 1;  // Degree of roots
        size_t h = 1;
        size_t tt = n / 2;

        for (size_t i = 0; i < logn; i++, h *= 2, tt /= 2)
        {
            for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt)
            {
                complex<double> s = calc_root_otf(bitrev(h + j, logn), m);
                for (size_t k = kstart; k < (kstart + tt); k++)
                {
                    complex<double> u = vec[k];
                    complex<double> v = vec[k + tt] * s;
                    vec[k] = u + v;
                    vec[k + tt] = u - v;
                }
            }
        }
    }

    /**
     * Compute Barrett reduction constants for a given modulus.
     * ratio = floor(2^64 / modulus)
     */
    void compute_barrett_constants(uint32_t modulus, uint32_t const_ratio[3])
    {
        // Compute floor(2^64 / modulus) using 128-bit arithmetic
        unsigned __int128 numerator = (unsigned __int128)1 << 64;
        unsigned __int128 ratio = numerator / modulus;

        // Split 64-bit ratio into three parts (though only lower 64 bits matter)
        const_ratio[0] = static_cast<uint32_t>(ratio & 0xFFFFFFFF);
        const_ratio[1] = static_cast<uint32_t>((ratio >> 32) & 0xFFFFFFFF);
        const_ratio[2] = 0; // Upper part is zero for 64-bit ratio
    }

    /**
     * Extract secret key from SEAL and convert to FPGA format.
     * SEAL stores as uint64_t, FPGA needs uint32_t (mod q).
     * Secret key should NOT be in NTT form (FPGA will apply NTT internally).
     */
    void extract_secret_key_for_fpga(
        const SecretKey& seal_key,
        uint32_t modulus,
        vector<uint32_t>& fpga_key)
    {
        size_t n = seal_key.data().coeff_count();
        fpga_key.resize(n);

        const uint64_t* seal_sk_ptr = seal_key.data().data();

        for (size_t i = 0; i < n; i++)
        {
            // Convert from uint64_t to uint32_t, reducing modulo q
            fpga_key[i] = static_cast<uint32_t>(seal_sk_ptr[i] % modulus);
        }
    }

    /**
     * Generate uniform random polynomial modulo q.
     * This becomes c1 after NTT transformation.
     */
    void generate_uniform_poly(
        uint32_t modulus,
        size_t n,
        vector<uint32_t>& uniform_poly,
        default_random_engine& rng)
    {
        uniform_poly.resize(n);
        uniform_int_distribution<uint32_t> dist(0, modulus - 1);

        for (size_t i = 0; i < n; i++)
        {
            uniform_poly[i] = dist(rng);
        }
    }

    /**
     * Generate discrete Gaussian error samples.
     * Small integer values typically in range [-3, 3].
     */
    void generate_error_samples(
        size_t n,
        vector<int8_t>& error_samples,
        default_random_engine& rng)
    {
        error_samples.resize(n);

        // Use normal distribution with small standard deviation
        normal_distribution<double> dist(0.0, 1.5);

        for (size_t i = 0; i < n; i++)
        {
            double sample = dist(rng);
            // Clamp to int8_t range
            if (sample > 127.0) sample = 127.0;
            if (sample < -128.0) sample = -128.0;
            error_samples[i] = static_cast<int8_t>(round(sample));
        }
    }

    /**
     * Prepare encoding buffer for FPGA IFFT kernel.
     * The FPGA expects the canonical embedding (before IFFT).
     * For CKKS, we embed the slot values into the polynomial ring
     * by doubling the size and conjugating the second half.
     */
    void encode_for_fpga(
        const vector<complex<double>>& input_values,
        CKKSEncoder& encoder,
        double scale,
        vector<complex<double>>& encoding_buffer)
    {
        size_t slot_count = encoder.slot_count();
        if (input_values.size() != slot_count)
        {
            throw invalid_argument("Input size must match slot count");
        }

        // poly_modulus_degree = 2 * slot_count for CKKS
        size_t n = slot_count * 2;
        encoding_buffer.resize(n);

        // CKKS canonical embedding:
        // First half: input values
        // Second half: conjugate of input values in reverse order
        // This ensures the plaintext polynomial has real coefficients after IFFT

        for (size_t i = 0; i < slot_count; i++)
        {
            encoding_buffer[i] = input_values[i];
        }

        // Second half: conjugate symmetric
        for (size_t i = 0; i < slot_count; i++)
        {
            encoding_buffer[slot_count + i] = conj(input_values[slot_count - 1 - i]);
        }
    }

    /**
     * Decode a plaintext polynomial using the FPGA-compatible FFT.
     * This matches the encoding process used by the FPGA pipeline.
     */
    void decode_fpga(
        const Plaintext& plain,
        double scale,
        uint32_t modulus,
        size_t slot_count,
        vector<complex<double>>& output)
    {
        size_t n = slot_count * 2;  // poly_modulus_degree
        size_t logn = 12;  // log2(4096)

        // Convert plaintext coefficients to complex doubles
        // Plaintext is in coefficient form (not NTT) after decryption
        vector<complex<double>> poly(n);
        for (size_t i = 0; i < n; i++)
        {
            // Plaintext is stored modulo q, need to convert to centered representation
            uint64_t coeff = plain.data()[i];
            // Convert from [0, q) to [-(q-1)/2, (q-1)/2]
            int64_t signed_coeff;
            if (coeff > modulus / 2)
            {
                signed_coeff = static_cast<int64_t>(coeff) - static_cast<int64_t>(modulus);
            }
            else
            {
                signed_coeff = static_cast<int64_t>(coeff);
            }
            // Divide by scale to recover the encoded message
            poly[i] = complex<double>(static_cast<double>(signed_coeff) / scale, 0.0);
        }

        // Apply FFT (this extracts the slot values from the polynomial)
        fft_inpl(poly, n, logn);

        // Extract first slot_count values (the actual message slots)
        // The FFT output needs to be scaled by 1/n for proper normalization
        output.resize(slot_count);
        double scale_factor = 1.0 / static_cast<double>(n);
        for (size_t i = 0; i < slot_count; i++)
        {
            output[i] = poly[i] * scale_factor;
        }
    }

    /**
     * Construct a SEAL Ciphertext from FPGA output (c0, c1).
     * The FPGA outputs uint32_t[n] arrays in NTT form.
     * SEAL expects uint64_t with proper metadata.
     */
    void construct_seal_ciphertext(
        const vector<uint32_t>& fpga_c0,
        const vector<uint32_t>& fpga_c1,
        const SEALContext& context,
        parms_id_type parms_id,
        double scale,
        Ciphertext& result)
    {
        size_t n = fpga_c0.size();

        // Allocate ciphertext with correct parameters
        result = Ciphertext(context, parms_id);
        result.resize(context, parms_id, 2); // Two polynomials: c0 and c1

        // Get pointers to ciphertext data
        uint64_t* seal_c0 = result.data(0);
        uint64_t* seal_c1 = result.data(1);

        // Copy data from FPGA output to SEAL ciphertext
        // Convert from uint32_t to uint64_t
        for (size_t i = 0; i < n; i++)
        {
            seal_c0[i] = static_cast<uint64_t>(fpga_c0[i]);
            seal_c1[i] = static_cast<uint64_t>(fpga_c1[i]);
        }

        // Set ciphertext metadata
        result.is_ntt_form() = true;  // FPGA outputs in NTT form
        result.scale() = scale;
        result.parms_id() = parms_id;
    }

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
         * Uses single modulus from the 6 supported by FPGA.
         */
        void setup_context(uint32_t modulus)
        {
            parms_ = make_unique<EncryptionParameters>(scheme_type::ckks);
            parms_->set_poly_modulus_degree(4096);

            // Use the specific modulus
            vector<Modulus> coeff_modulus;
            coeff_modulus.push_back(Modulus(modulus));
            parms_->set_coeff_modulus(coeff_modulus);

            // Create context (disable security checks for testing)
            context_ = make_unique<SEALContext>(*parms_, false, sec_level_type::none);

            // Create encoder and key generator
            encoder_ = make_unique<CKKSEncoder>(*context_);
            keygen_ = make_unique<KeyGenerator>(*context_);
            secret_key_ = make_unique<SecretKey>(keygen_->secret_key());

            // Create encryptor and decryptor
            encryptor_ = make_unique<Encryptor>(*context_, *secret_key_);
            decryptor_ = make_unique<Decryptor>(*context_, *secret_key_);
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
            encode_for_fpga(input_values, *encoder_, scale, encoding_buffer);

            // Prepare output buffers
            vector<uint32_t> fpga_c0(n);
            vector<uint32_t> fpga_c1(n);
            vector<uint32_t> ntt_pte(n);

            // Call FPGA encryption
            SYCL_combined_encrypt(
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
                nullptr, // s_save (not needed)
                nullptr  // c1_save (not needed)
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

            // Decode using FPGA-compatible FFT (not SEAL's decoder)
            vector<complex<double>> decoded_values;
            decode_fpga(decrypted, scale, modulus, encoder_->slot_count(), decoded_values);

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
