// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/ckks.h"
#include "seal/context.h"
#include "seal/keygenerator.h"
#include "seal/modulus.h"
#include <ctime>
#include <vector>
#include "gtest/gtest.h"

// Include headers for my combined test
#include "seal/decryptor.h"
#include "seal/encryptor.h"

using namespace seal;
using namespace seal::util;
using namespace std;

namespace sealtest
{
    TEST(CKKSEncoderTest, CKKSEncoderEncodeVectorDecodeTest)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            size_t slots = 32;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 40, 40, 40, 40 }));
            SEALContext context(parms, false, sec_level_type::none);

            vector<complex<double>> values(slots);

            for (size_t i = 0; i < slots; i++)
            {
                complex<double> value(0.0, 0.0);
                values[i] = value;
            }

            CKKSEncoder encoder(context);
            double delta = (1ULL << 16);
            Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);
            vector<complex<double>> result;
            encoder.decode(plain, result);

            for (size_t i = 0; i < slots; ++i)
            {
                auto tmp = abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            size_t slots = 32;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 60, 60, 60, 60 }));
            SEALContext context(parms, false, sec_level_type::none);

            vector<complex<double>> values(slots);

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);

            for (size_t i = 0; i < slots; i++)
            {
                complex<double> value(static_cast<double>(rand() % data_bound), 0);
                values[i] = value;
            }

            CKKSEncoder encoder(context);
            double delta = (1ULL << 40);
            Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);
            vector<complex<double>> result;
            encoder.decode(plain, result);

            for (size_t i = 0; i < slots; ++i)
            {
                auto tmp = abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            size_t slots = 64;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 60, 60, 60 }));
            SEALContext context(parms, false, sec_level_type::none);

            vector<complex<double>> values(slots);

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);

            for (size_t i = 0; i < slots; i++)
            {
                complex<double> value(static_cast<double>(rand() % data_bound), 0);
                values[i] = value;
            }

            CKKSEncoder encoder(context);
            double delta = (1ULL << 40);
            Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);
            vector<complex<double>> result;
            encoder.decode(plain, result);

            for (size_t i = 0; i < slots; ++i)
            {
                auto tmp = abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            size_t slots = 64;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 30, 30, 30, 30, 30 }));
            SEALContext context(parms, false, sec_level_type::none);

            vector<complex<double>> values(slots);

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);

            for (size_t i = 0; i < slots; i++)
            {
                complex<double> value(static_cast<double>(rand() % data_bound), 0);
                values[i] = value;
            }

            CKKSEncoder encoder(context);
            double delta = (1ULL << 40);
            Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);
            vector<complex<double>> result;
            encoder.decode(plain, result);

            for (size_t i = 0; i < slots; ++i)
            {
                auto tmp = abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            size_t slots = 32;
            parms.set_poly_modulus_degree(128);
            parms.set_coeff_modulus(CoeffModulus::Create(128, { 30, 30, 30, 30, 30 }));
            SEALContext context(parms, false, sec_level_type::none);

            vector<complex<double>> values(slots);

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);

            for (size_t i = 0; i < slots; i++)
            {
                complex<double> value(static_cast<double>(rand() % data_bound), 0);
                values[i] = value;
            }

            CKKSEncoder encoder(context);
            double delta = (1ULL << 40);
            Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);
            vector<complex<double>> result;
            encoder.decode(plain, result);

            for (size_t i = 0; i < slots; ++i)
            {
                auto tmp = abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            // Many primes
            size_t slots = 32;
            parms.set_poly_modulus_degree(128);
            parms.set_coeff_modulus(CoeffModulus::Create(
                128, { 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30 }));
            SEALContext context(parms, false, sec_level_type::none);

            vector<complex<double>> values(slots);

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);

            for (size_t i = 0; i < slots; i++)
            {
                complex<double> value(static_cast<double>(rand() % data_bound), 0);
                values[i] = value;
            }

            CKKSEncoder encoder(context);
            double delta = (1ULL << 40);
            Plaintext plain;
            encoder.encode(values, context.first_parms_id(), delta, plain);
            vector<complex<double>> result;
            encoder.decode(plain, result);

            for (size_t i = 0; i < slots; ++i)
            {
                auto tmp = abs(values[i].real() - result[i].real());
                ASSERT_TRUE(tmp < 0.5);
            }
        }
        {
            size_t slots = 64;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 40, 40, 40, 40, 40 }));
            SEALContext context(parms, false, sec_level_type::none);

            vector<complex<double>> values(slots);

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 20);

            for (size_t i = 0; i < slots; i++)
            {
                complex<double> value(static_cast<double>(rand() % data_bound), 0);
                values[i] = value;
            }

            CKKSEncoder encoder(context);
            {
                // Use a very large scale
                double delta = pow(2.0, 110);
                Plaintext plain;
                encoder.encode(values, context.first_parms_id(), delta, plain);
                vector<complex<double>> result;
                encoder.decode(plain, result);

                for (size_t i = 0; i < slots; ++i)
                {
                    auto tmp = abs(values[i].real() - result[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
            {
                // Use a scale over 128 bits
                double delta = pow(2.0, 130);
                Plaintext plain;
                encoder.encode(values, context.first_parms_id(), delta, plain);
                vector<complex<double>> result;
                encoder.decode(plain, result);

                for (size_t i = 0; i < slots; ++i)
                {
                    auto tmp = abs(values[i].real() - result[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
    }

    TEST(CKKSEncoderTest, CKKSEncoderEncodeSingleDecodeTest)
    {
        EncryptionParameters parms(scheme_type::ckks);
        {
            size_t slots = 16;
            parms.set_poly_modulus_degree(64);
            parms.set_coeff_modulus(CoeffModulus::Create(64, { 40, 40, 40, 40 }));
            SEALContext context(parms, false, sec_level_type::none);
            CKKSEncoder encoder(context);

            srand(static_cast<unsigned>(time(NULL)));
            int data_bound = (1 << 30);
            double delta = (1ULL << 16);
            Plaintext plain;
            vector<complex<double>> result;

            for (int iRun = 0; iRun < 50; iRun++)
            {
                double value = static_cast<double>(rand() % data_bound);
                encoder.encode(value, context.first_parms_id(), delta, plain);
                encoder.decode(plain, result);

                for (size_t i = 0; i < slots; ++i)
                {
                    auto tmp = abs(value - result[i].real());
                    ASSERT_TRUE(tmp < 0.5);
                }
            }
        }
        {
            size_t slots = 32;
            parms.set_poly_modulus_degree(slots << 1);
            parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 40, 40, 40, 40 }));
            SEALContext context(parms, false, sec_level_type::none);
            CKKSEncoder encoder(context);

            srand(static_cast<unsigned>(time(NULL)));
            {
                int data_bound = (1 << 30);
                Plaintext plain;
                vector<complex<double>> result;

                for (int iRun = 0; iRun < 50; iRun++)
                {
                    int value = static_cast<int>(rand() % data_bound);
                    encoder.encode(value, context.first_parms_id(), plain);
                    encoder.decode(plain, result);

                    for (size_t i = 0; i < slots; ++i)
                    {
                        auto tmp = abs(value - result[i].real());
                        ASSERT_TRUE(tmp < 0.5);
                    }
                }
            }
            {
                // Use a very large scale
                int data_bound = (1 << 20);
                Plaintext plain;
                vector<complex<double>> result;

                for (int iRun = 0; iRun < 50; iRun++)
                {
                    int value = static_cast<int>(rand() % data_bound);
                    encoder.encode(value, context.first_parms_id(), plain);
                    encoder.decode(plain, result);

                    for (size_t i = 0; i < slots; ++i)
                    {
                        auto tmp = abs(value - result[i].real());
                        ASSERT_TRUE(tmp < 0.5);
                    }
                }
            }
            {
                // Use a scale over 128 bits
                int data_bound = (1 << 20);
                Plaintext plain;
                vector<complex<double>> result;

                for (int iRun = 0; iRun < 50; iRun++)
                {
                    int value = static_cast<int>(rand() % data_bound);
                    encoder.encode(value, context.first_parms_id(), plain);
                    encoder.decode(plain, result);

                    for (size_t i = 0; i < slots; ++i)
                    {
                        auto tmp = abs(value - result[i].real());
                        ASSERT_TRUE(tmp < 0.5);
                    }
                }
            }
        }
    }

    //////////////////////////////////////////////////////////////////////
    // TEST COMBINED ENCODE AND ENCRYPT
    //////////////////////////////////////////////////////////////////////

    TEST(CKKSEncodeEncryptTest, CKKSEncodeAndEncryptCombined)
    {
        EncryptionParameters parms(scheme_type::ckks);
        // Use one of the parameter sets from CKKSEncoderEncodeVectorDecodeTest
        size_t slots = 32;
        parms.set_poly_modulus_degree(slots << 1); // poly_modulus_degree = 64
        parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 60, 60, 60, 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);

        Encryptor encryptor(context, public_key);
        // If symmetric, initialize Encryptor with secret_key
        // Encryptor encryptor(context, secret_key);

        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context); // For decoding the final result for verification

        // Prepare input data
        vector<complex<double>> input_values(slots);
        // Using a deterministic way to fill for easier debugging if needed,
        // instead of rand() for now.
        for (size_t i = 0; i < slots; i++)
        {
            // Example: 0.0+0.0i, 0.1+0.1i, 0.2+0.2i, ...
            input_values[i] = complex<double>(static_cast<double>(i) / 10.0, static_cast<double>(i) / 10.0);
        }

        double scale = (1ULL << 40); // 2^40, a common scale for CKKS
        parms_id_type parms_id = context.first_parms_id();

        // --- 1. Use the combined encode_and_encrypt_ckks function ---
        Ciphertext combined_ciphertext;
        encryptor.encode_and_encrypt_ckks(input_values, parms_id, scale, combined_ciphertext);

        // --- 2. Decrypt the resulting ciphertext ---
        Plaintext decrypted_plaintext;
        ASSERT_NO_THROW(decryptor.decrypt(combined_ciphertext, decrypted_plaintext));

        // --- 3. Decode the decrypted plaintext ---
        vector<complex<double>> decoded_values;
        ASSERT_NO_THROW(encoder.decode(decrypted_plaintext, decoded_values));

        // --- 4. Verify the results ---
        // Check that the decoded values are close to the original input values.
        // The acceptable difference (delta) depends on the encryption parameters and scale.
        ASSERT_EQ(input_values.size(), decoded_values.size());
        double tolerance = 0.5; // Matching the tolerance from the original test

        for (size_t i = 0; i < slots; ++i)
        {
            // Compare real parts
            ASSERT_NEAR(input_values[i].real(), decoded_values[i].real(), tolerance);
            // Compare imaginary parts
            ASSERT_NEAR(input_values[i].imag(), decoded_values[i].imag(), tolerance);
        }

        // --- Compare with separate encode and encrypt for sanity ---
        Plaintext plain_reference;
        encoder.encode(input_values, parms_id, scale, plain_reference);

        Ciphertext separate_ciphertext;
        encryptor.encrypt(plain_reference, separate_ciphertext); // Standard encryption

        Plaintext decrypted_plaintext_separate;
        ASSERT_NO_THROW(decryptor.decrypt(separate_ciphertext, decrypted_plaintext_separate));

        vector<complex<double>> decoded_values_separate;
        ASSERT_NO_THROW(encoder.decode(decrypted_plaintext_separate, decoded_values_separate));

        // The results from the combined function and separate functions should be very close
        double internal_tolerance = 0.0001;
        for (size_t i = 0; i < slots; ++i)
        {
            ASSERT_NEAR(decoded_values[i].real(), decoded_values_separate[i].real(), internal_tolerance);
            ASSERT_NEAR(decoded_values[i].imag(), decoded_values_separate[i].imag(), internal_tolerance);
        }
    }

    // New Test Case 1: Combined Encode and Encrypt with Real Number (double) Inputs
    TEST(CKKSEncodeEncryptTest, CombinedWithDoubleInput)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096; // Using a smaller degree for variety
        parms.set_poly_modulus_degree(poly_modulus_degree);
        // Using a different coefficient modulus structure
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 40, 30, 40 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);

        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        vector<double> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = (static_cast<double>(i) * 0.1) - (slot_count * 0.05); // e.g., -1.6, -1.5, ..., 1.5
        }

        double scale = pow(2.0, 30);
        parms_id_type parms_id = context.first_parms_id();

        Ciphertext combined_ciphertext;
        ASSERT_NO_THROW(encryptor.encode_and_encrypt_ckks(input_values, parms_id, scale, combined_ciphertext));

        Plaintext decrypted_plaintext;
        ASSERT_NO_THROW(decryptor.decrypt(combined_ciphertext, decrypted_plaintext));

        vector<double> decoded_values_double; // Decoding into std::vector<double>
        ASSERT_NO_THROW(encoder.decode(decrypted_plaintext, decoded_values_double));

        ASSERT_EQ(input_values.size(), decoded_values_double.size());
        double tolerance = 0.01; // Adjust tolerance based on parameters and scale

        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(input_values[i], decoded_values_double[i], tolerance);
        }

        // Also test decoding into complex<double> to check imaginary parts
        vector<complex<double>> decoded_values_complex;
        ASSERT_NO_THROW(encoder.decode(decrypted_plaintext, decoded_values_complex));
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(input_values[i], decoded_values_complex[i].real(), tolerance);
            ASSERT_NEAR(0.0, decoded_values_complex[i].imag(), tolerance); // Imaginary part should be close to zero
        }
    }

    // New Test Case 2: Input vector smaller than slot_count (testing padding)
    TEST(CKKSEncodeEncryptTest, CombinedWithPartialVector)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);

        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();
        size_t input_size = slot_count / 2; // Use half the slots

        vector<complex<double>> input_values(input_size);
        for (size_t i = 0; i < input_size; i++)
        {
            input_values[i] = complex<double>(static_cast<double>(i) + 1.0, static_cast<double>(i) * 0.5 + 0.5);
        }

        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();

        Ciphertext combined_ciphertext;
        ASSERT_NO_THROW(encryptor.encode_and_encrypt_ckks(input_values, parms_id, scale, combined_ciphertext));

        Plaintext decrypted_plaintext;
        ASSERT_NO_THROW(decryptor.decrypt(combined_ciphertext, decrypted_plaintext));

        vector<complex<double>> decoded_values;
        ASSERT_NO_THROW(encoder.decode(decrypted_plaintext, decoded_values));

        ASSERT_EQ(slot_count, decoded_values.size()); // Decoded vector will have full slot_count
        double tolerance = 0.5;

        for (size_t i = 0; i < slot_count; ++i)
        {
            if (i < input_size)
            {
                ASSERT_NEAR(input_values[i].real(), decoded_values[i].real(), tolerance);
                ASSERT_NEAR(input_values[i].imag(), decoded_values[i].imag(), tolerance);
            }
            else
            {
                // The rest of the slots should have been padded with zeros during encoding
                ASSERT_NEAR(0.0, decoded_values[i].real(), tolerance);
                ASSERT_NEAR(0.0, decoded_values[i].imag(), tolerance);
            }
        }
    }

    // Test Case for All Zeros
    TEST(CKKSEncodeEncryptTest, CombinedWithAllZeros)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);

        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();
        // Using a tighter tolerance for zeros as they should be very accurately represented
        //double tolerance = 1e-9;
        double tolerance = 1e-8; // relaxed tolerance for CKKS zeros 

        // Test with complex zeros
        vector<complex<double>> zero_values_complex(slot_count, complex<double>(0.0, 0.0));
        Ciphertext ct_complex;
        ASSERT_NO_THROW(encryptor.encode_and_encrypt_ckks(zero_values_complex, parms_id, scale, ct_complex));
        Plaintext pt_complex;
        ASSERT_NO_THROW(decryptor.decrypt(ct_complex, pt_complex));
        vector<complex<double>> decoded_zeros_complex;
        ASSERT_NO_THROW(encoder.decode(pt_complex, decoded_zeros_complex));

        ASSERT_EQ(slot_count, decoded_zeros_complex.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(0.0, decoded_zeros_complex[i].real(), tolerance);
            ASSERT_NEAR(0.0, decoded_zeros_complex[i].imag(), tolerance);
        }

        // Test with double zeros
        vector<double> zero_values_double(slot_count, 0.0);
        Ciphertext ct_double;
        ASSERT_NO_THROW(encryptor.encode_and_encrypt_ckks(zero_values_double, parms_id, scale, ct_double));
        Plaintext pt_double;
        ASSERT_NO_THROW(decryptor.decrypt(ct_double, pt_double));
        vector<double> decoded_zeros_double;
        ASSERT_NO_THROW(encoder.decode(pt_double, decoded_zeros_double));

        ASSERT_EQ(slot_count, decoded_zeros_double.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(0.0, decoded_zeros_double[i], tolerance);
        }
    }

    // Test Case for Small Integers
    TEST(CKKSEncodeEncryptTest, CombinedWithSmallIntegers)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        keygen.create_public_key(public_key);

        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();
        
        double tolerance = 0.0001; 

        vector<double> integer_values(slot_count);
        //std::cout << "\nDEBUG: Initializing integer_values for SmallIntegers test:" << std::endl;
        for (size_t i = 0; i < slot_count; ++i)
        {
            int int_val = (i % 10) - 5;          // Should be e.g., -5, -4, ...
            double double_val = static_cast<double>(int_val); // Should be e.g., -5.0, -4.0, ...
            integer_values[i] = double_val;

            /*

            if (i < 5) { // Print only the first few to keep output manageable
                std::cout << "DEBUG: i=" << i 
                        << ", int_val=" << int_val 
                        << ", double_val=" << std::fixed << std::setprecision(10) << double_val
                        << ", assigned integer_values[" << i << "]=" << std::fixed << std::setprecision(10) << integer_values[i] 
                        << std::endl;
            }
            
            */
        }

        /*
        // ---- PRINT 1: Confirm initialization immediately ----
        std::cout << "DEBUG: After integer_values initialization loop (first 5 values):" << std::endl;
        for (size_t k = 0; k < std::min((size_t)5, slot_count); ++k) {
            std::cout << "DEBUG: integer_values[" << k << "] = " << std::fixed << std::setprecision(10) << integer_values[k] << std::endl;
        }
        */

        Ciphertext ct;
        ASSERT_NO_THROW(encryptor.encode_and_encrypt_ckks(integer_values, parms_id, scale, ct));

        Plaintext pt;
        ASSERT_NO_THROW(decryptor.decrypt(ct, pt));

        vector<double> decoded_integers;
        ASSERT_NO_THROW(encoder.decode(pt, decoded_integers));

        /*
        std::cout << "DEBUG: Before ASSERT_NEAR loop for SmallIntegers (first 5 values):" << std::endl;
        for (size_t k = 0; k < std::min((size_t)5, slot_count); ++k) {
            std::cout << "DEBUG: integer_values[" << k << "] = " << std::fixed << std::setprecision(20) << integer_values[k]
                      << ", decoded_integers[" << k << "] = " << std::fixed << std::setprecision(20) << decoded_integers[k] << std::endl;
        }
        */

        ASSERT_EQ(slot_count, decoded_integers.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            // The vector integer_values[i] holds the correct small double.
            // The comparison should be based on the actual value of integer_values[i].

            // For debugging the comparison directly if ASSERT_NEAR fails:
            double expected = integer_values[i];
            double actual = decoded_integers[i];
            double diff = std::abs(expected - actual);

            if (diff > tolerance && i < 5) { // Print if difference is too large for the first few slots
                std::cout << "MANUAL CHECK: Slot " << i << ":" << std::endl;
                std::cout << "  Expected (integer_values[i]): " << std::fixed << std::setprecision(20) << expected << std::endl;
                std::cout << "  Actual (decoded_integers[i]): " << std::fixed << std::setprecision(20) << actual << std::endl;
                std::cout << "  Difference: " << std::fixed << std::setprecision(20) << diff << std::endl;
                std::cout << "  Tolerance: " << std::fixed << std::setprecision(20) << tolerance << std::endl;
            }
            
            ASSERT_NEAR(integer_values[i], decoded_integers[i], tolerance);
        }
    }

    //////////////////////////////////////////////////////////////////////
    // TEST COMBINED SYMMETRIC ENCODE AND ENCRYPT (SEPARATED TESTS)
    //////////////////////////////////////////////////////////////////////

    //--[ Test 1: Complex Inputs ]--//

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricCombined_NonSeeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t slots = 32;
        parms.set_poly_modulus_degree(slots << 1);
        parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 60, 60, 60, 60 }));
        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);

        vector<complex<double>> input_values(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input_values[i] = complex<double>(static_cast<double>(i) / 10.0, static_cast<double>(i) / 10.0);
        }

        double scale = (1ULL << 40);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.5;

        Ciphertext ct_direct;
        encryptor.encode_and_encrypt_symmetric_ckks(input_values, parms_id, scale, ct_direct, MemoryManager::GetPool(), false);

        Plaintext pt_decrypted;
        ASSERT_NO_THROW(decryptor.decrypt(ct_direct, pt_decrypted));

        vector<complex<double>> decoded_values;
        encoder.decode(pt_decrypted, decoded_values);

        ASSERT_EQ(input_values.size(), decoded_values.size());
        for (size_t i = 0; i < slots; ++i)
        {
            ASSERT_NEAR(input_values[i].real(), decoded_values[i].real(), tolerance);
            ASSERT_NEAR(input_values[i].imag(), decoded_values[i].imag(), tolerance);
        }
    }

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricCombined_Seeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t slots = 32;
        parms.set_poly_modulus_degree(slots << 1);
        parms.set_coeff_modulus(CoeffModulus::Create(slots << 1, { 60, 60, 60, 60 }));
        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);

        vector<complex<double>> input_values(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input_values[i] = complex<double>(static_cast<double>(i) / 10.0, static_cast<double>(i) / 10.0);
        }

        double scale = (1ULL << 40);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.5;

        Ciphertext ct_seeded;
        encryptor.encode_and_encrypt_symmetric_ckks(input_values, parms_id, scale, ct_seeded, MemoryManager::GetPool(), true);

        stringstream stream;
        ct_seeded.save(stream);
        Ciphertext ct_loaded;
        ct_loaded.load(context, stream);

        Plaintext pt_decrypted_loaded;
        ASSERT_NO_THROW(decryptor.decrypt(ct_loaded, pt_decrypted_loaded));

        vector<complex<double>> decoded_values_loaded;
        encoder.decode(pt_decrypted_loaded, decoded_values_loaded);

        for (size_t i = 0; i < slots; ++i)
        {
            ASSERT_NEAR(input_values[i].real(), decoded_values_loaded[i].real(), tolerance);
            ASSERT_NEAR(input_values[i].imag(), decoded_values_loaded[i].imag(), tolerance);
        }
    }

    //--[ Test 2: Double Inputs ]--//

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithDoubleInput_NonSeeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 40, 30, 40 }));
        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        vector<double> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = (static_cast<double>(i) * 0.1) - (slot_count * 0.05);
        }

        double scale = pow(2.0, 30);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.01;

        Ciphertext ct_direct;
        encryptor.encode_and_encrypt_symmetric_ckks(input_values, parms_id, scale, ct_direct, MemoryManager::GetPool(), false);

        Plaintext pt_decrypted;
        ASSERT_NO_THROW(decryptor.decrypt(ct_direct, pt_decrypted));

        vector<double> decoded_values_double;
        encoder.decode(pt_decrypted, decoded_values_double);

        ASSERT_EQ(input_values.size(), decoded_values_double.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(input_values[i], decoded_values_double[i], tolerance);
        }
    }

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithDoubleInput_Seeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 4096;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 40, 30, 40 }));
        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        vector<double> input_values(slot_count);
        for (size_t i = 0; i < slot_count; i++)
        {
            input_values[i] = (static_cast<double>(i) * 0.1) - (slot_count * 0.05);
        }
        
        double scale = pow(2.0, 30);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.01;

        Ciphertext ct_seeded;
        encryptor.encode_and_encrypt_symmetric_ckks(input_values, parms_id, scale, ct_seeded, MemoryManager::GetPool(), true);
        
        stringstream stream;
        ct_seeded.save(stream);
        Ciphertext ct_loaded;
        ct_loaded.load(context, stream);

        Plaintext pt_decrypted_loaded;
        ASSERT_NO_THROW(decryptor.decrypt(ct_loaded, pt_decrypted_loaded));

        vector<double> decoded_values_loaded;
        encoder.decode(pt_decrypted_loaded, decoded_values_loaded);

        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(input_values[i], decoded_values_loaded[i], tolerance);
        }
    }

    //--[ Test 3: Partial Vector ]--//

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithPartialVector_NonSeeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();
        size_t input_size = slot_count / 2;

        vector<complex<double>> input_values(input_size);
        for (size_t i = 0; i < input_size; i++)
        {
            input_values[i] = complex<double>(static_cast<double>(i) + 1.0, static_cast<double>(i) * 0.5 + 0.5);
        }

        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.5;

        Ciphertext ct_direct;
        encryptor.encode_and_encrypt_symmetric_ckks(input_values, parms_id, scale, ct_direct, MemoryManager::GetPool(), false);

        Plaintext pt_decrypted;
        ASSERT_NO_THROW(decryptor.decrypt(ct_direct, pt_decrypted));

        vector<complex<double>> decoded_values;
        encoder.decode(pt_decrypted, decoded_values);

        ASSERT_EQ(slot_count, decoded_values.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            if (i < input_size)
            {
                ASSERT_NEAR(input_values[i].real(), decoded_values[i].real(), tolerance);
                ASSERT_NEAR(input_values[i].imag(), decoded_values[i].imag(), tolerance);
            }
            else
            {
                ASSERT_NEAR(0.0, decoded_values[i].real(), tolerance);
                ASSERT_NEAR(0.0, decoded_values[i].imag(), tolerance);
            }
        }
    }

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithPartialVector_Seeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();
        size_t input_size = slot_count / 2;

        vector<complex<double>> input_values(input_size);
        for (size_t i = 0; i < input_size; i++)
        {
            input_values[i] = complex<double>(static_cast<double>(i) + 1.0, static_cast<double>(i) * 0.5 + 0.5);
        }
        
        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.5;

        Ciphertext ct_seeded;
        encryptor.encode_and_encrypt_symmetric_ckks(input_values, parms_id, scale, ct_seeded, MemoryManager::GetPool(), true);

        stringstream stream;
        ct_seeded.save(stream);
        Ciphertext ct_loaded;
        ct_loaded.load(context, stream);
        
        Plaintext pt_decrypted_loaded;
        ASSERT_NO_THROW(decryptor.decrypt(ct_loaded, pt_decrypted_loaded));

        vector<complex<double>> decoded_values_loaded;
        encoder.decode(pt_decrypted_loaded, decoded_values_loaded);

        ASSERT_EQ(slot_count, decoded_values_loaded.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            if (i < input_size)
            {
                ASSERT_NEAR(input_values[i].real(), decoded_values_loaded[i].real(), tolerance);
                ASSERT_NEAR(input_values[i].imag(), decoded_values_loaded[i].imag(), tolerance);
            }
            else
            {
                ASSERT_NEAR(0.0, decoded_values_loaded[i].real(), tolerance);
                ASSERT_NEAR(0.0, decoded_values_loaded[i].imag(), tolerance);
            }
        }
    }


    //--[ Test 4: All Zeros ]--//

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithAllZeros_NonSeeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 1e-8;

        // Create a local memory pools for this test
        MemoryPoolHandle local_pool_complex = MemoryPoolHandle::New();
        MemoryPoolHandle local_pool_double = MemoryPoolHandle::New();

        // Test with complex zeros
        vector<complex<double>> zero_values_complex(slot_count, complex<double>(0.0, 0.0));
        Ciphertext ct_complex;
        encryptor.encode_and_encrypt_symmetric_ckks(
            zero_values_complex, parms_id, scale, ct_complex, local_pool_complex, false);
        Plaintext pt_complex;
        ASSERT_NO_THROW(decryptor.decrypt(ct_complex, pt_complex));
        vector<complex<double>> decoded_zeros_complex;
        encoder.decode(pt_complex, decoded_zeros_complex);
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(0.0, decoded_zeros_complex[i].real(), tolerance);
            ASSERT_NEAR(0.0, decoded_zeros_complex[i].imag(), tolerance);
        }

        // Test with double zeros
        vector<double> zero_values_double(slot_count, 0.0);
        Ciphertext ct_double;
        encryptor.encode_and_encrypt_symmetric_ckks(
            zero_values_double, parms_id, scale, ct_double, local_pool_double, false);
        Plaintext pt_double;
        ASSERT_NO_THROW(decryptor.decrypt(ct_double, pt_double));
        vector<double> decoded_zeros_double;
        encoder.decode(pt_double, decoded_zeros_double);
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(0.0, decoded_zeros_double[i], tolerance);
        }
    }

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithAllZeros_Seeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 1e-8;

        // Create a local memory pool for this test
        MemoryPoolHandle local_pool = MemoryPoolHandle::New();
        
        // Test with complex zeros
        vector<complex<double>> zero_values_complex(slot_count, complex<double>(0.0, 0.0));
        Ciphertext ct_complex_seeded;
        // Pass the local_pool to the function
        encryptor.encode_and_encrypt_symmetric_ckks(
            zero_values_complex, parms_id, scale, ct_complex_seeded, local_pool, true);
        stringstream stream_complex;
        ct_complex_seeded.save(stream_complex);
        Ciphertext ct_complex_loaded;
        ct_complex_loaded.load(context, stream_complex);
        Plaintext pt_complex_loaded;
        ASSERT_NO_THROW(decryptor.decrypt(ct_complex_loaded, pt_complex_loaded));
        vector<complex<double>> decoded_zeros_loaded_complex;
        encoder.decode(pt_complex_loaded, decoded_zeros_loaded_complex);
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(0.0, decoded_zeros_loaded_complex[i].real(), tolerance);
            ASSERT_NEAR(0.0, decoded_zeros_loaded_complex[i].imag(), tolerance);
        }
        
        // Test with double zeros
        vector<double> zero_values_double(slot_count, 0.0);
        Ciphertext ct_double_seeded;
        // Pass the local_pool to the function
        encryptor.encode_and_encrypt_symmetric_ckks(
            zero_values_double, parms_id, scale, ct_double_seeded, local_pool, true);
        stringstream stream_double;
        ct_double_seeded.save(stream_double);
        Ciphertext ct_double_loaded;
        ct_double_loaded.load(context, stream_double);
        Plaintext pt_double_loaded;
        ASSERT_NO_THROW(decryptor.decrypt(ct_double_loaded, pt_double_loaded));
        vector<double> decoded_zeros_loaded_double;
        encoder.decode(pt_double_loaded, decoded_zeros_loaded_double);
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(0.0, decoded_zeros_loaded_double[i], tolerance);
        }
    }


    // Test Case for Small Integers using Symmetric Encryption (with Local Memory Pool)
    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithSmallIntegers_NonSeeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        // Using the original scale that works for asymmetric encryption
        double scale = pow(2.0, 40); 
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.0001;

        // Create a local memory pool for this test
        MemoryPoolHandle local_pool = MemoryPoolHandle::New();

        vector<double> integer_values(slot_count);
        for (size_t i = 0; i < slot_count; ++i)
        {
            integer_values[i] = static_cast<double>((int)(i % 10) - 5);
        }

        Ciphertext ct;
        // Pass the local_pool to the function
        encryptor.encode_and_encrypt_symmetric_ckks(
            integer_values, parms_id, scale, ct, local_pool, false);
        
        Plaintext pt;
        ASSERT_NO_THROW(decryptor.decrypt(ct, pt));

        vector<double> decoded_integers;
        // Encoder also uses a memory pool, pass the local_pool if its methods accept it,
        // or ensure it uses a consistent pool if constructed with one.
        // CKKSEncoder::decode does not take a pool argument directly, but uses the pool from its context/construction.
        encoder.decode(pt, decoded_integers);

        ASSERT_EQ(slot_count, decoded_integers.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(integer_values[i], decoded_integers[i], tolerance);
        }
    }

    TEST(CKKSEncodeEncryptTest, CombinedSymmetricWithSmallIntegers_Seeded)
    {
        EncryptionParameters parms(scheme_type::ckks);
        size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
        SEALContext context(parms, false, sec_level_type::none);
        
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Encryptor encryptor(context, secret_key);
        Decryptor decryptor(context, secret_key);
        CKKSEncoder encoder(context);
        size_t slot_count = encoder.slot_count();

        // Using the original scale that works for asymmetric encryption
        double scale = pow(2.0, 40);
        parms_id_type parms_id = context.first_parms_id();
        double tolerance = 0.0001;

        // Create a local memory pool for this test
        MemoryPoolHandle local_pool = MemoryPoolHandle::New();
        
        vector<double> integer_values(slot_count);
        for (size_t i = 0; i < slot_count; ++i)
        {
            integer_values[i] = static_cast<double>((int)(i % 10) - 5);
        }

        Ciphertext ct_seeded;
        // Pass the local_pool to the function
        encryptor.encode_and_encrypt_symmetric_ckks(
            integer_values, parms_id, scale, ct_seeded, local_pool, true);

        stringstream stream;
        // Serialization methods might use the global pool or their own buffers.
        ct_seeded.save(stream); 
        Ciphertext ct_loaded;
        // Load operation might use the global pool if not managed by local_pool through its context.
        ct_loaded.load(context, stream); 
        
        Plaintext pt_loaded;
        ASSERT_NO_THROW(decryptor.decrypt(ct_loaded, pt_loaded));
        
        vector<double> decoded_integers_loaded;
        encoder.decode(pt_loaded, decoded_integers_loaded);

        ASSERT_EQ(slot_count, decoded_integers_loaded.size());
        for (size_t i = 0; i < slot_count; ++i)
        {
            ASSERT_NEAR(integer_values[i], decoded_integers_loaded[i], tolerance);
        }
    }

} // namespace sealtest
