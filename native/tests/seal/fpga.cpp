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

#include "../../src/seal/fpga/Inc/fpga_ckks_context.h"
#include "../../src/seal/fpga/Inc/fpga_ckks_encoder.h"
#include "../../src/seal/fpga/Inc/fpga_dwt.h"
#include "../../src/seal/fpga/Inc/fpga_ntt.h"
#include "../../src/seal/fpga/Inc/fpga_encrypt.h"
#include "../../src/seal/fpga/Inc/fpga_ckks_encryptor.h"

using namespace seal;
using namespace seal::fpga;
using namespace std;

namespace sealtest
{
    class FPGAEncoderTest : public ::testing::TestWithParam<size_t>
    {
    protected:
        void SetUp() override
        {
            poly_modulus_degree_ = GetParam();
            log_poly_modulus_degree_ = 0;
            size_t n = poly_modulus_degree_;
            while (n > 1)
            {
                n >>= 1;
                log_poly_modulus_degree_++;
            }
        }

        size_t poly_modulus_degree_;
        int log_poly_modulus_degree_;
    };

    TEST_P(FPGAEncoderTest, IndexMapMatchesSEAL)
    {
        FPGACKKSEncoder fpga_encoder(poly_modulus_degree_);

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(poly_modulus_degree_);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree_, { 40, 40 }));
        SEALContext context(parms, false, sec_level_type::none);
        CKKSEncoder seal_encoder(context);

        size_t slots = poly_modulus_degree_ / 2;
        vector<complex<double>> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = complex<double>(static_cast<double>(i), static_cast<double>(i * 2));
        }

        vector<complex<double>> fpga_prepared;
        fpga_encoder.prepare_for_fpga(input, fpga_prepared);

        ASSERT_EQ(fpga_prepared.size(), poly_modulus_degree_);

        for (size_t i = 0; i < slots; i++)
        {
            size_t idx = fpga_encoder.index_map()[i];
            EXPECT_NEAR(fpga_prepared[idx].real(), input[i].real(), 1e-10)
                << "Mismatch at slot " << i << " index " << idx;
            EXPECT_NEAR(fpga_prepared[idx].imag(), input[i].imag(), 1e-10)
                << "Mismatch at slot " << i << " index " << idx;

            size_t conj_idx = fpga_encoder.index_map()[i + slots];
            EXPECT_NEAR(fpga_prepared[conj_idx].real(), input[i].real(), 1e-10)
                << "Conjugate mismatch at slot " << i;
            EXPECT_NEAR(fpga_prepared[conj_idx].imag(), -input[i].imag(), 1e-10)
                << "Conjugate mismatch at slot " << i;
        }
    }

    TEST_P(FPGAEncoderTest, DWTInverseProducesRealCoefficients)
    {
        size_t n = poly_modulus_degree_;
        int log_n = log_poly_modulus_degree_;
        size_t slots = n / 2;

        default_random_engine rng(42);
        uniform_real_distribution<double> dist(-100.0, 100.0);

        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = dist(rng);
        }

        FPGACKKSEncoder fpga_encoder(n);
        vector<complex<double>> fpga_prepared;
        fpga_encoder.prepare_for_fpga(input, fpga_prepared);

        vector<complex<double>> inv_roots(n);
        double m = static_cast<double>(n << 1);
        double angle_base = 2.0 * M_PI / m;

        auto reverse_bits = [](size_t value, int bit_count) {
            size_t result = 0;
            for (int i = 0; i < bit_count; i++)
            {
                result = (result << 1) | (value & 1);
                value >>= 1;
            }
            return result;
        };

        for (size_t i = 1; i < n; i++)
        {
            size_t inv_idx = reverse_bits(i - 1, log_n) + 1;
            double inv_angle = -angle_base * static_cast<double>(inv_idx);
            inv_roots[i] = complex<double>(cos(inv_angle), sin(inv_angle));
        }

        double scale = pow(2.0, 40);
        double scale_factor = scale / static_cast<double>(n);
        dwt_inverse_host(fpga_prepared.data(), n, log_n, inv_roots.data(), scale_factor);

        for (size_t i = 0; i < n; i++)
        {
            EXPECT_NEAR(fpga_prepared[i].imag(), 0.0, 1.0)
                << "DWT of conjugate-symmetric input should produce real coefficients at index " << i;
        }
    }

    TEST_P(FPGAEncoderTest, CBDSamplerDistribution)
    {
        FPGACBDSampler sampler(12345);
        size_t count = 10000;
        vector<int64_t> samples;
        sampler.sample(count, samples);

        double mean = 0.0;
        for (auto s : samples)
        {
            mean += static_cast<double>(s);
        }
        mean /= static_cast<double>(count);

        EXPECT_NEAR(mean, 0.0, 0.5) << "CBD mean should be close to 0";

        double variance = 0.0;
        for (auto s : samples)
        {
            double diff = static_cast<double>(s) - mean;
            variance += diff * diff;
        }
        variance /= static_cast<double>(count);
        double stddev = sqrt(variance);

        EXPECT_NEAR(stddev, 3.2, 0.5) << "CBD stddev should be close to 3.2";

        int64_t min_val = *min_element(samples.begin(), samples.end());
        int64_t max_val = *max_element(samples.begin(), samples.end());

        EXPECT_GE(min_val, -21) << "CBD min should be >= -21";
        EXPECT_LE(max_val, 21) << "CBD max should be <= 21";
    }

    TEST_P(FPGAEncoderTest, UniformSamplerDistribution)
    {
        FPGAUniformSampler sampler(54321);
        uint64_t modulus = 1099511627689ULL;
        size_t count = 10000;
        vector<uint64_t> samples;
        sampler.sample(count, modulus, samples);

        for (auto s : samples)
        {
            EXPECT_LT(s, modulus) << "Sample should be less than modulus";
        }

        double mean = 0.0;
        for (auto s : samples)
        {
            mean += static_cast<double>(s);
        }
        mean /= static_cast<double>(count);

        double expected_mean = static_cast<double>(modulus) / 2.0;
        double tolerance = expected_mean * 0.1;
        EXPECT_NEAR(mean, expected_mean, tolerance)
            << "Uniform mean should be approximately modulus/2";
    }

    INSTANTIATE_TEST_SUITE_P(
        PolyDegrees,
        FPGAEncoderTest,
        ::testing::Values(4096, 8192, 16384, 32768),
        [](const ::testing::TestParamInfo<size_t>& info) {
            return "N" + to_string(info.param);
        });

#ifdef SEAL_USE_FPGA

    class FPGADeviceTest : public ::testing::TestWithParam<size_t>
    {
    protected:
        void SetUp() override
        {
            poly_modulus_degree_ = GetParam();
        }

        size_t poly_modulus_degree_;
    };

    TEST_P(FPGADeviceTest, ContextInitialization)
    {
        FPGACKKSParams params;
        params.poly_modulus_degree = poly_modulus_degree_;

        size_t n = poly_modulus_degree_;
        int log_n = 0;
        while (n > 1)
        {
            n >>= 1;
            log_n++;
        }
        params.log_poly_modulus_degree = log_n;
        params.modulus = 1099511627689ULL;
        params.scale = pow(2.0, 40);

        EXPECT_NO_THROW({
            FPGACKKSContext context(params, true);

            EXPECT_EQ(context.slot_count(), poly_modulus_degree_ / 2);
            EXPECT_NE(context.dwt_root_powers(), nullptr);
            EXPECT_NE(context.dwt_inv_root_powers(), nullptr);
            EXPECT_NE(context.ntt_root_powers(), nullptr);
            EXPECT_NE(context.matrix_reps_index_map(), nullptr);
        });
    }

    INSTANTIATE_TEST_SUITE_P(
        PolyDegrees,
        FPGADeviceTest,
        ::testing::Values(4096, 8192),
        [](const ::testing::TestParamInfo<size_t>& info) {
            return "N" + to_string(info.param);
        });

#endif

    class FPGAPipelineTest : public ::testing::TestWithParam<size_t>
    {
    protected:
        void SetUp() override
        {
            poly_modulus_degree_ = GetParam();
            log_poly_modulus_degree_ = 0;
            size_t n = poly_modulus_degree_;
            while (n > 1)
            {
                n >>= 1;
                log_poly_modulus_degree_++;
            }
        }

        size_t poly_modulus_degree_;
        int log_poly_modulus_degree_;
    };

    TEST_P(FPGAPipelineTest, FullPipelineHostMatchesSEAL)
    {
        size_t n = poly_modulus_degree_;
        int log_n = log_poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));

        SEALContext context(parms, false, sec_level_type::none);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);
        auto& context_data = *context.first_context_data();
        auto& coeff_modulus = context_data.parms().coeff_modulus();
        uint64_t modulus = coeff_modulus[0].value();

        auto ntt_tables = context_data.small_ntt_tables();
        vector<uint64_t> ntt_root_powers(n);
        for (size_t i = 0; i < n; i++)
        {
            ntt_root_powers[i] = ntt_tables->get_from_root_powers(i).operand;
        }
        uint64_t inv_n = ntt_tables->inv_degree_modulo().operand;

        vector<uint64_t> secret_key_ntt(n);
        const uint64_t* sk_data = secret_key.data().data();
        for (size_t i = 0; i < n; i++)
        {
            secret_key_ntt[i] = sk_data[i];
        }

        FPGACKKSParams fpga_params;
        fpga_params.poly_modulus_degree = n;
        fpga_params.log_poly_modulus_degree = log_n;
        fpga_params.modulus = modulus;
        fpga_params.scale = scale;
        fpga_params.compute_derived_constants();

        FPGACKKSEncryptor fpga_encryptor(fpga_params, secret_key_ntt.data(), ntt_root_powers.data(), inv_n);

        default_random_engine rng(42);
        uniform_real_distribution<double> dist(-10.0, 10.0);
        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = dist(rng);
        }

        vector<uint64_t> c0_fpga, c1_fpga;
        uint64_t fixed_seed = 123456789ULL;
        fpga_encryptor.encrypt_host(input, c0_fpga, c1_fpga, fixed_seed);

        ASSERT_EQ(c0_fpga.size(), n);
        ASSERT_EQ(c1_fpga.size(), n);

        Ciphertext ciphertext;
        ciphertext.resize(context, context.first_parms_id(), 2);
        ciphertext.is_ntt_form() = true;
        ciphertext.scale() = scale;

        uint64_t* ct_data_0 = ciphertext.data(0);
        uint64_t* ct_data_1 = ciphertext.data(1);
        for (size_t i = 0; i < n; i++)
        {
            ct_data_0[i] = c0_fpga[i];
            ct_data_1[i] = c1_fpga[i];
        }

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double max_error = 0.0;
        for (size_t i = 0; i < slots; i++)
        {
            double error = abs(decoded[i] - input[i]);
            max_error = max(max_error, error);
        }

        double tolerance = 0.01;
        EXPECT_LT(max_error, tolerance)
            << "Max decryption error " << max_error << " exceeds tolerance " << tolerance;
    }

    TEST_P(FPGAPipelineTest, ScaleAndReduceCorrectness)
    {
        size_t n = poly_modulus_degree_;

        uint64_t modulus = 1099511627689ULL;
        uint64_t barrett_ratio[2];
        compute_barrett_ratio(modulus, barrett_ratio);

        default_random_engine rng(12345);
        uniform_real_distribution<double> dist(-1e10, 1e10);

        vector<complex<double>> input(n);
        for (size_t i = 0; i < n; i++)
        {
            input[i] = complex<double>(dist(rng), 0.0);
        }

        vector<uint64_t> output(n);
        scale_and_reduce_host(input.data(), output.data(), n, modulus, barrett_ratio);

        for (size_t i = 0; i < n; i++)
        {
            EXPECT_LT(output[i], modulus) << "Output at index " << i << " should be less than modulus";

            double rounded = round(input[i].real());
            int64_t expected;
            if (rounded >= 0)
            {
                expected = static_cast<int64_t>(fmod(rounded, static_cast<double>(modulus)));
            }
            else
            {
                double temp = fmod(-rounded, static_cast<double>(modulus));
                expected = (temp == 0) ? 0 : static_cast<int64_t>(modulus - temp);
            }

            EXPECT_EQ(output[i], static_cast<uint64_t>(expected))
                << "Mismatch at index " << i << " for input " << input[i].real();
        }
    }

    TEST_P(FPGAPipelineTest, EncryptFormulaCorrectness)
    {
        size_t n = poly_modulus_degree_;

        uint64_t modulus = 1099511627689ULL;

        default_random_engine rng(99999);
        uniform_int_distribution<uint64_t> dist(0, modulus - 1);

        vector<uint64_t> plaintext_ntt(n);
        vector<uint64_t> secret_key_ntt(n);
        vector<uint64_t> uniform_poly_ntt(n);
        for (size_t i = 0; i < n; i++)
        {
            plaintext_ntt[i] = dist(rng);
            secret_key_ntt[i] = dist(rng);
            uniform_poly_ntt[i] = dist(rng);
        }

        vector<uint64_t> c0(n), c1(n);

        for (size_t i = 0; i < n; i++)
        {
            __uint128_t product = static_cast<__uint128_t>(uniform_poly_ntt[i]) * secret_key_ntt[i];
            uint64_t as = static_cast<uint64_t>(product % modulus);

            uint64_t neg_as = (as == 0) ? 0 : modulus - as;

            uint64_t c0_val = neg_as + plaintext_ntt[i];
            if (c0_val >= modulus) c0_val -= modulus;

            c0[i] = c0_val;
            c1[i] = uniform_poly_ntt[i];
        }

        for (size_t i = 0; i < n; i++)
        {
            EXPECT_EQ(c1[i], uniform_poly_ntt[i]) << "c1 should equal uniform_poly at index " << i;
        }

        vector<uint64_t> decrypted_ntt(n);
        for (size_t i = 0; i < n; i++)
        {
            __uint128_t product = static_cast<__uint128_t>(c1[i]) * secret_key_ntt[i];
            uint64_t cs = static_cast<uint64_t>(product % modulus);

            uint64_t sum = c0[i] + cs;
            if (sum >= modulus) sum -= modulus;
            decrypted_ntt[i] = sum;
        }

        for (size_t i = 0; i < n; i++)
        {
            EXPECT_EQ(decrypted_ntt[i], plaintext_ntt[i])
                << "Decryption formula verification failed at index " << i;
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        PolyDegrees,
        FPGAPipelineTest,
        ::testing::Values(4096, 8192, 16384, 32768),
        [](const ::testing::TestParamInfo<size_t>& info) {
            return "N" + to_string(info.param);
        });

    class FPGACKKSCompatibilityTest : public ::testing::TestWithParam<size_t>
    {
    protected:
        void SetUp() override
        {
            poly_modulus_degree_ = GetParam();
            log_poly_modulus_degree_ = 0;
            size_t n = poly_modulus_degree_;
            while (n > 1)
            {
                n >>= 1;
                log_poly_modulus_degree_++;
            }
        }

        void SetupFPGAEncryptor(
            const SEALContext& context,
            const SecretKey& secret_key,
            double scale,
            FPGACKKSParams& fpga_params,
            unique_ptr<FPGACKKSEncryptor>& fpga_encryptor)
        {
            size_t n = poly_modulus_degree_;
            int log_n = log_poly_modulus_degree_;

            auto& context_data = *context.first_context_data();
            auto& coeff_modulus = context_data.parms().coeff_modulus();
            uint64_t modulus = coeff_modulus[0].value();

            auto ntt_tables = context_data.small_ntt_tables();
            vector<uint64_t> ntt_root_powers(n);
            for (size_t i = 0; i < n; i++)
            {
                ntt_root_powers[i] = ntt_tables->get_from_root_powers(i).operand;
            }
            uint64_t inv_n = ntt_tables->inv_degree_modulo().operand;

            vector<uint64_t> secret_key_ntt(n);
            const uint64_t* sk_data = secret_key.data().data();
            for (size_t i = 0; i < n; i++)
            {
                secret_key_ntt[i] = sk_data[i];
            }

            fpga_params.poly_modulus_degree = n;
            fpga_params.log_poly_modulus_degree = log_n;
            fpga_params.modulus = modulus;
            fpga_params.scale = scale;
            fpga_params.compute_derived_constants();

            fpga_encryptor = make_unique<FPGACKKSEncryptor>(
                fpga_params, secret_key_ntt.data(), ntt_root_powers.data(), inv_n);
        }

        void EncryptWithFPGA(
            const FPGACKKSEncryptor& fpga_encryptor,
            const vector<double>& input,
            const SEALContext& context,
            double scale,
            Ciphertext& ciphertext)
        {
            size_t n = poly_modulus_degree_;
            vector<uint64_t> c0, c1;
            fpga_encryptor.encrypt_host(input, c0, c1);

            ciphertext.resize(context, context.first_parms_id(), 2);
            ciphertext.is_ntt_form() = true;
            ciphertext.scale() = scale;

            uint64_t* ct_data_0 = ciphertext.data(0);
            uint64_t* ct_data_1 = ciphertext.data(1);
            for (size_t i = 0; i < n; i++)
            {
                ct_data_0[i] = c0[i];
                ct_data_1[i] = c1[i];
            }
        }

        void EncryptWithFPGA(
            const FPGACKKSEncryptor& fpga_encryptor,
            const vector<complex<double>>& input,
            const SEALContext& context,
            double scale,
            Ciphertext& ciphertext)
        {
            size_t n = poly_modulus_degree_;
            vector<uint64_t> c0, c1;
            fpga_encryptor.encrypt_host(input, c0, c1);

            ciphertext.resize(context, context.first_parms_id(), 2);
            ciphertext.is_ntt_form() = true;
            ciphertext.scale() = scale;

            uint64_t* ct_data_0 = ciphertext.data(0);
            uint64_t* ct_data_1 = ciphertext.data(1);
            for (size_t i = 0; i < n; i++)
            {
                ct_data_0[i] = c0[i];
                ct_data_1[i] = c1[i];
            }
        }

        size_t poly_modulus_degree_;
        int log_poly_modulus_degree_;
    };

    TEST_P(FPGACKKSCompatibilityTest, ComplexVectorEncodeDecode)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<complex<double>> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = complex<double>(
                static_cast<double>(i) / 10.0,
                static_cast<double>(i) / 20.0);
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<complex<double>> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.01;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(input[i].real(), decoded[i].real(), tolerance)
                << "Real part mismatch at slot " << i;
            EXPECT_NEAR(input[i].imag(), decoded[i].imag(), tolerance)
                << "Imaginary part mismatch at slot " << i;
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, DoubleVectorEncodeDecode)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = (static_cast<double>(i) * 0.1) - (slots * 0.05);
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.01;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(input[i], decoded[i], tolerance)
                << "Mismatch at slot " << i;
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, PartialVectorWithPadding)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;
        size_t input_size = slots / 4;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<complex<double>> input(input_size);
        for (size_t i = 0; i < input_size; i++)
        {
            input[i] = complex<double>(
                static_cast<double>(i) + 1.0,
                static_cast<double>(i) * 0.5 + 0.5);
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<complex<double>> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.5;
        ASSERT_EQ(slots, decoded.size());

        for (size_t i = 0; i < slots; i++)
        {
            if (i < input_size)
            {
                EXPECT_NEAR(input[i].real(), decoded[i].real(), tolerance)
                    << "Real part mismatch at slot " << i;
                EXPECT_NEAR(input[i].imag(), decoded[i].imag(), tolerance)
                    << "Imaginary part mismatch at slot " << i;
            }
            else
            {
                EXPECT_NEAR(0.0, decoded[i].real(), tolerance)
                    << "Padded real part should be zero at slot " << i;
                EXPECT_NEAR(0.0, decoded[i].imag(), tolerance)
                    << "Padded imaginary part should be zero at slot " << i;
            }
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, AllZeros)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<double> input(slots, 0.0);

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 1e-6;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(0.0, decoded[i], tolerance)
                << "Zero input should decode to zero at slot " << i;
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, SmallIntegers)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = static_cast<double>(static_cast<int>(i % 10) - 5);
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.001;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(input[i], decoded[i], tolerance)
                << "Mismatch at slot " << i;
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, LargeRandomValues)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        default_random_engine rng(42);
        uniform_real_distribution<double> dist(-1000.0, 1000.0);

        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = dist(rng);
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.01;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(input[i], decoded[i], tolerance)
                << "Mismatch at slot " << i << " (input: " << input[i] << ", decoded: " << decoded[i] << ")";
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, NegativeValues)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = -static_cast<double>(i + 1);
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.01;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(input[i], decoded[i], tolerance)
                << "Mismatch at slot " << i;
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, MixedPositiveNegative)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = (i % 2 == 0) ? static_cast<double>(i) : -static_cast<double>(i);
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.01;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(input[i], decoded[i], tolerance)
                << "Mismatch at slot " << i;
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, SingleSlotUsed)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<double> input = { 42.5 };

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 0.01;
        EXPECT_NEAR(input[0], decoded[0], tolerance) << "Single value mismatch";

        for (size_t i = 1; i < slots; i++)
        {
            EXPECT_NEAR(0.0, decoded[i], tolerance)
                << "Padded slot should be zero at " << i;
        }
    }

    TEST_P(FPGACKKSCompatibilityTest, VerySmallValues)
    {
        size_t n = poly_modulus_degree_;
        size_t slots = n / 2;

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(n);
        parms.set_coeff_modulus(CoeffModulus::Create(n, { 60 }));
        SEALContext context(parms, false, sec_level_type::none);

        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        Decryptor decryptor(context, secret_key);
        CKKSEncoder seal_encoder(context);

        double scale = pow(2.0, 40);

        FPGACKKSParams fpga_params;
        unique_ptr<FPGACKKSEncryptor> fpga_encryptor;
        SetupFPGAEncryptor(context, secret_key, scale, fpga_params, fpga_encryptor);

        vector<double> input(slots);
        for (size_t i = 0; i < slots; i++)
        {
            input[i] = static_cast<double>(i + 1) * 1e-6;
        }

        Ciphertext ciphertext;
        EncryptWithFPGA(*fpga_encryptor, input, context, scale, ciphertext);

        Plaintext decrypted;
        decryptor.decrypt(ciphertext, decrypted);

        vector<double> decoded;
        seal_encoder.decode(decrypted, decoded);

        double tolerance = 1e-6;
        for (size_t i = 0; i < slots; i++)
        {
            EXPECT_NEAR(input[i], decoded[i], tolerance)
                << "Mismatch at slot " << i;
        }
    }

    INSTANTIATE_TEST_SUITE_P(
        PolyDegrees,
        FPGACKKSCompatibilityTest,
        ::testing::Values(4096, 8192, 16384, 32768),
        [](const ::testing::TestParamInfo<size_t>& info) {
            return "N" + to_string(info.param);
        });

} // namespace sealtest
