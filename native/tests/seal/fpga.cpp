// FPGA test harness

#include "gtest/gtest.h"
#include "seal/seal.h"          // Main SEAL header
#include "seal/fpga/fpga.h"     // FPGA header
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision

// Use the sealtest namespace, common in other SEAL tests
namespace sealtest
{
    TEST(FPGATests, ProcessVectorDummyTest)
    {
        // Test with a vector of complex numbers
        std::vector<std::complex<double>> input_vector;
        input_vector.push_back({1.0, 2.0});
        input_vector.push_back({-3.5, 4.0});
        input_vector.push_back({0.0, -5.75});
        input_vector.push_back({100.25, 0.0});

        std::vector<std::complex<double>> expected_vector;
        for (const auto& val : input_vector)
        {
            // The dummy kernel adds 1.0 to the real part
            expected_vector.push_back({val.real() + 1.0, val.imag()});
        }

        std::vector<std::complex<double>> result_vector;
        ASSERT_NO_THROW(result_vector = seal::fpga::process_vector_fpga_dummy(input_vector));

        ASSERT_EQ(input_vector.size(), result_vector.size()) << "Output vector size does not match input vector size.";

        double tolerance = 1e-9; // A small tolerance for floating-point comparisons

        for (size_t i = 0; i < result_vector.size(); ++i)
        {
            ASSERT_NEAR(expected_vector[i].real(), result_vector[i].real(), tolerance)
                << "Real part mismatch at index " << i;
            ASSERT_NEAR(expected_vector[i].imag(), result_vector[i].imag(), tolerance)
                << "Imaginary part mismatch at index " << i;
        }

        // Test with an empty vector
        std::vector<std::complex<double>> empty_input_vector;
        std::vector<std::complex<double>> empty_result_vector;
        ASSERT_NO_THROW(empty_result_vector = seal::fpga::process_vector_fpga_dummy(empty_input_vector));
        ASSERT_TRUE(empty_result_vector.empty()) << "Processing an empty vector should result in an empty vector.";

        // Test with a vector containing a single element
        std::vector<std::complex<double>> single_input_vector = {{7.7, -8.8}};
        std::vector<std::complex<double>> single_expected_vector = {{7.7 + 1.0, -8.8}};
        std::vector<std::complex<double>> single_result_vector;
        ASSERT_NO_THROW(single_result_vector = seal::fpga::process_vector_fpga_dummy(single_input_vector));
        ASSERT_EQ(single_input_vector.size(), single_result_vector.size());
        ASSERT_NEAR(single_expected_vector[0].real(), single_result_vector[0].real(), tolerance);
        ASSERT_NEAR(single_expected_vector[0].imag(), single_result_vector[0].imag(), tolerance);
    }

    TEST(FPGATests, FPGAEncodeDecodeTestDouble)
    {
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        std::size_t poly_modulus_degree = 4096; // Using a smaller N for faster testing initially
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, { 40, 20, 40 })); // Example moduli

        seal::SEALContext context(parms, false, seal::sec_level_type::none); // expand_mod_chain = false for single parms_id
        
        seal::CKKSEncoder native_encoder(context); // Native SEAL encoder for decoding

        double scale = std::pow(2.0, 20); // Example scale

        std::vector<double> input_values;
        // Fill with some values, up to slots = N/2
        std::size_t slot_count = native_encoder.slot_count();
        ASSERT_EQ(slot_count, poly_modulus_degree / 2);

        for (size_t i = 0; i < slot_count; ++i) {
            input_values.push_back(static_cast<double>(i % 10) * 0.1 - 0.5); // Simple pattern: -0.5, -0.4, ..., 0.4
        }
        // If input_values.size() < slot_count, CKKS encoding pads with zeros.
        // For this test, let's use a full vector or a specific smaller size.
        // Let's test with a smaller vector than full slots to check padding behavior implicitly.
        if (slot_count > 4) { // Ensure we don't make it empty if slot_count is too small
            input_values.resize(std::min(slot_count, static_cast<std::size_t>(8))); // Test with 8 values or slot_count if smaller
        }


        seal::Plaintext fpga_encoded_pt;
        ASSERT_NO_THROW(seal::fpga::encode_ckks_fpga<double>(
            context, 
            input_values.data(), 
            input_values.size(), 
            context.first_parms_id(), 
            scale, 
            fpga_encoded_pt, 
            seal::MemoryPoolHandle::Global()
        ));

        // Decode the FPGA-encoded plaintext using the native CKKSEncoder
        std::vector<double> decoded_values;
        ASSERT_NO_THROW(native_encoder.decode(fpga_encoded_pt, decoded_values));
        
        // Check the size of the decoded vector
        // The native decoder will output a vector of size slot_count
        ASSERT_EQ(decoded_values.size(), slot_count);

        // Compare decoded values with original input values
        // CKKS is an approximate scheme, so use a tolerance.
        // The tolerance might need adjustment based on parameters and scale.
        double tolerance = 1e-5; 

        std::cout << std::fixed << std::setprecision(7);
        std::cout << "FPGAEncodeDecodeTestDouble: Comparing first " << input_values.size() << " decoded values (out of " << slot_count << " slots)." << std::endl;

        for (size_t i = 0; i < input_values.size(); ++i) {
            // std::cout << "Original[" << i << "]: " << input_values[i] 
            //           << ", Decoded[" << i << "]: " << decoded_values[i] 
            //           << ", Diff: " << std::abs(input_values[i] - decoded_values[i]) << std::endl;
            ASSERT_NEAR(input_values[i], decoded_values[i], tolerance)
                << "Mismatch at index " << i << ". Original: " << input_values[i]
                << ", Decoded: " << decoded_values[i];
        }
        
        // For the remaining slots (if input_values.size() < slot_count), CKKS encodes them as approximately zero.
        for (size_t i = input_values.size(); i < slot_count; ++i) {
            // std::cout << "Decoded (padded)[" << i << "]: " << decoded_values[i] << std::endl;
            ASSERT_NEAR(0.0, decoded_values[i], tolerance)
                << "Mismatch in padded zero at index " << i << ". Decoded: " << decoded_values[i];
        }
    }

    // TODO: Add FPGAEncodeDecodeTestComplex for std::complex<double> inputs.
    // TODO: Add tests with different poly_modulus_degree and coefficient moduli.

} // namespace sealtest
