// FPGA test harness

#include "gtest/gtest.h"
#include "seal/seal.h"          // Main SEAL header
#include "seal/fpga/fpga.h"     // FPGA header
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

} // namespace sealtest
