// FPGA test harness

#include "gtest/gtest.h"
#include "seal/seal.h"          // Main SEAL header
#include "seal/fpga/fpga.h"     // FPGA header
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For M_PI, cos, sin, round
#include <algorithm> // For std::generate

// CKKS specific headers
#include "seal/context.h"
#include "seal/encryptionparams.h"
#include "seal/ckks.h"
#include "seal/modulus.h"
#include "seal/util/rns.h" 
#include "seal/util/uintarithsmallmod.h" 


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to print complex vectors for debugging
void print_complex_vector(const std::string& label, const std::vector<std::complex<double>>& vec, int precision = 5) {
    std::cout << label << " (size " << vec.size() << "): [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(precision) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}


// Use the sealtest namespace, common in other SEAL tests
namespace sealtest
{
    /*
    // Dummy test remains the same
    TEST(FPGATests, ProcessVectorDummyTest)
    {
        std::vector<std::complex<double>> input_vector = {{1.0, 2.0}, {-3.5, 4.0}, {0.0, -5.75}, {100.25, 0.0}};
        std::vector<std::complex<double>> expected_vector;
        for (const auto& val : input_vector) {
            expected_vector.push_back({val.real() + 1.0, val.imag()});
        }
        std::vector<std::complex<double>> result_vector;
        ASSERT_NO_THROW(result_vector = seal::fpga::process_vector_fpga_dummy(input_vector));
        ASSERT_EQ(input_vector.size(), result_vector.size());
        double tolerance = 1e-9;
        for (size_t i = 0; i < result_vector.size(); ++i) {
            ASSERT_NEAR(expected_vector[i].real(), result_vector[i].real(), tolerance);
            ASSERT_NEAR(expected_vector[i].imag(), result_vector[i].imag(), tolerance);
        }
    }
    */
   


} // namespace sealtest
