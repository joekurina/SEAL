// native/tests/seal/fpga.cpp

#include "gtest/gtest.h"
#include "seal/seal.h"          // Main SEAL header
#include "seal/fpga/fpga_encoder.h"     // Your FPGAEncoder header
#include "seal/encryptionparams.h"
#include "seal/modulus.h"
#include "seal/context.h"
#include "seal/util/rns.h" 
#include "seal/util/uintarithsmallmod.h" 

#include <vector>
#include <complex>
#include <iostream>
#include <iomanip> // For std::fixed and std::setprecision
#include <cmath>   // For M_PI, cos, sin, round, std::fabs
#include <algorithm> // For std::generate
#include <random>    // For random number generation


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
    TEST(FPGAEncoderTests, EncodeDecodeAccuracyTest)
    {
        // 1. Set up CKKS parameters
        seal::EncryptionParameters parms(seal::scheme_type::ckks);
        
        std::size_t poly_modulus_degree = 8192; // Example value, must be a power of 2
        parms.set_poly_modulus_degree(poly_modulus_degree);
        
        // Choose CoeffModulus that supports the desired scale and data size
        // For CKKS, these are primes. Example values:
        parms.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

        // 2. Create SEALContext
        auto context = seal::SEALContext(parms);

        // 3. Create FPGAEncoder
    }

} // namespace sealtest
