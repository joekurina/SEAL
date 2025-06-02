#ifndef SEAL_FPGA_H
#define SEAL_FPGA_H

#include <vector>
#include <complex>
#include "seal/plaintext.h"         // For seal::Plaintext
#include "seal/context.h"           // For seal::SEALContext
#include "seal/memorymanager.h"     // For seal::MemoryPoolHandle
#include "seal/util/defines.h"      // For seal::util::parms_id_type (used by seal::parms_id_type)
#include "seal/encryptionparams.h"  // For seal::parms_id_type definition

namespace seal
{
    namespace fpga
    {
        /**
         * @brief Processes a vector of complex numbers using a SYCL kernel, adding 1.0 to the real part of each element.
         *
         * @param input_vector The input vector of complex numbers.
         * @return std::vector<std::complex<double>> The processed vector with 1.0 added to the real part of each element.
         * @throws std::exception if SYCL execution fails.
         */
        std::vector<std::complex<double>> process_vector_fpga_dummy(
            const std::vector<std::complex<double>> &input_vector);

        /**
         * @brief Performs an Inverse Fast Fourier Transform (IFFT) on a vector of complex numbers using a SYCL kernel.
         *
         * The input vector size must be a power of 2. If not, or if the vector is empty,
         * an empty vector is returned. The IFFT is performed in-place on a copy of the input data.
         *
         * @param input_vector The input vector of complex numbers in the frequency domain.
         * @param apply_scaling If true (default), scales the output by 1/N. If false, no such scaling is applied.
         * @return std::vector<std::complex<double>> The processed vector in the time domain.
         * Returns an empty vector if input size is not a power of 2 or is zero.
         * @throws std::exception if SYCL execution fails.
         */
        std::vector<std::complex<double>> process_vector_ifft_fpga(
            const std::vector<std::complex<double>> &input_vector,
            bool apply_scaling = true); // Added apply_scaling parameter

    } // namespace fpga
} // namespace seal

#endif // SEAL_FPGA_H