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
         * @brief Generates precomputation tables (matrix_reps_index_map and inv_root_powers)
         * required for CKKS encoding.
         * This logic is adapted from the CKKSEncoder constructor.
         *
         * @param[in] coeff_count The polynomial modulus degree (N).
         * @param[in] pool The MemoryPoolHandle for memory allocations.
         * @param[out] matrix_reps_index_map_host Vector to be populated with the matrix representatives index map.
         * @param[out] inv_root_powers_host Vector to be populated with the inverse root powers for IFFT.
         */
        void generate_ckks_encoding_tables(
            std::size_t coeff_count,
            MemoryPoolHandle pool,
            std::vector<std::size_t> &matrix_reps_index_map_host,
            std::vector<std::complex<double>> &inv_root_powers_host);

        /**
         * @brief Orchestrates the CKKS encoding process using FPGA-accelerated SYCL kernels.
         * This function takes numerical values (real or complex) and encodes them into a SEAL Plaintext object
         * by leveraging SYCL kernels for computationally intensive steps. It internally generates
         * necessary precomputation tables like matrix_reps_index_map and inv_root_powers by calling
         * generate_ckks_encoding_tables.
         *
         * @tparam T The type of the input values (typically double or std::complex<double>).
         * @param[in] context The SEALContext containing encryption parameters.
         * @param[in] values Pointer to the array of input values to be encoded.
         * @param[in] values_size The number of elements in the input values array.
         * @param[in] parms_id The parms_id determining the encryption parameters from the context to be used.
         * @param[in] scale The scaling factor (Delta) for CKKS encoding precision.
         * @param[out] destination The Plaintext object to be overwritten with the FPGA-encoded result.
         * @param[in] pool The MemoryPoolHandle for managing memory allocations.
         * @note This function assumes that the SYCL environment and necessary FPGA resources are available and properly configured.
         */
        template <typename T>
        void encode_ckks_fpga(
            const SEALContext &context,
            const T *values,
            std::size_t values_size,
            seal::parms_id_type parms_id, // Corrected: seal::parms_id_type
            double scale,
            Plaintext &destination,
            MemoryPoolHandle pool
        );
    } // namespace fpga
} // namespace seal

#endif // SEAL_FPGA_H