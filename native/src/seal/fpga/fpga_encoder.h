#ifndef SEAL_FPGA_FPGA_ENCODER_H
#define SEAL_FPGA_FPGA_ENCODER_H

#include "seal/context.h"
#include "seal/plaintext.h"
#include "seal/memorymanager.h"
#include "seal/util/defines.h"      // For parms_id_type if not in encryptionparams.h
#include "seal/encryptionparams.h"  // For parms_id_type definition
#include <vector>
#include <complex>
#include "seal/util/pointer.h" // For util::Pointer
#include "seal/util/common.h"       // For util::get_power_of_two, util::reverse_bits, etc.
#include "seal/util/polycore.h"     // For util::product_fits_in, util::mul_safe etc.
#include "seal/util/uintcore.h"     // For RNS decomposition helpers
#include "seal/util/rns.h"          // For RNSTool
#include "seal/util/ntt.h"          // For ntt_negacyclic_harvey
#include "seal/fpga/fft.h"          // For NormalFFTHandler
#include <cmath>                    // For std::log2, std::fabs, std::round, std::fmod, std::ceil, std::signbit, std::conj
#include <algorithm>                // For std::copy_n, std::fill_n, std::max
#include <limits>                   // For std::numeric_limits

// Ensure SEALContext::ContextData members are accessible
// Already included via seal/context.h

namespace seal
{
    namespace fpga
    {
        /**
         * @brief Provides functionality for encoding vectors of complex or real numbers into
         * plaintext polynomials for the CKKS scheme, specifically adapted for or an
         * FPGA-targeting workflow.
         *
         * This class mirrors the CKKSEncoder but is intended for use with FPGA-specific
         * FFT/IFFT handlers and potentially other hardware-accelerated components.
         * It handles the embedding of N/2 complex numbers (or N/2 real numbers,
         * which are treated as complex numbers with a zero imaginary part) into a plaintext
         * polynomial of degree N. This enables SIMD operations on encrypted data.
         *
         * @par Mathematical Background
         * Similar to CKKSEncoder, this class implements an approximation of the canonical
         * embedding Z[X]/(X^N+1) -> C^(N/2). It uses a specified FFT/IFFT handler
         * (NormalFFTHandler for FPGA context) and separates scaling operations.
         *
         * @par FPGA-Specific Adaptations
         * - Uses NormalFFTHandler for IFFT during encoding.
         * - Scaling (by `scale / N`) is applied as a distinct step after the IFFT.
         */
        class FPGAEncoder
        {
        public:
            /**
             * @brief Creates an FPGAEncoder instance initialized with the specified SEALContext.
             *
             * The SEALContext must be configured for the CKKS scheme and its parameters
             * must be valid.
             *
             * @param[in] context The SEALContext.
             * @throws std::invalid_argument if the encryption parameters are not valid for CKKS.
             * @throws std::invalid_argument if poly_modulus_degree is not a power of two.
             */
            FPGAEncoder(const SEALContext &context);

            /**
             * @brief Encodes a vector of double-precision floating-point real or complex numbers
             * into a plaintext polynomial.
             *
             * @tparam T Value type of the input vector, can be double or std::complex<double>.
             * @param[in] values The vector of numbers to encode. Its size must be at most slot_count().
             * @param[in] parms_id The parms_id determining the encryption parameters for the result.
             * @param[in] scale The scaling parameter for CKKS encoding.
             * @param[out] destination The Plaintext object to overwrite with the encoded result.
             * @param[in] pool The MemoryPoolHandle for dynamic memory allocations.
             * @throws std::invalid_argument if parameters are invalid (e.g., values_size too large,
             * invalid parms_id, non-positive scale, or if encoding results in values too large).
             */
            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            inline void encode(
                const std::vector<T> &values, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode_internal(values.data(), values.size(), parms_id, scale, destination, std::move(pool));
            }

            /**
             * @brief Encodes a vector of double-precision floating-point real or complex numbers
             * using the top-level encryption parameters from the SEALContext.
             *
             * @tparam T Value type of the input vector, can be double or std::complex<double>.
             * @param[in] values The vector of numbers to encode.
             * @param[in] scale The scaling parameter for CKKS encoding.
             * @param[out] destination The Plaintext object to overwrite with the encoded result.
             * @param[in] pool The MemoryPoolHandle for dynamic memory allocations.
             */
            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            inline void encode(
                const std::vector<T> &values, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode(values, context_.first_parms_id(), scale, destination, std::move(pool));
            }

            /**
             * @brief Encodes a single double-precision floating-point real number into a plaintext.
             * The value is replicated across all N/2 slots. This version directly encodes
             * a constant polynomial and does not use FFT.
             *
             * @param[in] value The real number to encode.
             * @param[in] parms_id The parms_id determining the encryption parameters.
             * @param[in] scale The scaling parameter.
             * @param[out] destination The Plaintext to overwrite.
             * @param[in] pool The MemoryPoolHandle for allocations.
             */
            inline void encode(
                double value, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode_internal(value, parms_id, scale, destination, std::move(pool));
            }

            /**
             * @brief Encodes a single double-precision floating-point real number using top-level parameters.
             * The value is replicated across all N/2 slots. This version directly encodes
             * a constant polynomial and does not use FFT.
             *
             * @param[in] value The real number to encode.
             * @param[in] scale The scaling parameter.
             * @param[out] destination The Plaintext to overwrite.
             * @param[in] pool The MemoryPoolHandle for allocations.
             */
            inline void encode(
                double value, double scale, Plaintext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode(value, context_.first_parms_id(), scale, destination, std::move(pool));
            }

            /**
             * @brief Encodes a single double-precision complex number into a plaintext.
             * The value is replicated across all N/2 slots. This version uses the
             * vector encoding path with FFT.
             *
             * @param[in] value The complex number to encode.
             * @param[in] parms_id The parms_id determining the encryption parameters.
             * @param[in] scale The scaling parameter.
             * @param[out] destination The Plaintext to overwrite.
             * @param[in] pool The MemoryPoolHandle for allocations.
             */
            inline void encode(
                std::complex<double> value, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode_internal(value, parms_id, scale, destination, std::move(pool));
            }

            /**
             * @brief Encodes a single double-precision complex number using top-level parameters.
             * The value is replicated across all N/2 slots. This version uses the
             * vector encoding path with FFT.
             *
             * @param[in] value The complex number to encode.
             * @param[in] scale The scaling parameter.
             * @param[out] destination The Plaintext to overwrite.
             * @param[in] pool The MemoryPoolHandle for allocations.
             */
            inline void encode(
                std::complex<double> value, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode(value, context_.first_parms_id(), scale, destination, std::move(pool));
            }

            /**
             * @brief Encodes a single 64-bit integer into a plaintext without scaling (scale is 1.0).
             * The integer is replicated across all N/2 slots. This version directly encodes
             * a constant polynomial and does not use FFT.
             *
             * @param[in] value The integer to encode.
             * @param[in] parms_id The parms_id determining the encryption parameters.
             * @param[out] destination The Plaintext to overwrite.
             */
            inline void encode(std::int64_t value, parms_id_type parms_id, Plaintext &destination) const
            {
                encode_internal(value, parms_id, destination);
            }

            /**
             * @brief Encodes a single 64-bit integer using top-level parameters without scaling.
             * The integer is replicated across all N/2 slots. This version directly encodes
             * a constant polynomial and does not use FFT.
             *
             * @param[in] value The integer to encode.
             * @param[out] destination The Plaintext to overwrite.
             */
            inline void encode(std::int64_t value, Plaintext &destination) const
            {
                encode(value, context_.first_parms_id(), destination);
            }

            /**
             * @brief Returns the number of slots available for encoding (N/2).
             */
            SEAL_NODISCARD inline std::size_t slot_count() const noexcept
            {
                return slots_;
            }

        private:
            /**
             * @brief Internal implementation for encoding a C-style array of real or complex numbers.
             *
             * This function prepares the input vector (embedding conjugates for real inputs),
             * performs an IFFT using NormalFFTHandler, scales the result by (scale / N),
             * rounds the coefficients, performs RNS decomposition, and finally transforms
             * the plaintext to NTT form.
             *
             * @tparam T Value type, can be double or std::complex<double>.
             * @param[in] values Pointer to the first element of the array to encode.
             * @param[in] values_size Number of elements in the values array.
             * @param[in] parms_id parms_id for the destination plaintext.
             * @param[in] scale CKKS scaling factor.
             * @param[out] destination Resulting plaintext.
             * @param[in] pool MemoryPoolHandle for allocations.
             */
            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            void encode_internal(
                const T *values, std::size_t values_size, parms_id_type parms_id, double scale,
                Plaintext &destination, MemoryPoolHandle pool) const
            {
                // Verify parameters.
                auto context_data_ptr = context_.get_context_data(parms_id);
                if (!context_data_ptr)
                {
                    throw std::invalid_argument("parms_id is not valid for encryption parameters");
                }
                if (!values && values_size > 0)
                {
                    throw std::invalid_argument("values cannot be null");
                }
                if (values_size > slots_)
                {
                    throw std::invalid_argument("values_size is too large");
                }
                if (!pool)
                {
                    throw std::invalid_argument("pool is uninitialized");
                }

                auto &context_data = *context_data_ptr;
                auto &parms = context_data.parms();
                auto &coeff_modulus = parms.coeff_modulus();
                std::size_t coeff_modulus_size = coeff_modulus.size();
                std::size_t coeff_count = parms.poly_modulus_degree(); // This is N

                if (!util::product_fits_in(coeff_modulus_size, coeff_count))
                {
                    throw std::logic_error("invalid parameters");
                }

                if (scale <= 0 || (static_cast<int>(std::log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count()))
                {
                    throw std::invalid_argument("scale out of bounds");
                }

                auto ntt_tables = context_data.small_ntt_tables();
                std::size_t n_for_ifft = coeff_count;

                // Allocate a temporary vector for IFFT input, initialized to zero.
                auto temp_ifft_values_ptr = util::allocate<std::complex<double>>(n_for_ifft, pool);
                std::fill_n(temp_ifft_values_ptr.get(), n_for_ifft, std::complex<double>(0.0, 0.0));
                std::complex<double>* conj_values = temp_ifft_values_ptr.get();

                // Embed input values and their conjugates into the IFFT input vector.
                // values_size is at most slots_ (N/2).
                // Input values are placed at specific bit-reversed indices.
                // For real inputs, their conjugates are themselves.
                // For complex inputs, their complex conjugates are used.
                for (std::size_t i = 0; i < values_size; i++)
                {
                    conj_values[matrix_reps_index_map_[i]] = static_cast<std::complex<double>>(values[i]);
                    if (std::is_same<T, double>::value) {
                        // For real T, conj(values[i]) is just values[i] (as a complex number with 0 imaginary part)
                         conj_values[matrix_reps_index_map_[i + slots_]] = static_cast<std::complex<double>>(values[i]);
                    } else { 
                        // For complex T, use std::conj
                        conj_values[matrix_reps_index_map_[i + slots_]] = std::conj(static_cast<std::complex<double>>(values[i]));
                    }
                }

                // Perform IFFT using NormalFFTHandler. This IFFT is unnormalized.
                NormalFFTHandler normal_fft_handler;
                int log_n_val = util::get_power_of_two(n_for_ifft);
                 if (log_n_val < 0) { 
                    throw std::logic_error("n_for_ifft (coeff_count) is not a power of two");
                }
                normal_fft_handler.inverse_fft(conj_values, log_n_val);
                
                // Scale the real parts of the IFFT output by (scale / N).
                // The IFFT of a conjugate-symmetric input yields real outputs.
                double overall_scale_factor = scale / static_cast<double>(n_for_ifft);
                for (std::size_t i = 0; i < n_for_ifft; i++)
                {
                    // Result of IFFT on conjugate-symmetric data is real (imaginary part should be close to 0 due to precision).
                    // We scale the real part.
                    conj_values[i].real(conj_values[i].real() * overall_scale_factor);
                    conj_values[i].imag(0.0); // Explicitly zero out imaginary part
                }

                // Check for coefficient overflow before rounding and RNS decomposition.
                double max_coeff = 0;
                for (std::size_t i = 0; i < n_for_ifft; i++)
                {
                    max_coeff = std::max<>(max_coeff, std::fabs(conj_values[i].real()));
                }
                // Approximate bit count of the largest scaled coefficient.
                // Add 1 for sign, 1 for possible carry from rounding.
                int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
                if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
                {
                    throw std::invalid_argument("encoded values are too large for the coefficient modulus");
                }

                // Prepare destination Plaintext object.
                destination.parms_id() = parms_id_zero; // Temporarily set to zero before resize.
                destination.resize(util::mul_safe(coeff_count, coeff_modulus_size)); // coeff_count is N here.

                // Round the scaled real coefficients to nearest integers and perform RNS decomposition.
                double two_pow_64 = std::pow(2.0, 64); // Used for multi-precision decomposition.
                if (max_coeff_bit_count <= 64) // Coefficients fit in a single uint64_t
                {
                    for (std::size_t i = 0; i < n_for_ifft; i++) // n_for_ifft is coeff_count (N)
                    { 
                        double coeffd = std::round(conj_values[i].real());
                        bool is_negative = std::signbit(coeffd);
                        std::uint64_t coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

                        if (is_negative) {
                            for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                                // (poly_coeffs[j])[i] = RNS component of -coeffu
                                destination[i + (j * coeff_count)] = util::negate_uint_mod(
                                    util::barrett_reduce_64(coeffu, coeff_modulus[j]), coeff_modulus[j]);
                            }
                        } else {
                            for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                                // (poly_coeffs[j])[i] = RNS component of coeffu
                                destination[i + (j * coeff_count)] = util::barrett_reduce_64(coeffu, coeff_modulus[j]);
                            }
                        }
                    }
                }
                else if (max_coeff_bit_count <= 128) // Coefficients fit in two uint64_t's
                {
                    for (std::size_t i = 0; i < n_for_ifft; i++)
                    {
                        double coeffd = std::round(conj_values[i].real());
                        bool is_negative = std::signbit(coeffd);
                        coeffd = std::fabs(coeffd);

                        // Decompose into two 64-bit parts.
                        std::uint64_t coeffu_parts[2]{ static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64)),
                                                     static_cast<std::uint64_t>(coeffd / two_pow_64) };

                        if (is_negative) {
                            for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                                destination[i + (j * coeff_count)] = util::negate_uint_mod(
                                    util::barrett_reduce_128(coeffu_parts, coeff_modulus[j]), coeff_modulus[j]);
                            }
                        } else {
                            for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                                destination[i + (j * coeff_count)] = util::barrett_reduce_128(coeffu_parts, coeff_modulus[j]);
                            }
                        }
                    }
                }
                else // Coefficients are larger and require general RNS decomposition
                { 
                    // Temporary buffer for a single multi-precision coefficient.
                    auto coeffu_alloc = util::allocate_uint(coeff_modulus_size, pool); 
                    for (std::size_t i = 0; i < n_for_ifft; i++) {
                        double coeffd = std::round(conj_values[i].real());
                        bool is_negative = std::signbit(coeffd);
                        coeffd = std::fabs(coeffd);

                        util::set_zero_uint(coeff_modulus_size, coeffu_alloc.get());
                        auto coeffu_ptr_base = coeffu_alloc.get();
                        
                        // Decompose coeffd (absolute value) into base 2^64 representation.
                        int k_idx = 0; 
                        while (coeffd >= 1 && k_idx < static_cast<int>(coeff_modulus_size)) { 
                            coeffu_ptr_base[k_idx++] = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
                            coeffd /= two_pow_64;
                        }
                        
                        // Perform RNS decomposition for this coefficient.
                        context_data.rns_tool()->base_q()->decompose_array(coeffu_alloc.get(), 1, pool); 

                        // Store RNS components in destination, negating if necessary.
                        if (is_negative) {
                            for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                                destination[i + (j * coeff_count)] = util::negate_uint_mod(coeffu_alloc[j], coeff_modulus[j]);
                            }
                        } else {
                            for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                                destination[i + (j * coeff_count)] = coeffu_alloc[j];
                            }
                        }
                    }
                }
                
                // Transform destination plaintext to NTT domain (CKKS plaintexts are in NTT form).
                for (std::size_t i = 0; i < coeff_modulus_size; i++)
                {
                    // util::ntt_negacyclic_harvey operates on one RNS component at a time.
                    util::ntt_negacyclic_harvey(destination.data(i * coeff_count), ntt_tables[i]);
                }

                // Set final Plaintext properties.
                destination.parms_id() = parms_id;
                destination.scale() = scale; // Store the original scale.
            }


            // Declarations for non-templated internal functions (definitions will be in .cpp)
            void encode_internal(
                double value, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool) const;

            void encode_internal(
                std::complex<double> value, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool) const;
            
            void encode_internal(std::int64_t value, parms_id_type parms_id, Plaintext &destination) const;


            MemoryPoolHandle pool_ = MemoryManager::GetPool();
            SEALContext context_;
            std::size_t slots_;
            util::Pointer<std::size_t> matrix_reps_index_map_;
        };
    } // namespace fpga
} // namespace seal

#endif // SEAL_FPGA_FPGA_ENCODER_H