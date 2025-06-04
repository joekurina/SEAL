#ifndef SEAL_FPGA_FPGA_ENCODER_H
#define SEAL_FPGA_FPGA_ENCODER_H

#include "seal/context.h"
#include "seal/plaintext.h"
#include "seal/memorymanager.h"
#include "seal/util/defines.h"
#include "seal/encryptionparams.h"
#include <vector>
#include <complex>
#include "seal/util/pointer.h"
#include "seal/util/common.h"
#include "seal/util/polycore.h"
#include "seal/util/uintcore.h"
#include "seal/util/rns.h"
#include "seal/util/ntt.h"
#include "seal/fpga/fft.h" // For NormalFFTHandler
#include <cmath>
#include <algorithm>
#include <limits>
#ifdef SEAL_USE_MSGSL
#include "gsl/span"
#endif

// Forward declaration from seal/valcheck.h to break circular dependency if any
namespace seal {
    SEAL_NODISCARD bool is_valid_for(const Plaintext &in, const SEALContext &context);

    // Define from_complex and its specializations here, before FPGAEncoder
    // This helper function converts a std::complex<double> to T_out.
    // If T_out is double, it returns the real part.
    // If T_out is std::complex<double>, it returns the complex number itself.
    template <
        typename T_out, typename = std::enable_if_t<
                            std::is_same<std::remove_cv_t<T_out>, double>::value ||
                            std::is_same<std::remove_cv_t<T_out>, std::complex<double>>::value>>
    SEAL_NODISCARD inline T_out from_complex_internal_helper(std::complex<double> in);

    template <>
    SEAL_NODISCARD inline double from_complex_internal_helper<double>(std::complex<double> in)
    {
        return in.real();
    }

    template <>
    SEAL_NODISCARD inline std::complex<double> from_complex_internal_helper<std::complex<double>>(std::complex<double> in)
    {
        return in;
    }
}


namespace seal
{
    namespace fpga
    {
        /**
         * @brief Provides functionality for encoding vectors of complex or real numbers into
         * plaintext polynomials for the CKKS scheme, and decoding them back.
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
         * - Uses NormalFFTHandler for IFFT during encoding and FFT during decoding.
         * - Scaling is applied as a distinct step after IFFT (encoding) and before FFT (decoding).
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

            // ENCODING METHODS

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
             * @throws std::invalid_argument if parameters are invalid.
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
#ifdef SEAL_USE_MSGSL
            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            inline void encode(
                gsl::span<const T> values, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode_internal(
                    values.data(), static_cast<std::size_t>(values.size()), parms_id, scale, destination, std::move(pool));
            }

            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            inline void encode(
                gsl::span<const T> values, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode(values, context_.first_parms_id(), scale, destination, std::move(pool));
            }
#endif
            inline void encode(
                double value, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode_internal(value, parms_id, scale, destination, std::move(pool));
            }

            inline void encode(
                double value, double scale, Plaintext &destination, MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode(value, context_.first_parms_id(), scale, destination, std::move(pool));
            }

            inline void encode(
                std::complex<double> value, parms_id_type parms_id, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode_internal(value, parms_id, scale, destination, std::move(pool));
            }

            inline void encode(
                std::complex<double> value, double scale, Plaintext &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                encode(value, context_.first_parms_id(), scale, destination, std::move(pool));
            }

            inline void encode(std::int64_t value, parms_id_type parms_id, Plaintext &destination) const
            {
                encode_internal(value, parms_id, destination);
            }

            inline void encode(std::int64_t value, Plaintext &destination) const
            {
                encode(value, context_.first_parms_id(), destination);
            }

            // DECODING METHODS

            /**
             * @brief Decodes a plaintext polynomial into a vector of double-precision
             * floating-point real or complex numbers.
             *
             * @tparam T Value type of the output vector, can be double or std::complex<double>.
             * @param[in] plain The Plaintext to decode. It must be in NTT form.
             * @param[out] destination The vector to be overwritten with the decoded slot values.
             * @param[in] pool The MemoryPoolHandle for dynamic memory allocations.
             * @throws std::invalid_argument if plain is not valid for the encryption parameters
             * or not in NTT form.
             */
            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            inline void decode(
                const Plaintext &plain, std::vector<T> &destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                destination.resize(slots_);
                decode_internal(plain, destination.data(), std::move(pool));
            }
#ifdef SEAL_USE_MSGSL
            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            inline void decode(
                const Plaintext &plain, gsl::span<T> destination,
                MemoryPoolHandle pool = MemoryManager::GetPool()) const
            {
                if (destination.size() != static_cast<std::ptrdiff_t>(slots_)) // GSL uses ptrdiff_t for size
                {
                    throw std::invalid_argument("destination has invalid size");
                }
                decode_internal(plain, destination.data(), std::move(pool));
            }
#endif
            /**
             * @brief Returns the number of slots available for encoding (N/2).
             */
            SEAL_NODISCARD inline std::size_t slot_count() const noexcept
            {
                return slots_;
            }

        private:
            // Internal encoding methods (definitions in header or .cpp)
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
                        // decompose_array expects an array of multi-precision numbers. Here we have one.
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

            // Internal decoding method (definition in header)
            template <
                typename T, typename = std::enable_if_t<
                                std::is_same<std::remove_cv_t<T>, double>::value ||
                                std::is_same<std::remove_cv_t<T>, std::complex<double>>::value>>
            void decode_internal(const Plaintext &plain, T *destination, MemoryPoolHandle pool) const
            {
                // Verify parameters.
                if (!is_valid_for(plain, context_)) 
                {
                    throw std::invalid_argument("plain is not valid for encryption parameters");
                }
                if (!plain.is_ntt_form())
                {
                    throw std::invalid_argument("plain is not in NTT form");
                }
                if (!destination)
                {
                    throw std::invalid_argument("destination cannot be null");
                }
                if (!pool)
                {
                    throw std::invalid_argument("pool is uninitialized");
                }

                auto &context_data = *context_.get_context_data(plain.parms_id());
                auto &parms = context_data.parms();
                std::size_t coeff_modulus_size = parms.coeff_modulus().size();
                std::size_t coeff_count = parms.poly_modulus_degree(); // N
                std::size_t rns_poly_uint64_count = util::mul_safe(coeff_count, coeff_modulus_size);

                auto ntt_tables = context_data.small_ntt_tables();

                if (plain.scale() <= 0 || (static_cast<int>(std::log2(plain.scale())) >= context_data.total_coeff_modulus_bit_count()))
                {
                    throw std::invalid_argument("plain.scale() out of bounds");
                }

                // Create mutable copy of input plaintext data.
                auto plain_copy_poly_ptr = util::allocate_uint(rns_poly_uint64_count, pool);
                util::set_uint(plain.data(), rns_poly_uint64_count, plain_copy_poly_ptr.get());
                util::RNSIter plain_copy_iter(plain_copy_poly_ptr.get(), coeff_count);

                // Transform each RNS component from NTT form to coefficient form.
                for (std::size_t i = 0; i < coeff_modulus_size; i++)
                {
                    util::inverse_ntt_negacyclic_harvey(plain_copy_iter[i], ntt_tables[i]);
                }

                // CRT-compose the RNS polynomial components into a single multi-precision integer polynomial (coeffs mod Q).
                context_data.rns_tool()->base_q()->compose_array(plain_copy_poly_ptr.get(), coeff_count, pool);

                // Convert multi-precision integer coefficients to std::complex<double>.
                auto temp_fft_values_ptr = util::allocate<std::complex<double>>(coeff_count, pool);
                std::complex<double>* fft_input_values = temp_fft_values_ptr.get();

                // Scaling factor to apply to the double-converted coefficients BEFORE FFT.
                // P_int_k approx IFFT_unnorm(phi_embed)_k * (plain.scale() / N)
                // We want fft_input_values_k approx IFFT_unnorm(phi_embed)_k / N
                // So, fft_input_values_k = P_double_k / plain.scale()
                double inv_scale_for_fft_input = 1.0 / plain.scale();

                double two_pow_64 = std::pow(2.0, 64);
                auto decryption_modulus = context_data.total_coeff_modulus();
                auto upper_half_threshold = context_data.upper_half_threshold();
                
                // Temporary buffer for multi-precision subtraction result if needed
                auto magnitude_ptr_alloc = util::allocate_uint(coeff_modulus_size, pool);
                uint64_t* magnitude_ptr = magnitude_ptr_alloc.get();

                for (std::size_t i = 0; i < coeff_count; i++)
                {
                    const uint64_t* coeff_composed_ptr = plain_copy_poly_ptr.get() + (i * coeff_modulus_size);
                    double current_val_double = 0;
                    double current_power_of_2_64 = 1.0; 

                    if (util::is_greater_than_or_equal_uint(coeff_composed_ptr, upper_half_threshold, coeff_modulus_size))
                    {
                        // It's negative. Actual value is coeff_composed_ptr - Q.
                        // We compute Q - coeff_composed_ptr to get the magnitude of the negative value.
                        util::sub_uint(decryption_modulus, coeff_composed_ptr, coeff_modulus_size, magnitude_ptr);
                        for (size_t k = 0; k < coeff_modulus_size; k++)
                        {
                            current_val_double += static_cast<double>(magnitude_ptr[k]) * current_power_of_2_64;
                            if (k < coeff_modulus_size - 1) 
                            {
                                 current_power_of_2_64 *= two_pow_64;
                            }
                        }
                        current_val_double = -current_val_double;
                    }
                    else // It's positive or zero
                    {
                        for (size_t k = 0; k < coeff_modulus_size; k++)
                        {
                            current_val_double += static_cast<double>(coeff_composed_ptr[k]) * current_power_of_2_64;
                             if (k < coeff_modulus_size - 1)
                            {
                                 current_power_of_2_64 *= two_pow_64;
                            }
                        }
                    }
                    // Apply scaling before FFT
                    fft_input_values[i] = { current_val_double * inv_scale_for_fft_input, 0.0 };
                }

                // Perform forward FFT using NormalFFTHandler.
                NormalFFTHandler normal_fft_handler;
                int log_n_val = util::get_power_of_two(coeff_count);
                if (log_n_val < 0) {
                    throw std::logic_error("coeff_count is not a power of two in decode");
                }
                normal_fft_handler.forward_fft(fft_input_values, log_n_val);
                
                // After FFT_unnorm( IFFT_unnorm(phi_embed)/N ), the result is phi_embed.
                // No further scaling (like division by N) is needed here.
                for (std::size_t i = 0; i < slots_; i++)
                {
                    std::complex<double> val_from_fft = fft_input_values[matrix_reps_index_map_[i]];
                    destination[i] = seal::from_complex_internal_helper<T>(val_from_fft);
                }
            }

            MemoryPoolHandle pool_ = MemoryManager::GetPool();
            SEALContext context_;
            std::size_t slots_;
            util::Pointer<std::size_t> matrix_reps_index_map_;
        };
    } // namespace fpga
} // namespace seal

#endif // SEAL_FPGA_FPGA_ENCODER_H
