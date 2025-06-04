#include "seal/fpga/fpga_encoder.h"
// #include "seal/fpga/fft.h" // NormalFFTHandler is now used within the templated method in the header
#include "seal/util/common.h"
#include "seal/util/polycore.h" 
#include "seal/util/uintarithsmallmod.h" 
#include "seal/util/rns.h" 
#include "seal/util/uintcore.h" 
#include "seal/util/ntt.h" 
#include <cmath>      
#include <complex>
#include <vector>
#include <stdexcept>
#include <algorithm>    
#include <limits>      

// Ensure SEALContext::ContextData members are accessible
#include "seal/context.h"


namespace seal
{
    namespace fpga
    {
        FPGAEncoder::FPGAEncoder(const SEALContext &context) : context_(context)
        {
            // Verify context parameters are set and scheme is CKKS.
            if (!context_.parameters_set())
            {
                throw std::invalid_argument("encryption parameters are not set correctly");
            }

            auto &context_data = *context_.first_context_data();
            if (context_data.parms().scheme() != scheme_type::ckks)
            {
                throw std::invalid_argument("unsupported scheme; FPGAEncoder only supports CKKS");
            }

            // Get polynomial modulus degree (N) and calculate number of slots (N/2).
            std::size_t coeff_count = context_data.parms().poly_modulus_degree();
            slots_ = coeff_count >> 1; // N/2

            // Calculate logN for bit reversal.
            int logn = util::get_power_of_two(coeff_count);
             if (logn < 0) { 
                throw std::invalid_argument("poly_modulus_degree must be a power of two");
            }

            // Allocate and populate the matrix_reps_index_map_.
            // This map is used for the CKKS embedding: it maps the N/2 input complex values
            // (and their conjugates) to the correct N positions in the vector that will be
            // inverse transformed (IFFT).
            matrix_reps_index_map_ = util::allocate<std::size_t>(coeff_count, pool_);

            // The generator 'gen = 3' and modulus 'm = 2*N' are standard for CKKS's
            // use of 2N-th roots of unity.
            uint64_t gen = 3;
            uint64_t pos = 1; // Current power of the generator.
            uint64_t m = static_cast<uint64_t>(coeff_count) << 1; // 2*N
            for (std::size_t i = 0; i < slots_; i++)
            {
                // Calculate indices for the i-th complex value and its conjugate.
                uint64_t index1 = (pos - 1) >> 1;
                uint64_t index2 = (m - pos - 1) >> 1;

                // Store the bit-reversed indices.
                matrix_reps_index_map_[i] = static_cast<std::size_t>(util::reverse_bits(index1, logn));
                matrix_reps_index_map_[slots_ | i] = static_cast<std::size_t>(util::reverse_bits(index2, logn));

                // Advance to the next power of the generator.
                pos *= gen;
                pos &= (m - 1); // Modulo 2N
            }
        }
        
        // Internal implementation for encoding a single double value.
        // This creates a constant polynomial; no FFT is used.
        void FPGAEncoder::encode_internal(
            double value, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool) const
        {
            // Verify SEALContext and MemoryPoolHandle.
            auto context_data_ptr = context_.get_context_data(parms_id);
            if (!context_data_ptr) {
                throw std::invalid_argument("parms_id is not valid for encryption parameters");
            }
            if (!pool) { 
                throw std::invalid_argument("pool is uninitialized");
            }

            // Get relevant parameters from context_data.
            auto &context_data = *context_data_ptr;
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            std::size_t coeff_modulus_size = coeff_modulus.size();
            std::size_t coeff_count = parms.poly_modulus_degree(); 

            // Check for potential overflow if parameters are too large.
            if (!util::product_fits_in(coeff_modulus_size, coeff_count)) {
                throw std::logic_error("invalid parameters: product of dimensions too large");
            }
            
            // Validate scale: must be positive and within bounds determined by coefficient modulus.
            if (scale <= 0 || (static_cast<int>(std::log2(scale)) +1 >= context_data.total_coeff_modulus_bit_count())) { // std::fabs removed for scale, as scale must be positive
                 throw std::invalid_argument("scale is out of bounds");
            }
            
            // Scale the input value.
            double scaled_value = value * scale;

            // Check if the scaled value is too large to be encoded.
            // Add 2 bits for sign and potential rounding carry.
            int value_coeff_bit_count = static_cast<int>(std::log2(std::fabs(scaled_value))) + 2; 
            if (value_coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
                throw std::invalid_argument("encoded value is too large for the coefficient modulus");
            }

            double two_pow_64 = std::pow(2.0, 64); // For multi-precision decomposition.

            // Prepare destination plaintext: set parms_id to zero before resize, then resize.
            destination.parms_id() = parms_id_zero; 
            destination.resize(coeff_count * coeff_modulus_size);

            // Round the scaled value to the nearest integer.
            double rounded_scaled_value = std::round(scaled_value);
            bool is_negative = std::signbit(rounded_scaled_value);
            double abs_rounded_scaled_value = std::fabs(rounded_scaled_value);
            
            // RNS decomposition of the constant term (c_0). Other coefficients (c_1, ..., c_{N-1}) are zero.
            if (value_coeff_bit_count <= 64) { // Fits in one uint64_t
                std::uint64_t coeffu = static_cast<std::uint64_t>(abs_rounded_scaled_value);
                if (is_negative) {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                        destination[j * coeff_count] = util::negate_uint_mod(
                            util::barrett_reduce_64(coeffu, coeff_modulus[j]), coeff_modulus[j]);
                        // Set other coefficients to 0 for this RNS component.
                        std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                    }
                } else {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                        destination[j * coeff_count] = util::barrett_reduce_64(coeffu, coeff_modulus[j]);
                        std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                    }
                }
            } else if (value_coeff_bit_count <= 128) { // Fits in two uint64_t's
                std::uint64_t coeffu_parts[2]{ static_cast<std::uint64_t>(std::fmod(abs_rounded_scaled_value, two_pow_64)),
                                                 static_cast<std::uint64_t>(abs_rounded_scaled_value / two_pow_64) };
                if (is_negative) {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                         destination[j * coeff_count] = util::negate_uint_mod(
                            util::barrett_reduce_128(coeffu_parts, coeff_modulus[j]), coeff_modulus[j]);
                         std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                    }
                } else {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                        destination[j * coeff_count] = util::barrett_reduce_128(coeffu_parts, coeff_modulus[j]);
                        std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                    }
                }
            } else { // Requires general RNS decomposition for larger values.
                auto coeffu_alloc = util::allocate_uint(coeff_modulus_size, pool); 
                util::set_zero_uint(coeff_modulus_size, coeffu_alloc.get());
                auto coeffu_ptr_base = coeffu_alloc.get();
                
                double temp_abs_val = abs_rounded_scaled_value;
                int k_idx = 0; 
                // Decompose into base 2^64 representation.
                while (temp_abs_val >= 1 && k_idx < static_cast<int>(coeff_modulus_size)) { 
                    coeffu_ptr_base[k_idx++] = static_cast<std::uint64_t>(std::fmod(temp_abs_val, two_pow_64));
                    temp_abs_val /= two_pow_64;
                }
                
                // Perform RNS decomposition using RNSTool.
                // decompose_array expects an array of multi-precision numbers. Here we have one.
                context_data.rns_tool()->base_q()->decompose_array(coeffu_alloc.get(), 1, pool); 
                
                // Store the RNS components.
                if (is_negative) {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                        destination[j * coeff_count] = util::negate_uint_mod(coeffu_alloc[j], coeff_modulus[j]);
                        std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                    }
                } else {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                        destination[j * coeff_count] = coeffu_alloc[j];
                         std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                    }
                }
            }

            // Transform the constant polynomial to NTT form for CKKS.
            auto ntt_tables = context_data.small_ntt_tables();
            for (std::size_t i = 0; i < coeff_modulus_size; i++) {
                // ntt_negacyclic_harvey operates on one RNS component (a polynomial) at a time.
                util::ntt_negacyclic_harvey(destination.data(i * coeff_count), ntt_tables[i]);
            }

            // Set final Plaintext properties.
            destination.parms_id() = parms_id;
            destination.scale() = scale;
        }
        
        // Internal implementation for encoding a single complex double value.
        // This uses the vector encoding path (defined in the header) with FFT.
        void FPGAEncoder::encode_internal(
            std::complex<double> value, parms_id_type parms_id, double scale, Plaintext &destination,
            MemoryPoolHandle pool) const
        {
            // Create a temporary vector of size slots_ (N/2), filled with the input complex value.
            std::vector<std::complex<double>> temp_values(slots_, value);
            // Call the templated vector encode_internal (defined in fpga_encoder.h).
            encode_internal(temp_values.data(), temp_values.size(), parms_id, scale, destination, std::move(pool));
        }

        // Internal implementation for encoding a 64-bit integer (constant polynomial, no FFT).
        void FPGAEncoder::encode_internal(std::int64_t value, parms_id_type parms_id, Plaintext &destination) const
        {
            // Verify SEALContext.
            auto context_data_ptr = context_.get_context_data(parms_id);
            if (!context_data_ptr) {
                throw std::invalid_argument("parms_id is not valid for encryption parameters");
            }

            // Get relevant parameters.
            auto &context_data = *context_data_ptr;
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            std::size_t coeff_modulus_size = coeff_modulus.size();
            std::size_t coeff_count = parms.poly_modulus_degree();

            // Check for potential overflow.
            if (!util::product_fits_in(coeff_modulus_size, coeff_count)) {
                throw std::logic_error("invalid parameters: product of dimensions too large");
            }

            // Check if the integer value is too large to be encoded.
            // Add 2 bits for sign and potential for the value to be close to a modulus.
            int value_coeff_bit_count = util::get_significant_bit_count(static_cast<std::uint64_t>(std::llabs(value))) + 2;
            if (value_coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
                throw std::invalid_argument("encoded integer value is too large");
            }

            // Prepare destination plaintext.
            destination.parms_id() = parms_id_zero; 
            destination.resize(coeff_count * coeff_modulus_size);

            // Encode the integer as a constant polynomial c_0 = value, c_i = 0 for i > 0.
            // Convert to RNS representation.
            if (value < 0) {
                for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                    std::uint64_t tmp = static_cast<std::uint64_t>(value); // Implicit conversion for negative.
                    // Bring to [0, modulus-1] by adding modulus.
                    tmp = util::add_uint_mod(tmp, coeff_modulus[j].value(), coeff_modulus[j]); 
                    destination[j * coeff_count] = tmp; // Set constant term.
                    // Set other coefficients to 0 for this RNS component.
                    std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                }
            } else {
                for (std::size_t j = 0; j < coeff_modulus_size; j++) {
                    std::uint64_t tmp = static_cast<std::uint64_t>(value);
                    tmp = util::barrett_reduce_64(tmp, coeff_modulus[j]); // Reduce if positive.
                    destination[j * coeff_count] = tmp; // Set constant term.
                    std::fill_n(destination.data() + j * coeff_count + 1, coeff_count - 1, 0ULL);
                }
            }
            
            // Transform the constant polynomial to NTT form for CKKS.
            auto ntt_tables = context_data.small_ntt_tables();
            for (std::size_t i = 0; i < coeff_modulus_size; i++) {
                util::ntt_negacyclic_harvey(destination.data(i * coeff_count), ntt_tables[i]);
            }

            // Set final Plaintext properties. Scale for integer encoding is 1.0.
            destination.parms_id() = parms_id;
            destination.scale() = 1.0;
        }

    } // namespace fpga
} // namespace seal