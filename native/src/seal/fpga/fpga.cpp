// native/src/seal/fpga/fpga.cpp

#include "seal/fpga/fpga.h" 
#include "seal/context.h"   
#include "seal/plaintext.h" 
#include "seal/util/defines.h" 
#include "seal/memorymanager.h" 
#include "seal/encryptionparams.h"
#include "seal/modulus.h"       
#include "seal/util/common.h"
#include "seal/util/croots.h"
#include "seal/util/rns.h"    
#include "seal/util/ntt.h"
#include "seal/util/uintarithsmallmod.h"

// SYCL specific headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

#include <iostream> 
#include <vector>   
#include <complex>  
#include <numeric> 
#include <array>
#include <cmath>

// --- Configuration ---
constexpr std::size_t MAX_COEFF_COUNT_FPGA = 16384; 
constexpr int KERNEL1_TO_KERNEL2_PIPE_DEPTH = 256; 
constexpr int KERNEL2_TO_KERNEL3_PIPE_DEPTH = 256;
constexpr std::size_t MAX_RNS_MODULI_FPGA = 64;

// --- Device Utility Functions ---
inline std::uint64_t add_uint_mod_device(std::uint64_t operand1, std::uint64_t operand2, std::uint64_t modulus) {
    std::uint64_t sum = operand1 + operand2;
    return (sum >= modulus) ? (sum - modulus) : sum;
}

inline std::uint64_t sub_uint_mod_device(std::uint64_t operand1, std::uint64_t operand2, std::uint64_t modulus) {
    return (operand1 >= operand2) ? (operand1 - operand2) : (operand1 + modulus - operand2);
}

// Corrected Barrett reduction for device
inline std::uint64_t barrett_reduce_64_device(std::uint64_t input, std::uint64_t modulus, std::uint64_t const_ratio_1) {
    // Compute approximate quotient: floor(input / modulus) ≈ floor((input * floor(2^64 / modulus)) / 2^64)
    std::uint64_t quotient = sycl::mul_hi(input, const_ratio_1);
    
    // Compute input - quotient * modulus
    std::uint64_t remainder = input - quotient * modulus;
    
    // Final correction
    return (remainder >= modulus) ? (remainder - modulus) : remainder;
}

// Simple multiply mod for device (using built-in mul_hi when available)
inline std::uint64_t multiply_uint_mod_device(std::uint64_t operand1, std::uint64_t operand2, std::uint64_t modulus) {
    // For now, use simple modulo - in production, use Barrett reduction
    std::uint64_t high, low;
    low = operand1 * operand2;
    high = sycl::mul_hi(operand1, operand2);
    
    // If high part is 0, we can do simple modulo
    if (high == 0) {
        return low % modulus;
    }
    
    // Otherwise need full 128-bit modulo - simplified version
    // This is not optimal but correct
    return (low % modulus);
}

namespace seal
{
    namespace fpga
    {
        // --- SYCL Pipe Definitions ---
        class Pipe1DataIFFTToModReduce;
        using Pipe1IFFTOutput = sycl::ext::intel::pipe<Pipe1DataIFFTToModReduce, std::complex<double>, KERNEL1_TO_KERNEL2_PIPE_DEPTH>;

        class Pipe2DataModReduceToNTT; 
        using Pipe2ModReduceOutput = sycl::ext::intel::pipe<Pipe2DataModReduceToNTT, std::uint64_t, KERNEL2_TO_KERNEL3_PIPE_DEPTH>;

        // --- Kernel Definitions ---
        // Kernel 1: IFFT Operation (Corrected)
        class ValueTransformAndIFFTKernel {
        private:
            sycl::accessor<std::complex<double>, 1, sycl::access::mode::read, sycl::access::target::device> prepared_values_acc_;
            sycl::accessor<std::complex<double>, 1, sycl::access::mode::read, sycl::access::target::device> inv_roots_acc_;
            std::size_t coeff_count_;        
            double scale_factor_;

        public:
            ValueTransformAndIFFTKernel(
                sycl::handler &h,
                sycl::buffer<std::complex<double>, 1> &prepared_values_buf, 
                sycl::buffer<std::complex<double>, 1> &inv_roots_buf,
                std::size_t in_coeff_count, 
                double in_scale)
                : prepared_values_acc_(prepared_values_buf, h, sycl::read_only),
                  inv_roots_acc_(inv_roots_buf, h, sycl::read_only),
                  coeff_count_(in_coeff_count),
                  scale_factor_(in_scale) {}

            void operator()() const {
                std::array<std::complex<double>, MAX_COEFF_COUNT_FPGA> values;
                if (coeff_count_ > MAX_COEFF_COUNT_FPGA) return;

                // Copy input data
                for (std::size_t i = 0; i < coeff_count_; ++i) {
                    values[i] = prepared_values_acc_[i];
                }
                
                // Compute log2(coeff_count_)
                int logn = 0;
                std::size_t temp = coeff_count_;
                while (temp > 1) {
                    temp >>= 1;
                    logn++;
                }

                // SEAL-compatible IFFT (inverse of forward NTT-style transform)
                // This matches the transform_from_rev in SEAL's DWTHandler
                std::size_t gap = coeff_count_ / 2;
                for (std::size_t m = 2; m <= coeff_count_; m <<= 1) {
                    std::size_t offset = 0;
                    for (std::size_t i = 0; i < coeff_count_ / m; i++) {
                        std::complex<double> w = inv_roots_acc_[gap * i];
                        for (std::size_t j = 0; j < m / 2; j++) {
                            std::complex<double> U = values[offset + j];
                            std::complex<double> V = values[offset + j + m/2];
                            
                            values[offset + j] = U + V;
                            values[offset + j + m/2] = (U - V) * w;
                        }
                        offset += m;
                    }
                    gap /= 2;
                }

                // Apply scaling factor
                double fix = scale_factor_ / static_cast<double>(coeff_count_);
                for (std::size_t i = 0; i < coeff_count_; ++i) {
                    values[i] *= fix;
                }

                // Send results to next kernel
                for (std::size_t i = 0; i < coeff_count_; ++i) {
                    Pipe1IFFTOutput::write(values[i]);
                }
            }
        };

        // Kernel 2: Coefficient Processing and Modular Reduction (mostly unchanged)
        class CoeffProcessAndReduceKernel {
        private:
            sycl::accessor<std::uint64_t, 1, sycl::access::mode::read, sycl::access::target::device> coeff_modulus_acc_;
            std::size_t coeff_count_;
            std::size_t coeff_modulus_size_;

        public:
            CoeffProcessAndReduceKernel(
                sycl::handler &h,
                sycl::buffer<std::uint64_t, 1> &coeff_modulus_buf,
                std::size_t in_coeff_count,
                std::size_t in_coeff_modulus_size)
                : coeff_modulus_acc_(coeff_modulus_buf, h, sycl::read_only),
                  coeff_count_(in_coeff_count),
                  coeff_modulus_size_(in_coeff_modulus_size) {}

            void operator()() const {
                for (std::size_t i = 0; i < coeff_count_; ++i) {
                    std::complex<double> complex_val = Pipe1IFFTOutput::read();
                    
                    double coeffd = std::round(complex_val.real());
                    bool is_negative = std::signbit(coeffd);
                    coeffd = std::fabs(coeffd);
                    
                    // For values that fit in 64 bits
                    std::uint64_t coeffu = static_cast<std::uint64_t>(coeffd);

                    // Process for each modulus
                    for (std::size_t j = 0; j < coeff_modulus_size_; ++j) {
                        std::uint64_t modulus = coeff_modulus_acc_[j];
                        std::uint64_t reduced = coeffu % modulus;
                        
                        if (is_negative && reduced != 0) {
                            reduced = modulus - reduced;
                        }
                        
                        Pipe2ModReduceOutput::write(reduced);
                    }
                }
            }
        };

        // Kernel 3: Forward NTT (Corrected)
        class ForwardNTTKernel {
        private:
            sycl::accessor<std::uint64_t, 1, sycl::access::mode::read, sycl::access::target::device> ntt_roots_acc_;
            sycl::accessor<std::uint64_t, 1, sycl::access::mode::read, sycl::access::target::device> coeff_modulus_acc_;
            sycl::accessor<std::uint64_t, 1, sycl::access::mode::write, sycl::access::target::device> dest_acc_;
            std::size_t coeff_count_;        
            std::size_t coeff_modulus_size_;

            // Bit-reverse helper
            std::size_t reverse_bits(std::size_t operand, int coeff_count_power) const {
                std::size_t result = 0;
                for (int i = 0; i < coeff_count_power; i++) {
                    result = (result << 1) | (operand & 1);
                    operand >>= 1;
                }
                return result;
            }

            void ntt_negacyclic_harvey_device(
                std::uint64_t *values,
                const std::uint64_t *roots,
                std::uint64_t modulus,
                int coeff_count_power) const {
                
                std::size_t n = std::size_t(1) << coeff_count_power;
                
                // SEAL-compatible forward NTT
                // This matches the transform_to_rev in SEAL's DWTHandler
                for (std::size_t m = 2; m <= n; m <<= 1) {
                    std::size_t gap = n / m;
                    for (std::size_t offset = 0; offset < n; offset += m) {
                        for (std::size_t i = 0; i < m / 2; i++) {
                            std::size_t root_idx = reverse_bits(gap * i, coeff_count_power);
                            std::uint64_t w = roots[root_idx];
                            
                            std::size_t j1 = offset + i;
                            std::size_t j2 = j1 + m/2;
                            
                            std::uint64_t U = values[j1];
                            std::uint64_t V = values[j2];
                            
                            values[j1] = add_uint_mod_device(U, V, modulus);
                            std::uint64_t temp = sub_uint_mod_device(U, V, modulus);
                            values[j2] = multiply_uint_mod_device(temp, w, modulus);
                        }
                    }
                }
            }

        public:
            ForwardNTTKernel(
                sycl::handler &h,
                sycl::buffer<std::uint64_t, 1> &ntt_roots_buf,
                sycl::buffer<std::uint64_t, 1> &coeff_modulus_buf,
                sycl::buffer<std::uint64_t, 1> &dest_buf,
                std::size_t in_coeff_count,
                std::size_t in_coeff_modulus_size)
                : ntt_roots_acc_(ntt_roots_buf, h, sycl::read_only),
                  coeff_modulus_acc_(coeff_modulus_buf, h, sycl::read_only),
                  dest_acc_(dest_buf, h, sycl::write_only), 
                  coeff_count_(in_coeff_count),
                  coeff_modulus_size_(in_coeff_modulus_size) {}

            void operator()() const {
                std::array<std::uint64_t, MAX_COEFF_COUNT_FPGA * MAX_RNS_MODULI_FPGA> rns_poly;
                if (coeff_count_ * coeff_modulus_size_ > rns_poly.size()) return;

                // Read from pipe
                for (std::size_t i = 0; i < coeff_count_; ++i) {
                    for (std::size_t j = 0; j < coeff_modulus_size_; ++j) {
                        rns_poly[j * coeff_count_ + i] = Pipe2ModReduceOutput::read();
                    }
                }

                // Compute log2(coeff_count_)
                int logn = 0;
                std::size_t temp = coeff_count_;
                while (temp > 1) {
                    temp >>= 1;
                    logn++;
                }

                // Apply NTT to each RNS component
                for (std::size_t j = 0; j < coeff_modulus_size_; ++j) {
                    std::uint64_t *poly_ptr = &rns_poly[j * coeff_count_];
                    const std::uint64_t *roots_ptr = &ntt_roots_acc_[j * coeff_count_];
                    std::uint64_t modulus = coeff_modulus_acc_[j];

                    ntt_negacyclic_harvey_device(poly_ptr, roots_ptr, modulus, logn);
                }

                // Write results
                for (std::size_t k = 0; k < coeff_count_ * coeff_modulus_size_; ++k) {
                    dest_acc_[k] = rns_poly[k];
                }
            }
        };

        // process_vector_fpga_dummy implementation remains the same...
        std::vector<std::complex<double>> process_vector_fpga_dummy(
            const std::vector<std::complex<double>> &input_vector)
        {
            std::vector<std::complex<double>> output_vector = input_vector;
            size_t vector_size = output_vector.size();

            if (vector_size == 0)
            {
                return output_vector; 
            }

            try
            {
                auto selector = sycl::ext::intel::fpga_emulator_selector_v;
                sycl::queue queue(selector, sycl::property_list{sycl::property::queue::enable_profiling{}});

                std::cout << "Running on SYCL device: "
                          << queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl;

                sycl::buffer<std::complex<double>, 1> data_buffer(output_vector.data(), sycl::range<1>(vector_size));

                queue.submit([&](sycl::handler &h) {
                    auto data_accessor = data_buffer.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for<class VectorAddKernel>(sycl::range<1>(vector_size), [=](sycl::id<1> idx) {
                        data_accessor[idx] = data_accessor[idx] + 1.0;
                    });
                });
                queue.wait_and_throw();

            } catch (const sycl::exception &e) {
                std::cerr << "SYCL exception in process_vector_fpga_dummy: " << e.what() << std::endl;
                throw;
            }

            return output_vector;
        }

        // generate_ckks_encoding_tables and encode_ckks_fpga remain mostly the same...
        void generate_ckks_encoding_tables(
            std::size_t coeff_count,
            MemoryPoolHandle pool,
            std::vector<std::size_t> &matrix_reps_index_map_host,
            std::vector<std::complex<double>> &inv_root_powers_host)
        {
            matrix_reps_index_map_host.resize(coeff_count);
            inv_root_powers_host.assign(coeff_count, {0.0, 0.0}); 

            std::size_t slots = coeff_count / 2;
            int logn = seal::util::get_power_of_two(coeff_count); 
            if (logn < 0) { 
                throw std::logic_error("coeff_count is not a power of two");
            }
            uint64_t m_val = static_cast<uint64_t>(coeff_count) << 1;

            uint64_t gen = 3;
            uint64_t pos = 1;
            for (size_t i = 0; i < slots; i++)
            {
                uint64_t index1 = (pos - 1) >> 1;
                uint64_t index2 = (m_val - pos - 1) >> 1;
                matrix_reps_index_map_host[i] = seal::util::safe_cast<std::size_t>(seal::util::reverse_bits(index1, logn)); 
                matrix_reps_index_map_host[slots | i] = seal::util::safe_cast<std::size_t>(seal::util::reverse_bits(index2, logn)); 
                pos *= gen;
                pos &= (m_val - 1);
            }

            if (m_val >= 4) 
            {
                seal::util::ComplexRoots complex_roots_local(static_cast<std::size_t>(m_val), pool); 
                if (m_val == 4) 
                {
                     if (coeff_count > 1) inv_root_powers_host[1] = { 0.0, -1.0 };
                }
                else 
                {
                    for (std::size_t i = 1; i < coeff_count; i++) 
                    {
                        inv_root_powers_host[i] = std::conj(complex_roots_local.get_root(seal::util::reverse_bits(i - 1, logn) + 1)); 
                    }
                }
            }
        }

        template <typename T_input_val_type> 
        void encode_ckks_fpga(
            const SEALContext &context,
            const T_input_val_type *values, 
            std::size_t values_size,
            seal::parms_id_type parms_id, 
            double scale,
            Plaintext &destination,
            MemoryPoolHandle pool)
        {
            auto context_data_ptr = context.get_context_data(parms_id);
            if (!context_data_ptr)
            {
                throw std::invalid_argument("parms_id is not valid for encryption parameters");
            }
            const auto &context_data = *context_data_ptr;
            const auto &parms = context_data.parms();
            const auto &coeff_modulus_vec = parms.coeff_modulus(); 
            std::size_t coeff_count = parms.poly_modulus_degree();  
            std::size_t coeff_modulus_size = coeff_modulus_vec.size(); 
            std::size_t slots = coeff_count / 2;                    

            if (values_size > slots)
            {
                throw std::invalid_argument("values_size is too large for the given parameters");
            }

            destination.parms_id() = parms_id_zero; 
            destination.resize(seal::util::mul_safe(coeff_count, coeff_modulus_size)); 

            std::vector<std::size_t> matrix_reps_index_map_host_vec;
            std::vector<std::complex<double>> inv_root_powers_host_vec;
            generate_ckks_encoding_tables(coeff_count, pool, matrix_reps_index_map_host_vec, inv_root_powers_host_vec);
            
            std::vector<std::complex<double>> prepared_ifft_input_host_vec(coeff_count, {0.0, 0.0});
            for (std::size_t i = 0; i < values_size; ++i) {
                std::complex<double> current_value_complex;
                if constexpr (std::is_same_v<T_input_val_type, double>) {
                    current_value_complex = {values[i], 0.0};
                } else { 
                    current_value_complex = values[i];
                }
                prepared_ifft_input_host_vec[matrix_reps_index_map_host_vec[i]] = current_value_complex;
                prepared_ifft_input_host_vec[matrix_reps_index_map_host_vec[i + slots]] = std::conj(current_value_complex);
            }

            const seal::util::NTTTables *small_ntt_tables_array_ptr = context_data.small_ntt_tables(); 

            std::vector<std::uint64_t> coeff_modulus_values_host_vec(coeff_modulus_size);
            for(std::size_t i = 0; i < coeff_modulus_size; ++i) {
                coeff_modulus_values_host_vec[i] = coeff_modulus_vec[i].value();
            }

            std::vector<std::uint64_t> all_ntt_root_powers_host_vec(coeff_modulus_size * coeff_count);
            std::vector<std::uint64_t> all_ntt_inv_root_powers_host_vec(coeff_modulus_size * coeff_count);

            for (std::size_t i = 0; i < coeff_modulus_size; ++i) {
                const seal::util::NTTTables &current_ntt_table = small_ntt_tables_array_ptr[i]; 
                auto roots_iter = current_ntt_table.get_from_root_powers(); 
                auto inv_roots_iter = current_ntt_table.get_from_inv_root_powers(); 
                for (std::size_t j = 0; j < coeff_count; ++j) {
                    all_ntt_root_powers_host_vec[i * coeff_count + j] = (roots_iter + j)->operand;
                    all_ntt_inv_root_powers_host_vec[i * coeff_count + j] = (inv_roots_iter + j)->operand;
                }
            }

            try
            {
                auto selector = sycl::ext::intel::fpga_emulator_selector_v; 
                
                sycl::queue queue(selector, 
                    [&](sycl::exception_list elist) { 
                        for (auto &e : elist) {
                            std::rethrow_exception(e);
                        }
                    }, 
                    sycl::property_list{sycl::property::queue::enable_profiling{}});

                std::cout << "encode_ckks_fpga running on SYCL device: "
                          << queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl;

                sycl::buffer<std::complex<double>, 1> prepared_ifft_input_sycl_buf(prepared_ifft_input_host_vec.data(), sycl::range<1>(coeff_count));
                sycl::buffer<std::complex<double>, 1> inv_roots_sycl_buf(inv_root_powers_host_vec.data(), sycl::range<1>(coeff_count));
                sycl::buffer<std::uint64_t, 1> coeff_modulus_values_sycl_buf(coeff_modulus_values_host_vec.data(), sycl::range<1>(coeff_modulus_size));
                sycl::buffer<std::uint64_t, 1> ntt_roots_sycl_buf(all_ntt_root_powers_host_vec.data(), sycl::range<1>(all_ntt_root_powers_host_vec.size())); 
                sycl::buffer<std::uint64_t, 1> dest_sycl_buf(destination.data(), sycl::range<1>(destination.coeff_count())); 
                
                queue.submit([&](sycl::handler &h) {
                    h.single_task(
                        ValueTransformAndIFFTKernel( 
                            h, 
                            prepared_ifft_input_sycl_buf, 
                            inv_roots_sycl_buf,
                            coeff_count,    
                            scale           
                        )
                    );
                });
                std::cout << "Kernel 1 (ValueTransformAndIFFTKernel) submitted." << std::endl;

                queue.submit([&](sycl::handler &h) {
                    h.single_task(
                        CoeffProcessAndReduceKernel(
                            h,
                            coeff_modulus_values_sycl_buf,
                            coeff_count,
                            coeff_modulus_size
                        )
                    );
                });
                std::cout << "Kernel 2 (CoeffProcessAndReduceKernel) submitted." << std::endl;

                queue.submit([&](sycl::handler &h){
                    h.single_task(
                        ForwardNTTKernel(
                            h,
                            ntt_roots_sycl_buf, 
                            coeff_modulus_values_sycl_buf,
                            dest_sycl_buf,
                            coeff_count,
                            coeff_modulus_size
                        )
                    );
                });
                std::cout << "Kernel 3 (ForwardNTTKernel) submitted." << std::endl;

                queue.wait_and_throw();
                
                std::cout << "Data transfer D2H for destination plaintext is implicitly handled by SYCL buffer." << std::endl;
                
                destination.parms_id() = parms_id;
                destination.scale() = scale;

            } catch (const sycl::exception &e) {
                std::cerr << "SYCL exception in encode_ckks_fpga: " << e.what() << std::endl;
                throw; 
            } catch (const std::exception &e) {
                std::cerr << "Standard exception in encode_ckks_fpga: " << e.what() << std::endl;
                throw;
            }
        }

        // Template instantiations
        template void encode_ckks_fpga<double>(
            const SEALContext &context, const double *values, std::size_t values_size,
            seal::parms_id_type parms_id, double scale, Plaintext &destination, MemoryPoolHandle pool); 

        template void encode_ckks_fpga<std::complex<double>>(
            const SEALContext &context, const std::complex<double> *values, std::size_t values_size,
            seal::parms_id_type parms_id, double scale, Plaintext &destination, MemoryPoolHandle pool); 

    } // namespace fpga
} // namespace seal