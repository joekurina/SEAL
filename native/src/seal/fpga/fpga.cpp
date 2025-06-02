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
constexpr int KERNEL1_TO_KERNEL2_PIPE_DEPTH = 16384; 
constexpr int KERNEL2_TO_KERNEL3_PIPE_DEPTH = 16384;
constexpr std::size_t MAX_RNS_MODULI_FPGA = 64;

// --- Device Utility Functions ---
inline std::uint64_t add_uint_mod_device(std::uint64_t operand1, std::uint64_t operand2, std::uint64_t modulus) {
    std::uint64_t sum = operand1 + operand2;
    return (sum >= modulus) ? (sum - modulus) : sum;
}

inline std::uint64_t sub_uint_mod_device(std::uint64_t operand1, std::uint64_t operand2, std::uint64_t modulus) {
    return (operand1 >= operand2) ? (operand1 - operand2) : (operand1 + modulus - operand2);
}

// Iterative "Russian Peasant" modular multiplication
inline std::uint64_t multiply_uint_mod_device(std::uint64_t operand1, std::uint64_t operand2, std::uint64_t modulus) {
    std::uint64_t result = 0;
    std::uint64_t current_operand1 = operand1;

    if (modulus == 0) {
        // Consider error handling or returning a specific value if modulus can be 0
        // For SEAL, moduli are always > 0.
        return 0; 
    }

    // In the context of NTT, operand1 and operand2 are typically already < modulus.
    // operand1 is a polynomial coefficient (e.g., U or V from butterfly).
    // operand2 is an NTT root (W_op), which is < modulus.
    // If current_operand1 could be >= modulus, it should be reduced first:
    // current_operand1 %= modulus; // or barrett_reduce_64_device if applicable

    while (operand2 > 0) {
        if (operand2 & 1) { // If the LSB of operand2 is 1
            result = add_uint_mod_device(result, current_operand1, modulus);
        }
        // Double current_operand1 modulo modulus
        current_operand1 = add_uint_mod_device(current_operand1, current_operand1, modulus); 
        operand2 >>= 1; // Right shift operand2
    }
    return result;
}

namespace seal
{
    namespace fpga
    {
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

        std::vector<std::complex<double>> process_vector_ifft_fpga(
            const std::vector<std::complex<double>> &input_vector)
        {
            size_t n = input_vector.size();
            if (n == 0 || (n & (n - 1)) != 0) { // Check if n is a power of 2
                std::cerr << "Input vector size must be a power of 2 and non-zero for IFFT." << std::endl;
                return {};
            }

            std::vector<std::complex<double>> output_vector = input_vector;

            try
            {
                auto selector = sycl::ext::intel::fpga_emulator_selector_v; // Or sycl::default_selector_v
                sycl::queue queue(selector, sycl::property_list{sycl::property::queue::enable_profiling{}});

                std::cout << "Running IFFT on SYCL device: "
                          << queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl;

                sycl::buffer<std::complex<double>, 1> data_buffer(output_vector.data(), sycl::range<1>(n));

                queue.submit([&](sycl::handler &h) {
                    auto data_accessor = data_buffer.get_access<sycl::access::mode::read_write>(h);

                    h.single_task<class IFFTKernel>([=]() {
                        // Bit-reversal permutation (in-place)
                        size_t j = 0;
                        for (size_t i = 0; i < n; ++i) {
                            if (j > i) {
                                std::complex<double> temp_val = data_accessor[i];
                                data_accessor[i] = data_accessor[j];
                                data_accessor[j] = temp_val;
                            }
                            size_t bit = n >> 1;
                            while (j >= bit) {
                                j -= bit;
                                bit >>= 1;
                            }
                            j += bit;
                        }

                        // Cooley-Tukey IFFT algorithm (iterative, DIT)
                        for (size_t len = 2; len <= n; len <<= 1) {
                            double angle_step = 2.0 * M_PI / len; // Inverse twiddle factor angle (positive)
                            std::complex<double> wlen_step(cos(angle_step), sin(angle_step));
                            for (size_t i = 0; i < n; i += len) {
                                std::complex<double> w(1.0, 0.0);
                                for (size_t k = 0; k < len / 2; ++k) {
                                    std::complex<double> u = data_accessor[i + k];
                                    std::complex<double> v = data_accessor[i + k + len / 2] * w;

                                    data_accessor[i + k] = u + v;
                                    data_accessor[i + k + len / 2] = u - v;
                                    w = w * wlen_step;
                                }
                            }
                        }

                        // Scale by 1/N for IFFT
                        std::complex<double> scale_factor(1.0 / n, 0.0);
                        for (size_t i = 0; i < n; ++i) {
                            data_accessor[i] = data_accessor[i] * scale_factor;
                        }
                    });
                });
                queue.wait_and_throw();

            } catch (const sycl::exception &e) {
                std::cerr << "SYCL exception in process_vector_ifft_fpga: " << e.what() << std::endl;
                throw;
            }

            return output_vector;
        }

    } // namespace fpga
} // namespace seal