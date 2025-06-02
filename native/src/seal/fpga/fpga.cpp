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
#include <sycl/ext/intel/ac_types/ac_complex.hpp> // For ac_complex, if used explicitly

#include <iostream>
#include <vector>
#include <complex>
#include <numeric>
#include <array>
#include <cmath> // For M_PI, cos, sin

namespace seal
{
    namespace fpga
    {
        // process_vector_fpga_dummy implementation (assumed to be correct and unchanged)
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

                std::cout << "Running dummy processing on SYCL device: "
                          << queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl;

                sycl::buffer<std::complex<double>, 1> data_buffer(output_vector.data(), sycl::range<1>(vector_size));

                queue.submit([&](sycl::handler &h) {
                    auto data_accessor = data_buffer.get_access<sycl::access::mode::read_write>(h);
                    h.parallel_for<class VectorAddKernel>(sycl::range<1>(vector_size), [=](sycl::id<1> idx) {
                        data_accessor[idx] = data_accessor[idx] + std::complex<double>(1.0, 0.0); // Ensure complex addition
                    });
                });
                queue.wait_and_throw();

            } catch (const sycl::exception &e) {
                std::cerr << "SYCL exception in process_vector_fpga_dummy: " << e.what() << std::endl;
                throw; // Re-throw to allow test framework to catch it
            }
            return output_vector;
        }

        // Updated process_vector_ifft_fpga with optional scaling and bit-reversal fix
        std::vector<std::complex<double>> process_vector_ifft_fpga(
            const std::vector<std::complex<double>> &input_vector,
            bool apply_scaling /* = true */) // Default value is in header
        {
            size_t n = input_vector.size();
            if (n == 0) { // Handle empty vector case first
                 std::cout << "Input vector for IFFT is empty. Returning empty vector." << std::endl;
                return {};
            }
            if ((n & (n - 1)) != 0) { // Check if n is a power of 2
                std::cerr << "Input vector size (" << n << ") must be a power of 2 for IFFT. Returning empty vector." << std::endl;
                return {};
            }

            std::vector<std::complex<double>> output_vector = input_vector;

            try
            {
                auto selector = sycl::ext::intel::fpga_emulator_selector_v; 
                sycl::queue queue(selector, sycl::property_list{sycl::property::queue::enable_profiling{}});

                std::cout << "Running IFFT (apply_scaling=" << apply_scaling << ") on SYCL device: "
                          << queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl;

                sycl::buffer<std::complex<double>, 1> data_buffer(output_vector.data(), sycl::range<1>(n));

                queue.submit([&](sycl::handler &h) {
                    auto data_accessor = data_buffer.get_access<sycl::access::mode::read_write>(h);
                    h.single_task<class IFFTKernelSingle>([=]() { 
                        // Bit-reversal permutation (in-place)
                        // This generates the bit-reversed indices iteratively.
                        size_t j = 0;
                        for (size_t i = 0; i < n; ++i) {
                            if (j > i) {
                                std::complex<double> temp_val = data_accessor[i];
                                data_accessor[i] = data_accessor[j];
                                data_accessor[j] = temp_val;
                            }
                            // Calculate next j (bit-reversed version of i+1)
                            size_t bit = n >> 1; 
                            // CRITICAL FIX: Add '&& bit > 0' to prevent infinite loop when bit becomes 0.
                            while (j >= bit && bit > 0) { 
                                j -= bit;       
                                bit >>= 1;      
                            }
                            j += bit;           
                        }

                        // Cooley-Tukey IFFT algorithm (iterative, Decimation In Time - DIT)
                        for (size_t len = 2; len <= n; len <<= 1) { 
                            double angle_step = 2.0 * M_PI / static_cast<double>(len);
                            std::complex<double> wlen_step(cos(angle_step), sin(angle_step)); 
                            for (size_t i = 0; i < n; i += len) {
                                std::complex<double> w(1.0, 0.0); 
                                for (size_t k = 0; k < len / 2; ++k) {
                                    std::complex<double> u = data_accessor[i + k];
                                    std::complex<double> v = data_accessor[i + k + len / 2] * w;

                                    data_accessor[i + k]           = u + v;
                                    data_accessor[i + k + len / 2] = u - v;
                                    w = w * wlen_step; 
                                }
                            }
                        }

                        // Optional scaling by 1/N for IFFT
                        if (apply_scaling) {
                            if (n > 0) { 
                                std::complex<double> scale_factor(1.0 / static_cast<double>(n), 0.0);
                                for (size_t i = 0; i < n; ++i) {
                                    data_accessor[i] = data_accessor[i] * scale_factor;
                                }
                            }
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
