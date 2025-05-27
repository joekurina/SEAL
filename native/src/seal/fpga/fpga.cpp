// native/src/seal/fpga/fpga.cpp

#include "seal/fpga/fpga.h"     // Assuming fpga.h is placed in seal/fpga/
#include "sycl/sycl.hpp"        // Main SYCL header
#include <iostream>             // For optional error messages
#include <vector>               // For std::vector
#include <complex>              // For std::complex
#include <sycl/ext/intel/fpga_extensions.hpp> // For Intel FPGAs

namespace seal
{
    namespace fpga
    {

        std::vector<std::complex<double>> process_vector_fpga_dummy(
            const std::vector<std::complex<double>> &input_vector)
        {
            // Create a copy of the input vector to be modified or use a new output vector
            std::vector<std::complex<double>> output_vector = input_vector;
            size_t vector_size = output_vector.size();

            if (vector_size == 0)
            {
                return output_vector; // Return empty or handle as an error
            }

            try
            {
                // Select a SYCL device.
                auto selector = sycl::ext::intel::fpga_emulator_selector_v;
                sycl::queue queue(selector, sycl::property_list{sycl::property::queue::enable_profiling{}});

                std::cout << "Running on SYCL device: "
                          << queue.get_device().get_info<sycl::info::device::name>()
                          << std::endl;

                // Create SYCL buffer
                sycl::buffer<std::complex<double>, 1> data_buffer(output_vector.data(), sycl::range<1>(vector_size));

                // Submit a command group to the queue
                queue.submit([&](sycl::handler &h) {
                    // Get an accessor to the buffer
                    auto data_accessor = data_buffer.get_access<sycl::access::mode::read_write>(h);

                    // Define the kernel
                    h.parallel_for<class VectorAddKernel>(sycl::range<1>(vector_size), [=](sycl::id<1> idx) {
                        // Add 1.0 to the real part of each complex number
                        data_accessor[idx] = data_accessor[idx] + 1.0;
                    });
                });

                // Wait for the queue to finish
                queue.wait_and_throw();

            } catch (const sycl::exception &e) {
                std::cerr << "SYCL exception in process_vector_fpga_dummy: " << e.what() << std::endl;
                throw;
            }

            return output_vector;
        }

    } // namespace fpga
} // namespace seal