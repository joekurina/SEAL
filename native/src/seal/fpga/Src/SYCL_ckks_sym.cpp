#include "SYCL_ckks_sym.h"
#include <iostream>
#include <vector>

using namespace sycl;


// Implementation of the C-compatible function (Main host interface)
extern "C" void SYCL_encrypt(

) {

    // Create queue
#if FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    queue q{selector, property::queue::enable_profiling()};

    // Execute the full pipeline
    pipeline(

    );

}

// Integrated pipeline function with modified NTTKernel_2 call
void pipeline(
   
) {
   
    try {
        // submit kernels
        q.submit([&](handler &h) {

        });


    } catch (std::exception const &e) { // Catch other standard exceptions
        std::cout << "[Pipeline] STANDARD EXCEPTION CAUGHT!" << std::endl;
        std::cerr << "Caught a standard exception in pipeline: "
                  << e.what() << std::endl;
        std::exit(1);
    }
} // End of pipeline function
