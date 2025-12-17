#include "SYCL_ckks_sym.h"
#include "the_nwc_4k_ntt_sycl.hpp"
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

uint32_t NTT_root(std::size_t n, uint32_t mod_val)
{
    uint32_t root = 1;

    switch (n)
    {
        case 4096:
            switch (mod_val)
            {
                case 134012929u: root =  7470;  break;
                case 134111233u: root =  3856;  break;
                case 134176769u: root = 24149;  break;
                case 1053818881u: root = 503422; break;
                case 1054015489u: root = 16768;  break;
                case 1054212097u: root =  7305;  break;
                default:                        /* keep root = 1 */ ;
            }
            break;

        case 8192:
            switch (mod_val)
            {
                case 1053818881u: root = 374229; break;
                case 1054015489u: root = 123363; break;
                case 1054212097u: root =  79941; break;
                case 1055260673u: root =  38869; break;
                case 1056178177u: root = 162146; break;
                case 1056440321u: root =  81884; break;
                default:                        /* keep root = 1 */ ;
            }
            break;

        case 16384:
            switch (mod_val)
            {
                case 1053818881u: root =  13040;  break;
                case 1054015489u: root =    507;  break;
                case 1054212097u: root =   1595;  break;
                case 1055260673u: root =  68507;  break;
                case 1056178177u: root =   3073;  break;
                case 1056440321u: root =   6854;  break;
                case 1058209793u: root =  44467;  break;
                case 1060175873u: root =  16117;  break;
                case 1060700161u: root =  27607;  break;
                case 1060765697u: root = 222391;  break;
                case 1061093377u: root = 105471;  break;
                case 1062469633u: root = 310222;  break;
                case 1062535169u: root =   2005;  break;
                default:                        /* keep root = 1 */ ;
            }
            break;

        default:
            /* keep root = 1 */
            break;
    }

    return root;
} // End of NTT_root function