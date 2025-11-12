#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cstdint>
#include <stddef.h>
#include <stdint.h>
#include <complex>

#define M_PI 3.14159265358979323846

typedef std::complex<double> complex_double;

// Input data structure for NTT RTL pipeline
typedef struct
{
    int32_t port_x_in_0;
    int32_t port_x_in_1;
    int32_t port_x_in_2;
    int32_t port_x_in_3;
} NTT_RTL_Input_Data;

// Output data structure for NTT RTL pipeline
typedef struct
{
    int32_t port_out_q_0;
    int32_t port_out_q_1;
    int32_t port_out_q_2;
    int32_t port_out_q_3;
} NTT_RTL_Output_Data;

// Define pipe capacities
constexpr size_t PIPE_CAPACITY = 4096;

// NTT RTL processing capacity - 4K points = 1024 structs of 4 elements each
constexpr size_t NTT_RTL_CAPACITY = 1024;

// Forward declare kernel names
class IFFTKernel;
class RTLNTTKernel_A_Input;
class RTLNTTKernel_A;
class RTLNTTKernel_A_Output;
class RTLNTTKernel_B_Input;
class RTLNTTKernel_B;
class RTLNTTKernel_B_Output;
class PolyMultNegNTTKernel;
class PolyAddModKernel;
class ScaleAndReduceKernel;

// Forward declarations for pipe name classes
class NTTAInputPipeName;
class NTTAModSelectorPipeName;
class NTTAOutputPipeName;
class NTTBInputPipeName;
class NTTBModSelectorPipeName;
class NTTBOutputPipeName;

// Pipe from IFFT to ScaleAndConvert kernel for transformed values
using IFFTToScaleAndReducePipe = sycl::ext::intel::pipe<class IFFTToScaleAndReducePipeID, std::complex<double>, PIPE_CAPACITY>;

// Pipe to pass error samples from IFFT to ScaleAndConvert kernel
using IFFTErrorToScaleAndReducePipe = sycl::ext::intel::pipe<class IFFTErrorToScaleAndReducePipeID, int8_t, PIPE_CAPACITY>;

// Pipe from ScaleAndConvertKernel to NTTKernel_1 for plaintext+error values
using ScaleReduceToNTTBPipe = sycl::ext::intel::pipe<class ScaleReduceToNTTBPipeID, uint32_t, PIPE_CAPACITY>;

// Pipe from NTTKernel_B to AddModKernel for NTT(PTE+error)
using NTTToAddModPipe = sycl::ext::intel::pipe<class NTTToAddModPipeID, uint32_t, PIPE_CAPACITY>;

// Pipe from NTTKernel_A to PolyMultNegNTTKernel for NTT(s)
using NTTToPolyMultNegPipe = sycl::ext::intel::pipe<class NTTToPolyMultNegPipeID, uint32_t, PIPE_CAPACITY>;

// Pipe from PolyMultNegNTTKernel to PolyAddModKernel for -(NTT(s)*c1)
using PolyMultNegToPolyAddModPipe = sycl::ext::intel::pipe<class PolyMultNegToPolyAddModPipeID, uint32_t, PIPE_CAPACITY>;

// Internal NTT A Pipeline Pipes (between the 3 NTT A stages)
using NTTAInputPipe = sycl::ext::intel::pipe<NTTAInputPipeName, NTT_RTL_Input_Data, NTT_RTL_CAPACITY>;
using NTTAModSelectorPipe = sycl::ext::intel::pipe<NTTAModSelectorPipeName, uint8_t, NTT_RTL_CAPACITY>;
using NTTAOutputPipe = sycl::ext::intel::pipe<NTTAOutputPipeName, NTT_RTL_Output_Data, NTT_RTL_CAPACITY>;

// Internal NTT B Pipeline Pipes (between the 3 NTT B stages)
using NTTBInputPipe = sycl::ext::intel::pipe<NTTBInputPipeName, NTT_RTL_Input_Data, NTT_RTL_CAPACITY>;
using NTTBModSelectorPipe = sycl::ext::intel::pipe<NTTBModSelectorPipeName, uint8_t, NTT_RTL_CAPACITY>;
using NTTBOutputPipe = sycl::ext::intel::pipe<NTTBOutputPipeName, NTT_RTL_Output_Data, NTT_RTL_CAPACITY>;

void pipeline(
    sycl::queue q,
    size_t n,
    size_t logn,
    double scale,
    uint32_t mod_value,
    uint32_t root,
    const uint32_t* const_ratio,
    sycl::buffer<std::complex<double>, 1>& encoding_buf,
    sycl::buffer<int8_t, 1>& error_samples_buf,
    sycl::buffer<uint32_t, 1>& ntt_pte_buf,
    sycl::buffer<uint32_t, 1>& c0_s_buf,
    sycl::buffer<uint32_t, 1>& c1_buf,
    sycl::buffer<uint32_t, 1>& s_save_buf
);

uint32_t NTT_root(size_t n, uint32_t mod_val);

// Modulus selector function
inline uint8_t get_rtl_modulus_selector(uint32_t mod_value) {
    switch(mod_value) {
        case 134012929:  return 0;  // root = 7470
        case 134111233:  return 1;  // root = 3856
        case 134176769:  return 2;  // root = 24149
        case 1053818881: return 3;  // root = 503422
        case 1054015489: return 4;  // root = 16768
        case 1054212097: return 5;  // root = 7305
        default:          return 0;  // fallback to first modulus
    }
}


/**
 * SYCL-accelerated combined encode and encrypt function for CKKS symmetric encryption.
 * This is the C interface to the SYCL implementation using only standard C types
 * with unpacked struct values.
 * 
 * @param n                  Polynomial degree
 * @param logn               Log of polynomial degree
 * @param scale              CKKS scale value
 * @param mod_value          Modulus value (q)
 * @param const_ratio        Pointer to modulus const_ratio array
 * @param encoding_buffer    Buffer containing encoded values
 * @param expanded_s         Expanded secret key buffer
 * @param uniform_poly       Uniform polynomial (c1) buffer
 * @param error_samples      Error samples buffer
 * @param pt_with_error      Buffer for plaintext + error
 * @param ntt_pte            Scratch space for NTT of plaintext+error
 * @param c0_s               Output: 1st ciphertext component
 * @param c1                 Output: 2nd ciphertext component
 * @param s_save             Optional: Save expanded s (for testing)
 * @param c1_save            Optional: Save c1 (for testing)
 */
extern "C" {
    void SYCL_combined_encrypt(
        /* parms related values */
        size_t n,                       // Polynomial degree
        size_t logn,                    // Log of polynomial degree
        double scale,                   // Scale value
        
        /* modulus related values */
        uint32_t mod_value,             // Modulus value (q)
        const uint32_t* const_ratio,    // Const ratio for Barrett reduction
        
        /* data buffers */
        complex_double* encoding_buffer, // Buffer for encoding
        uint32_t* expanded_s,           // Expanded secret key
        uint32_t* uniform_poly,         // Uniform polynomial (c1)
        int8_t* error_samples,          // Error samples
        uint32_t* ntt_pte,              // Scratch space for NTT
        uint32_t* c0_s,                 // Output: 1st ciphertext component
        uint32_t* c1,                   // Output: 2nd ciphertext component
        uint32_t* s_save,               // Optional: Save expanded s (for testing)
        uint32_t* c1_save               // Optional: Save c1 (for testing)
    );
} // extern "C"