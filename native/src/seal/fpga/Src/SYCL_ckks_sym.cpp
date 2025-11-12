#include "SYCL_ckks_sym.h"
#include "the_nwc_4k_ntt_sycl.hpp"
#include <iostream>
#include <vector>

using namespace sycl;

// IFFT Kernel
class IFFTKernel {
private:
    size_t n;
    size_t logn;
    mutable sycl::buffer<std::complex<double>, 1> encoding_acc;

public:
    IFFTKernel( size_t n_val, size_t logn_val,
                sycl::buffer<std::complex<double>, 1>& encoding_buf)
            :   n(n_val), logn(logn_val), 
                encoding_acc(encoding_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the buffers
        auto encoding = encoding_acc.get_access<sycl::access::mode::read_write>(h);

        // Capture kernel variables
        size_t kernel_n = n;
        size_t kernel_logn = logn;
        
        h.single_task<class IFFT>([=]() [[intel::kernel_args_restrict]] {

            // IFFT implementation 
            size_t tt = 1, h = kernel_n / 2;
            
            for (size_t i = 0; i < kernel_logn; i++, tt *= 2, h /= 2) 
            {
                for (size_t j = 0, kstart = 0; j < h; j++, kstart += 2 * tt) 
                {
                    // Compute bit-reversed index for twiddle factor
                    size_t br_input = h + j;
                    size_t br_numbits = kernel_logn;
                    size_t br_t = (((br_input & 0xaaaa) >> 1) | ((br_input & 0x5555) << 1));
                    br_t        = (((br_t & 0xcccc) >> 2) | ((br_t & 0x3333) << 2));
                    br_t        = (((br_t & 0xf0f0) >> 4) | ((br_t & 0x0f0f) << 4));
                    br_t        = (((br_t & 0xff00) >> 8) | ((br_t & 0x00ff) << 8));
                    size_t br = (br_numbits == 0) ? 0 : (br_t >> (16 - br_numbits));
                    
                    // Compute twiddle factor (conjugate of root of unity)
                    size_t twiddle_k = br;
                    size_t twiddle_m = kernel_n << 1;
                    double twiddle_angle = 2.0 * M_PI * static_cast<double>(twiddle_k) / static_cast<double>(twiddle_m);
                    std::complex<double> s(sycl::cos(twiddle_angle), -sycl::sin(twiddle_angle));
                    
                    for (size_t k = kstart; k < kstart + tt; k++) 
                    {
                        std::complex<double> u = encoding[k];
                        std::complex<double> v = encoding[k + tt];
                        encoding[k]      = u + v;
                        encoding[k + tt] = (u - v) * s;
                    }
                }
            } // End of IFFT computation

            // Pass the transformed values to the pipe
            for (size_t i = 0; i < kernel_n; i++)
            {
                // Write transformed encoding values to pipe
                IFFTToScaleAndReducePipe::write(encoding[i]);
            }
        }); // End of single_task
    } // End of operator()
}; // End of IFFTKernel class

// RTL NTT A Input Kernel
class RTLNTTKernel_A_Input {
private:
    size_t n;                                       // Number of elements to process
    uint8_t mod_sel;                               // Modulus selector
    mutable sycl::buffer<uint32_t, 1> vec_acc;      // Input buffer (secret key data)

public:
    // Constructor accepting input and save buffers
    RTLNTTKernel_A_Input(size_t n_val,
                         uint8_t mod_selector,
                         sycl::buffer<uint32_t, 1>& vec_buf)  // Input (secret key data)
                        : n(n_val),
                          mod_sel(mod_selector), 
                          vec_acc(vec_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for input buffer (read only)
        auto data = vec_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n;
        uint8_t kernel_mod_sel = mod_sel;

        h.single_task<class NTTAInput>([=]() [[intel::kernel_args_restrict]] {
            // Calculate number of structs needed (4 elements per struct)
            size_t num_structs = kernel_n / 4;

            // Process data in chunks of 4 elements
            for (size_t i = 0; i < num_structs; ++i) {
                // Read 4 consecutive elements from input buffer
                uint32_t elem_0 = data[i * 4 + 0];
                uint32_t elem_1 = data[i * 4 + 1];
                uint32_t elem_2 = data[i * 4 + 2];
                uint32_t elem_3 = data[i * 4 + 3];

                // Create RTL input data structure
                NTT_RTL_Input_Data rtl_input;
                rtl_input.port_x_in_0 = static_cast<int32_t>(elem_0);
                rtl_input.port_x_in_1 = static_cast<int32_t>(elem_1);
                rtl_input.port_x_in_2 = static_cast<int32_t>(elem_2);
                rtl_input.port_x_in_3 = static_cast<int32_t>(elem_3);
                
                // Write modulus selector to pipe
                NTTAModSelectorPipe::write(kernel_mod_sel);
                // Write to NTT A input pipe
                NTTAInputPipe::write(rtl_input);
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Input class

// RTL NTT A Main Kernel
class RTLNTTKernel_A {
private:

public:
    RTLNTTKernel_A() {}

    void operator()(sycl::handler& h) const {
        h.single_task<RTLNTTKernel_A>([=]() [[intel::kernel_args_restrict]] {
#ifdef FPGA_EMULATOR
            // Create RTL instance for emulator mode
            reg_test_verifyNTT_multi_DUT* instance = the_nwc_4k_ntt_new_instance();
#endif
            // Calculate number of structs to process (4K points = 1024 structs)
            size_t num_structs = NTT_RTL_CAPACITY;

            // Process data structures through the RTL
            [[intel::initiation_interval(1)]]
            while (1) {
                // Read from input pipe (non-blocking)
                bool input_valid = false;
                bool mod_sel_valid = false;
                NTT_RTL_Input_Data pipe_input = NTTAInputPipe::read(input_valid);
                uint8_t kernel_mod_selector = NTTAModSelectorPipe::read(mod_sel_valid);

                // If no valid input, exit the loop
                //if (!input_valid) break;

                // Prepare RTL input structure
                the_nwc_4k_ntt_input_t rtl_input;
                rtl_input.port_in_v_s = input_valid;  // Set valid flag based on pipe read
                rtl_input.port_in_c_s = kernel_mod_selector;  // Modulus selector
                rtl_input.port_x_in_0 = pipe_input.port_x_in_0;
                rtl_input.port_x_in_1 = pipe_input.port_x_in_1;
                rtl_input.port_x_in_2 = pipe_input.port_x_in_2;
                rtl_input.port_x_in_3 = pipe_input.port_x_in_3;

                // Call RTL function every iteration
#ifdef FPGA_EMULATOR
                the_nwc_4k_ntt_output_t rtl_output = the_nwc_4k_ntt(instance, rtl_input);
#else
                the_nwc_4k_ntt_output_t rtl_output = the_nwc_4k_ntt(rtl_input);
#endif

                // Check if RTL output is valid and write to output pipe
                if (rtl_output.port_out_v_s == 1) {
                    NTT_RTL_Output_Data pipe_output;
                    pipe_output.port_out_q_0 = rtl_output.port_out_q_0;
                    pipe_output.port_out_q_1 = rtl_output.port_out_q_1;
                    pipe_output.port_out_q_2 = rtl_output.port_out_q_2;
                    pipe_output.port_out_q_3 = rtl_output.port_out_q_3;

                    // Write to output pipe
                    NTTAOutputPipe::write(pipe_output);
                }
            }

#ifdef FPGA_EMULATOR
            // Clean up RTL instance for emulator mode
            the_nwc_4k_ntt_delete_instance(instance);
#endif
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A class

class RTLNTTKernel_A_Output {
private:
    size_t n;  // Number of elements to process
    sycl::buffer<uint32_t, 1>& save_acc;

public:
    RTLNTTKernel_A_Output(size_t n_val, sycl::buffer<uint32_t, 1>& save_buf) : n(n_val), save_acc(save_buf) {}

    void operator()(sycl::handler& h) const {
        // Accessor for save buffer (write only)
        auto s_save = save_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel lambda
        size_t kernel_n = n;

        h.single_task<class NTTAOutput>([=]() [[intel::kernel_args_restrict]] {
            // Calculate number of structs to process (4K points = 1024 structs)
            size_t num_structs = kernel_n / 4;

            // Process each output structure from RTL NTT A
            for (size_t i = 0; i < num_structs; ++i) {
                // Read from NTT A output pipe (blocking read)
                NTT_RTL_Output_Data rtl_output = NTTAOutputPipe::read();

                // Convert RTL output back to individual uint32_t values
                // and write them to the existing pipeline
                uint32_t elem_0 = static_cast<uint32_t>(rtl_output.port_out_q_0);
                uint32_t elem_1 = static_cast<uint32_t>(rtl_output.port_out_q_1);
                uint32_t elem_2 = static_cast<uint32_t>(rtl_output.port_out_q_2);
                uint32_t elem_3 = static_cast<uint32_t>(rtl_output.port_out_q_3);

                // Write each element to the existing NTTToPolyMultNegPipe
                // This maintains compatibility with the rest of the pipeline
                NTTToPolyMultNegPipe::write(elem_0);
                NTTToPolyMultNegPipe::write(elem_1);
                NTTToPolyMultNegPipe::write(elem_2);
                NTTToPolyMultNegPipe::write(elem_3);

                // Save the NTT(s) state to the provided buffer
                s_save[i * 4 + 0] = elem_0;
                s_save[i * 4 + 1] = elem_1;
                s_save[i * 4 + 2] = elem_2;
                s_save[i * 4 + 3] = elem_3;
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_A_Output class

// RTL NTT B Input Kernel
class RTLNTTKernel_B_Input {
private:
    size_t n;                   // Number of elements to process
    uint8_t mod_sel;            // Modulus selector

public:
    RTLNTTKernel_B_Input(size_t n_val, uint8_t mod_selector) : n(n_val), mod_sel(mod_selector)  {}

    void operator()(sycl::handler& h) const {
        // Capture necessary variables
        size_t kernel_n = n;
        uint8_t kernel_mod_sel = mod_sel;

        h.single_task<class NTTBINPUT>([=]() [[intel::kernel_args_restrict]] {
            // Calculate number of structs needed (4 elements per struct)
            size_t num_structs = kernel_n / 4;

            // Buffer to accumulate 4 elements before creating a struct
            uint32_t element_buffer[4];
            size_t buffer_index = 0;

            // Read individual elements from the existing pipeline
            for (size_t i = 0; i < kernel_n; ++i) {
                // Read from the existing ScaleReduceToNTTBPipe (blocking read)
                uint32_t pipe_element = ScaleReduceToNTTBPipe::read();

                // Accumulate elements in buffer
                element_buffer[buffer_index] = pipe_element;
                buffer_index++;

                // When we have 4 elements, create an RTL input struct
                if (buffer_index == 4) {
                    // Create RTL input data structure
                    NTT_RTL_Input_Data rtl_input;
                    rtl_input.port_x_in_0 = static_cast<int32_t>(element_buffer[0]);
                    rtl_input.port_x_in_1 = static_cast<int32_t>(element_buffer[1]);
                    rtl_input.port_x_in_2 = static_cast<int32_t>(element_buffer[2]);
                    rtl_input.port_x_in_3 = static_cast<int32_t>(element_buffer[3]);

                    // Write modulus selector to pipe
                    NTTBModSelectorPipe::write(kernel_mod_sel);
                    // Write to NTT B input pipe
                    NTTBInputPipe::write(rtl_input);

                    // Reset buffer
                    buffer_index = 0;
                }
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B_Input class

// RTL NTT B Main Kernel
class RTLNTTKernel_B {
private:

public:
    RTLNTTKernel_B() {}

    void operator()(sycl::handler& h) const {
        h.single_task<RTLNTTKernel_B>([=]() [[intel::kernel_args_restrict]] {
#ifdef FPGA_EMULATOR
            // Create RTL instance for emulator mode
            reg_test_verifyNTT_multi_DUT* instance = the_nwc_4k_ntt_new_instance();
#endif

            // Calculate number of structs to process (4K points = 1024 structs)
            size_t num_structs = NTT_RTL_CAPACITY;

            // Get RTL modulus selector for this modulus value
            //sycl::ext::oneapi::experimental::printf("NTTKernel_B: Processing modulus selector %u\n", kernel_mod_selector);

            // Process data structures through the RTL
            [[intel::initiation_interval(1)]]
            while (1) {
                // Read from input pipe (non-blocking)
                bool input_valid = false;
                bool mod_sel_valid = false;

                NTT_RTL_Input_Data pipe_input = NTTBInputPipe::read(input_valid);
                uint8_t kernel_mod_selector = NTTBModSelectorPipe::read(mod_sel_valid);

                // If no valid input, exit the loop
                //if (!input_valid) break;

                // Prepare RTL input structure (always, following RTL_EXAMPLE pattern)
                the_nwc_4k_ntt_input_t rtl_input;
                rtl_input.port_in_v_s = input_valid;  // Set valid flag based on pipe read
                rtl_input.port_in_c_s = kernel_mod_selector;  // Modulus selector
                rtl_input.port_x_in_0 = pipe_input.port_x_in_0;
                rtl_input.port_x_in_1 = pipe_input.port_x_in_1;
                rtl_input.port_x_in_2 = pipe_input.port_x_in_2;
                rtl_input.port_x_in_3 = pipe_input.port_x_in_3;

                // Call RTL function every iteration (following RTL_EXAMPLE pattern)
#ifdef FPGA_EMULATOR
                the_nwc_4k_ntt_output_t rtl_output = the_nwc_4k_ntt(instance, rtl_input);
#else
                the_nwc_4k_ntt_output_t rtl_output = the_nwc_4k_ntt(rtl_input);
#endif

                // Check if RTL output is valid and write to output pipe
                if (rtl_output.port_out_v_s == 1) {
                    NTT_RTL_Output_Data pipe_output;
                    pipe_output.port_out_q_0 = rtl_output.port_out_q_0;
                    pipe_output.port_out_q_1 = rtl_output.port_out_q_1;
                    pipe_output.port_out_q_2 = rtl_output.port_out_q_2;
                    pipe_output.port_out_q_3 = rtl_output.port_out_q_3;

                    // Write to output pipe
                    NTTBOutputPipe::write(pipe_output);
                }
            }
#ifdef FPGA_EMULATOR
            // Clean up RTL instance for emulator mode
            the_nwc_4k_ntt_delete_instance(instance);
#endif
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B class

// RTL NTT B Output Kernel
class RTLNTTKernel_B_Output {
private:
    size_t n;                                     // Number of elements to process
    mutable sycl::buffer<uint32_t, 1> result_acc; // Result output buffer

public:
    RTLNTTKernel_B_Output(size_t n_val, sycl::buffer<uint32_t, 1>& result_buf)
        : n(n_val), result_acc(result_buf) {}

    void operator()(sycl::handler& h) const {
        // Get write access to the result buffer
        auto out_data_accessor = result_acc.get_access<sycl::access::mode::write>(h);

        // Capture necessary variables for the kernel
        size_t kernel_n = n;

        h.single_task<class NTTBOutput>([=]() [[intel::kernel_args_restrict]] {
            // Calculate number of structs to process (4K points = 1024 structs)
            size_t num_structs = kernel_n / 4;

            // Process each output structure from RTL NTT B
            for (size_t i = 0; i < num_structs; ++i) {
                // Read from NTT B output pipe (blocking read)
                NTT_RTL_Output_Data rtl_output = NTTBOutputPipe::read();

                // Convert RTL output back to individual uint32_t values
                uint32_t elem_0 = static_cast<uint32_t>(rtl_output.port_out_q_0);
                uint32_t elem_1 = static_cast<uint32_t>(rtl_output.port_out_q_1);
                uint32_t elem_2 = static_cast<uint32_t>(rtl_output.port_out_q_2);
                uint32_t elem_3 = static_cast<uint32_t>(rtl_output.port_out_q_3);

                // Write each element to the existing NTTToAddModPipe
                NTTToAddModPipe::write(elem_0);
                NTTToAddModPipe::write(elem_1);
                NTTToAddModPipe::write(elem_2);
                NTTToAddModPipe::write(elem_3);

                // Also write to the result buffer for output
                out_data_accessor[i * 4 + 0] = elem_0;
                out_data_accessor[i * 4 + 1] = elem_1;
                out_data_accessor[i * 4 + 2] = elem_2;
                out_data_accessor[i * 4 + 3] = elem_3;
            }
        }); // End single_task lambda
    } // End operator()
}; // End RTLNTTKernel_B_Output class

class PolyMultNegNTTKernel {
private:
    size_t n;
    uint32_t mod_value;
    const uint32_t* const_ratio; // For Barrett reduction in multiplication
    mutable sycl::buffer<uint32_t, 1> b_acc; // Input buffer

public:
    PolyMultNegNTTKernel(size_t n_val, uint32_t mod_val, const uint32_t* const_ratio_val,
                         sycl::buffer<uint32_t, 1>& b_buf) // In
        : n(n_val),
          mod_value(mod_val),
          const_ratio(const_ratio_val),
          b_acc(b_buf) {}

    void operator()(sycl::handler& h) const {
        // Get access to buffer
        auto b = b_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        h.single_task<class PolyMultNegate>([=]() [[intel::kernel_args_restrict]] {
            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get initial values
                uint32_t a_val = NTTToPolyMultNegPipe::read(); // Read from the pipe
                uint32_t b_val = b[i];

                // --- Step 1: Polynomial Multiplication (a_val * b_val) mod q ---
                // Logic copied directly from PolyMultNTTKernel with original formatting
                uint32_t mult_result;
                {
                    // 1. Multiply to get wide result
                    uint64_t res_temp = (uint64_t)a_val * (uint64_t)b_val;
                    uint32_t product[2];
                    product[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    product[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);

                    // 2. Barrett reduction starts here
                    // Round 1
                    uint32_t right_hw;
                    {
                        uint32_t res[2];
                        uint64_t rt_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[0];
                        res[0] = (uint32_t)(rt_temp & 0xFFFFFFFF);
                        res[1] = (uint32_t)((rt_temp >> 32) & 0xFFFFFFFF);
                        right_hw = res[1];
                    }

                    uint32_t middle_temp[2];
                    {
                        uint64_t mt_temp = (uint64_t)product[0] * (uint64_t)kernel_const_ratio[1];
                        middle_temp[0] = (uint32_t)(mt_temp & 0xFFFFFFFF);
                        middle_temp[1] = (uint32_t)((mt_temp >> 32) & 0xFFFFFFFF);
                    }

                    uint32_t middle_lw;
                    uint32_t middle_lw_carry;
                    {
                        middle_lw = right_hw + middle_temp[0];
                        middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                    }

                    uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                    // Round 2
                    uint32_t middle2_temp[2];
                    {
                        uint64_t mt2_temp = (uint64_t)product[1] * (uint64_t)kernel_const_ratio[0];
                        middle2_temp[0] = (uint32_t)(mt2_temp & 0xFFFFFFFF);
                        middle2_temp[1] = (uint32_t)((mt2_temp >> 32) & 0xFFFFFFFF);
                    }

                    uint32_t middle2_lw;
                    uint32_t middle2_lw_carry;
                    {
                        middle2_lw = middle_lw + middle2_temp[0];
                        middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                    }

                    uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;

                    uint32_t tmp = product[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                    // Barrett subtraction
                    tmp = product[0] - tmp * kernel_mod_val;

                    // Final reduction if needed
                    // Note: Original PolyMultNTTKernel used '>=' check here which is standard for Barrett.
                    // If result can be exactly 'q', this reduces it to 0.
                    int32_t is_ge_q = (int32_t)(tmp >= kernel_mod_val);
                    uint32_t mask_red = (uint32_t)(-is_ge_q);
                    mult_result = tmp - (kernel_mod_val & mask_red); // Store result of multiplication
                } // End of multiplication logic


                // --- Step 2: Polynomial Negation (-mult_result) mod q ---
                // Logic copied directly from PolyNegModKernel, applied to mult_result
                uint32_t neg_result;
                {
                    uint32_t coeff_to_negate = mult_result; // Use the multiplication result

                    // Compute if coefficient is non-zero
                    int32_t non_zero = (int32_t)(coeff_to_negate != 0);
                    uint32_t mask_neg = (uint32_t)(-non_zero);

                    // Compute negation: if coeff == 0, result = 0; else result = q - coeff
                    neg_result = (kernel_mod_val - coeff_to_negate) & mask_neg;
                } // End of negation logic

                // --- Step 3: Write the negated result to the pipe for further processing
                PolyMultNegToPolyAddModPipe::write(neg_result); // Write to the pipe
            } 
        }); // End of single_task
    } // End of operator()
}; // End of PolyMultNegNTTKernel class

// Merged Kernel: Performs Scaling/Conversion and Reduction
class ScaleAndReduceKernel 
{
private:
    size_t n;
    double scale;
    uint32_t mod_value;
    const uint32_t* const_ratio;
    mutable sycl::buffer<int8_t, 1> error_samples_acc;

public:
    // Constructor takes combined arguments
    ScaleAndReduceKernel(size_t n_val, double scale_val, uint32_t mod_val,
                         const uint32_t* const_ratio_val,
                         sycl::buffer<int8_t, 1>& error_samples_buf)
        : n(n_val),
          scale(scale_val),
          mod_value(mod_val),
          const_ratio(const_ratio_val), 
          error_samples_acc(error_samples_buf) {}

    void operator()(sycl::handler& h) const 
    {
        // Get access to the error samples buffer
        auto error_samples = error_samples_acc.get_access<sycl::access::mode::read>(h);

        // Capture necessary variables
        size_t kernel_n = n; 
        double kernel_scale = scale;
        uint32_t kernel_mod_val = mod_value;
        const uint32_t* kernel_const_ratio = const_ratio;

        h.single_task<class ScaleAndReduce>([=]() [[intel::kernel_args_restrict]] 
        {
            // --- Local array to buffer pipe data ---
            std::complex<double> local_encoded_data[PIPE_CAPACITY];

            // --- Processing Phase ---
            double n_inv = kernel_scale / static_cast<double>(kernel_n);

            for (size_t i = 0; i < kernel_n; i++) {
                std::complex<double> encoded_value = IFFTToScaleAndReducePipe::read(); 

                double real_val = encoded_value.real();
                double scaled = sycl::round(real_val * n_inv);
                int64_t int_val = static_cast<int64_t>(scaled);
                int64_t intermediate_result = int_val + error_samples[i];

                int64_t val = intermediate_result;
                uint64_t coeff_abs = (val < 0) ? static_cast<uint64_t>(-val) : static_cast<uint64_t>(val);
                uint32_t mask = static_cast<uint32_t>(val < 0);

                uint32_t coeff_abs_vec[2];
                coeff_abs_vec[0] = static_cast<uint32_t>(coeff_abs & 0xFFFFFFFF);
                coeff_abs_vec[1] = static_cast<uint32_t>((coeff_abs >> 32) & 0xFFFFFFFF);

                uint32_t right_hw;
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio[0];
                    right_hw = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                uint32_t middle_temp[2];
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[0] * (uint64_t)kernel_const_ratio[1];
                    middle_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                uint32_t middle_lw;
                uint32_t middle_lw_carry;
                {
                    middle_lw = right_hw + middle_temp[0];
                    middle_lw_carry = (uint8_t)(middle_lw < right_hw);
                }
                uint32_t middle_hw = middle_temp[1] + middle_lw_carry;

                uint32_t middle2_temp[2];
                {
                    uint64_t res_temp = (uint64_t)coeff_abs_vec[1] * (uint64_t)kernel_const_ratio[0];
                    middle2_temp[0] = (uint32_t)(res_temp & 0xFFFFFFFF);
                    middle2_temp[1] = (uint32_t)((res_temp >> 32) & 0xFFFFFFFF);
                }
                uint32_t middle2_lw;
                uint32_t middle2_lw_carry;
                {
                    middle2_lw = middle_lw + middle2_temp[0];
                    middle2_lw_carry = (uint8_t)(middle2_lw < middle_lw);
                }
                uint32_t middle2_hw = middle2_temp[1] + middle2_lw_carry;
                uint32_t tmp = coeff_abs_vec[1] * kernel_const_ratio[1] + middle_hw + middle2_hw;

                tmp = coeff_abs_vec[0] - tmp * kernel_mod_val;

                uint32_t coeff_crt;
                {
                    int32_t is_2q = (int32_t)(tmp >= kernel_mod_val);
                    uint32_t tmp_mask = (uint32_t)(-is_2q);
                    coeff_crt = (uint32_t)(tmp) - (kernel_mod_val & tmp_mask);
                }

                uint32_t final_result = ((kernel_mod_val - coeff_crt) & (-mask)) + (coeff_crt & (mask - 1));

                ScaleReduceToNTTBPipe::write(final_result); 
            } // End of for loop
        }); // End single_task
    } // End operator()
}; // End of ScaleAndReduceKernel class

// Kernel for modular addition of two polynomials
class PolyAddModKernel {
private:
    size_t n;
    uint32_t mod_value;
    mutable sycl::buffer<uint32_t, 1> output_acc;

public:
    PolyAddModKernel(size_t n_val, uint32_t mod_val,
                        sycl::buffer<uint32_t, 1>& output_buf)
        : n(n_val), mod_value(mod_val), output_acc(output_buf) {}
    
    void operator()(sycl::handler& h) const {
        // Get access to the output buffer
        auto output = output_acc.get_access<sycl::access::mode::write>(h);
        
        // Capture necessary variables
        size_t kernel_n = n;
        uint32_t kernel_mod_val = mod_value;
        
        h.single_task<class PolynomialAddition>([=]() [[intel::kernel_args_restrict]] {
            
            // Process each coefficient
            for (size_t i = 0; i < kernel_n; i++) {
                // Get coefficients
                uint32_t coeff1 = PolyMultNegToPolyAddModPipe::read(); // Read from the pipe
                uint32_t coeff2 = NTTToAddModPipe::read(); // Read from the pipe
                
                // Add coefficients
                uint32_t sum = coeff1 + coeff2;
                
                // Reduce modulo q: 
                // If sum >= q, subtract q
                int32_t is_ge_q = (int32_t)(sum >= kernel_mod_val);
                uint32_t mask = (uint32_t)(-is_ge_q);
                uint32_t result = sum - (kernel_mod_val & mask);
                
                // Write the result to the output buffer
                output[i] = result;
            }
        }); // End of single_task
    } // End operator()
}; // End of PolyAddModKernel class



// Implementation of the C-compatible function (Main host interface)
extern "C" void SYCL_combined_encrypt(
    // --- CKKS Parameters ---
    size_t n,                       // Polynomial degree.
    size_t logn,                    // Base-2 logarithm of the polynomial degree.
    double scale,                   // CKKS scaling factor.

    // --- Modulus Information ---
    uint32_t mod_value,             // Current modulus prime (q).
    const uint32_t* const_ratio,    // Precomputed constant ratio for Barrett reduction modulo q.

    // --- Input Data Buffers (Host Pointers) ---
    complex_double* encoding_buffer, // Input: Buffer holding complex-encoded plaintext values (fed to IFFTKernel).
    uint32_t* expanded_s,           // Input: Expanded secret key polynomial 's'.
    uint32_t* uniform_poly,         // Input: Uniformly sampled polynomial 'a' (becomes ciphertext component c1).
    int8_t* error_samples,          // Input: Buffer holding pre-sampled noise/error values.

    // --- Input/Output & Scratch Buffers (Host Pointers) ---
    uint32_t* ntt_pte,              // Intermediate/Scratch: Output buffer for ScaleAndReduce, then Input/Output buffer for NTTKernel_1. Holds NTT(plaintext + error).
    uint32_t* c0_s,                 // Input/Output: Starts with expanded_s, used for NTT(s), then -(NTT(s)*c1), finally holds the resulting ciphertext component c0.
    uint32_t* c1,                   // Output: Destination for the uniform polynomial 'a', becomes ciphertext component c1.

    // --- Output Buffers for Testing (Host Pointers) ---
    uint32_t* s_save,               // Output: Destination buffer for saving the NTT(s) state from NTTKernel_2. NULL if not needed.
    uint32_t* c1_save               // Output: Destination buffer for saving the original uniform polynomial 'a'. NULL if not needed.
) {
    // Calculate the NTT root
    uint32_t root = NTT_root(n, mod_value);

    // Copy the pre-generated uniform polynomial 'a' into the host memory buffer 'c1'.
    // This buffer 'c1' will be associated with a SYCL buffer and also represents
    // the second component of the final ciphertext (c1 = a).
    std::memcpy(c1, uniform_poly, n * sizeof(uint32_t));

    // Check if the save buffer 'c1_save' was provided by the caller.
    if (c1_save != nullptr) {
        // If provided, copy the original uniform polynomial 'a' into 'c1_save'
        // for testing, before 'c1' might be used otherwise.
        std::memcpy(c1_save, uniform_poly, n * sizeof(uint32_t));
    }

    // Copy the pre-generated expanded secret key 's' into the host memory buffer 'c0_s'.
    // This buffer 'c0_s' will be associated with a SYCL buffer and used as the
    // primary working buffer for calculating the first ciphertext component c0 = [-a*s + m + e].
    std::memcpy(c0_s, expanded_s, n * sizeof(uint32_t));

    // Create SYCL buffers that associate host memory pointers with device-accessible objects.
    // Data will be implicitly managed (copied to/from device) by the SYCL runtime as needed by kernel accessors.

    // Buffer for the input complex-encoded plaintext values. Read by IFFTKernel.
    buffer<std::complex<double>, 1> encoding_buf(encoding_buffer, range(n));

    // Buffer for the input error samples. Read by IFFTKernel.
    buffer<int8_t, 1> error_samples_buf(error_samples, range(n));

    // Buffer used for intermediate storage and NTT of plaintext+error.
    // Written by ScaleAndReduceKernel, Read/Written by NTTKernel_1, Read by PolyAddModKernel.
    buffer<uint32_t, 1> ntt_pte_buf(ntt_pte, range(n));

    // Main working buffer for ciphertext component c0.
    // Written by PolyAddModKernel. Contains final c0 result at the end.
    buffer<uint32_t, 1> c0_s_buf(c0_s, range(n));

    // Input buffer holding the uniform polynomial 'a' (ciphertext component c1).
    // Read by PolyMultNegNTTKernel.
    buffer<uint32_t, 1> c1_buf(c1, range(n));

    // Buffer for optionally saving the NTT(s) state. Initialized to null/invalid range.
    buffer<uint32_t, 1> s_save_buf{nullptr, range(n)};

    // Check if the host requested saving the NTT(s) state (s_save pointer is not NULL).
    if (s_save != nullptr) {
         // If requested, create a valid SYCL buffer associated with the host s_save pointer.
         // This buffer will be written to by NTTKernel_2.
         s_save_buf = buffer<uint32_t, 1>(s_save, range(n));
    }

    // Create queue
#if FPGA_HARDWARE
    auto selector = ext::intel::fpga_selector_v;
#else
    auto selector = ext::intel::fpga_emulator_selector_v;
#endif
    queue q{selector, property::queue::enable_profiling()};

    // Execute the full pipeline
    pipeline(
        q,
        n,
        logn,
        scale,
        mod_value,
        root,
        const_ratio,
        encoding_buf,
        error_samples_buf,
        ntt_pte_buf,
        c0_s_buf,
        c1_buf,
        s_save_buf
    );

    // Results are now in c0_s and s_save host pointers (due to buffer destruction sync)
}

// Integrated pipeline function with modified NTTKernel_2 call
void pipeline(
    queue q,                                        // The SYCL queue for submitting kernels.
    size_t n,                                       // The polynomial degree.
    size_t logn,                                    // Base-2 logarithm of the polynomial degree.
    double scale,                                   // The CKKS scaling factor.
    uint32_t mod_value,                             // The modulus value (q).
    uint32_t root,                                  // The NTT root for the polynomial ring.
    const uint32_t* const_ratio,                    // Precomputed constant ratio for Barrett reduction modulo q.
    buffer<std::complex<double>, 1>& encoding_buf,  // Input buffer: Complex-encoded plaintext values.
    buffer<int8_t, 1>& error_samples_buf,           // Input buffer: Noise/error samples.
    buffer<uint32_t, 1>& ntt_pte_buf,               // MODIFIED ROLE: Now primarily the OUTPUT buffer for NTTKernel_1. Holds NTT(plaintext + error).
    buffer<uint32_t, 1>& c0_s_buf,                  // Main work buffer: Input is expanded_s, intermediate results include NTT(s) and -(NTT(s)*c1), final output is ciphertext component c0.
    buffer<uint32_t, 1>& c1_buf,                    // Input buffer: Uniform polynomial 'a' (ciphertext component c1). Read by PolyMultNeg.
    buffer<uint32_t, 1>& s_save_buf                 // Output buffer: Destination for saving the NTT(s) state from NTTKernel_2 if requested.
) {
    uint8_t modulus_selector = get_rtl_modulus_selector(mod_value);
    std::cout << "[Pipeline] Using modulus selector: " << static_cast<int>(modulus_selector) << " for modulus " << mod_value << std::endl;
    try {

        // Submit consumers first to avoid pipe deadlocks
        // RTL NTT A Output: reads from NTTAOutputPipe, writes to NTTToPolyMultNegPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_A_Output kernel(n, s_save_buf);
            kernel(h);
        });

        // RTL NTT A Main: reads from NTTAInputPipe, writes to NTTAOutputPipe
        //std::cout << "[HOST] About to submit RTLNTTKernel_A (infinite loop)" << std::endl;
        q.submit([&](handler &h) {
            RTLNTTKernel_A kernel{};
            kernel(h);
        });
        //std::cout << "[HOST] RTLNTTKernel_A submitted" << std::endl;

        // Submit RTL NTT A Input: reads from secret_key_input_buf, writes to NTTAInputPipe
        //std::cout << "[HOST] About to submit RTLNTTKernel_A_Input" << std::endl;
        q.submit([&](handler &h) {
            RTLNTTKernel_A_Input(n, modulus_selector, c0_s_buf)(h);
        });
        //std::cout << "[HOST] RTLNTTKernel_A_Input submitted" << std::endl;

        // Submit PolyMultNegNTTKernel (reads from NTTToPolyMultNegPipe)
        q.submit([&](handler &h) {
            PolyMultNegNTTKernel(n, mod_value, const_ratio, c1_buf)(h);
        });

        // Submit IFFTKernel (writes to pipes read by ScaleAndReduce)
        q.submit([&](handler &h) {
            IFFTKernel(n, logn, encoding_buf)(h);
        });

        // ScaleAndReduceKernel reads from IFFT pipes and writes to ScaleReduceToNTTBPipe
        //std::cout << "[HOST] About to submit ScaleAndReduceKernel" << std::endl;
        q.submit([&](handler &h) {
            ScaleAndReduceKernel(n, scale, mod_value, const_ratio, error_samples_buf)(h);
        });
        //std::cout << "[HOST] ScaleAndReduceKernel submitted" << std::endl;

        // Submit RTL NTT B Output: reads from NTTBOutputPipe, writes to NTTToAddModPipe and ntt_pte_buf
        q.submit([&](handler &h) {
            RTLNTTKernel_B_Output kernel(n, ntt_pte_buf);
            kernel(h);
        });

        // RTL NTT B Main: reads from NTTBInputPipe, writes to NTTBOutputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_B kernel{};
            kernel(h);
        });

        // Submit RTL NTT B Input: reads from ScaleReduceToNTTBPipe, writes to NTTBInputPipe
        q.submit([&](handler &h) {
            RTLNTTKernel_B_Input kernel(n, modulus_selector);
            kernel(h);
        });

        // PolyAddModKernel reads from pipes written by PolyMultNeg and RTL B output - this is the final kernel
        q.submit([&](handler &h) {
            PolyAddModKernel(n, mod_value, c0_s_buf)(h);
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