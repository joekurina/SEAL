#ifndef IFFT_H
#define IFFT_H

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <complex>
#include <cmath> // For M_PI

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SYCL_ifft {
private:
    size_t n;
    size_t log_n;
    mutable sycl::buffer<std::complex<double>, 1> encoding_buf;

public:
    SYCL_ifft( size_t N_val, size_t logn_val, sycl::buffer<std::complex<double>, 1>& encoding_buf)
                : n(N_val), logn(logn_val), encoding_acc(encoding_buf) {}

    void operator()(sycl::handler& h) const {
        // get access to the buffer
        auto encoding_acc = encoding_buf.get_access<sycl::access::mode::read_write>(h);

        // Capture the log_n and n values
        size_t kernel_n = n;
        size_t kernel_log_n = log_n;

        h.single_task([=]() [[intel::kernel_args_restrict]] {
            // In-place bit-reverse reordering on encoding[0..n-1]
            {
                size_t N = kernel_n;
                size_t j = 0;
                for (size_t i = 1; i < N; ++i) {
                    size_t bit = N >> 1;
                    for (; j & bit; bit >>= 1) {
                        j ^= bit;
                    }
                    j |= bit;
                    if (i < j) {
                        // Swap encoding[i] and encoding[j]
                        auto temp       = encoding[i];
                        encoding[i]     = encoding[j];
                        encoding[j]     = temp;
                    }
                }
            }

            // Iterative Cooley–Tukey IFFT (unnormalized)
            for (int s = 1; s <= static_cast<int>(kernel_logn); ++s) {
                size_t m  = static_cast<size_t>(1) << s;   // m = 2^s
                size_t m2 = m >> 1;                        // m/2
                // Compute w_m_inv = e^{-2πi / m}
                double theta = -2.0 * M_PI / static_cast<double>(m);
                std::complex<double> w_m_inv(std::cos(theta), std::sin(theta));

                for (size_t k = 0; k < kernel_n; k += m) {
                    std::complex<double> w(1.0, 0.0);
                    for (size_t j = 0; j < m2; ++j) {
                        auto u = encoding[k + j];
                        auto t = w * encoding[k + j + m2];
                        encoding[k + j]       = u + t;
                        encoding[k + j + m2]  = u - t;
                        w *= w_m_inv;
                    }
                }
            }
        }); // end of single_task
    } // end of operator()
} // end of SYCL_ifft class

#endif // IFFT_H