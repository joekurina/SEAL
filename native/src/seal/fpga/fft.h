#pragma once

#include <complex>
#include <cmath>
#include <cstddef>

namespace seal
{
    namespace fpga
    {
        /**
        Provides a simple in-place Cooley–Tukey FFT / IFFT implementation
        for arrays of std::complex<double> of length N = 2^log_n.

        This version **does not** apply any 1/N or additional scaling.
        It purely computes the forward or inverse discrete Fourier transform.

        Usage:
            seal::fpga::NormalFFTHandler fft;
            // Forward (DFT) on values[0..N-1]:
            fft.forward_fft(values, log_n);
            // Inverse (IDFT) on values[0..N-1]:
            fft.inverse_fft(values, log_n);
            // Later, if you need to divide by N or multiply by some factor,
            // do that in a separate step.
        */
        class NormalFFTHandler
        {
        public:
            NormalFFTHandler() = default;

            /**
            In-place forward FFT on `values[0..N-1]`, where N = 2^log_n.
            After this call, `values` contains the discrete Fourier transform
            under ω = e^{2πi / N}, in normal order.
            */
            void forward_fft(std::complex<double>* values, int log_n) const
            {
                const std::size_t N = std::size_t(1) << log_n;
                bit_reverse_reorder(values, log_n);

                for (int s = 1; s <= log_n; ++s)
                {
                    const std::size_t m  = std::size_t(1) << s;   // m = 2^s
                    const std::size_t m2 = m >> 1;                 // m/2
                    // w_m = e^{2πi / m}
                    const double theta = 2.0 * PI / double(m);
                    const std::complex<double> w_m(std::cos(theta), std::sin(theta));

                    for (std::size_t k = 0; k < N; k += m)
                    {
                        std::complex<double> w(1.0, 0.0);
                        for (std::size_t j = 0; j < m2; ++j)
                        {
                            const std::complex<double> t = w * values[k + j + m2];
                            const std::complex<double> u = values[k + j];
                            values[k + j]      = u + t;
                            values[k + j + m2] = u - t;
                            w *= w_m;
                        }
                    }
                }
            }

            /**
            In-place inverse FFT on `values[0..N-1]`, where N = 2^log_n.
            This computes the unnormalized IDFT under ω^{-1} = e^{-2πi / N}.
            After this call, `values` contains the inverse transform in normal order,
            but has **not** been divided by N.
            */
            void inverse_fft(std::complex<double>* values, int log_n) const
            {
                const std::size_t N = std::size_t(1) << log_n;
                bit_reverse_reorder(values, log_n);

                for (int s = 1; s <= log_n; ++s)
                {
                    const std::size_t m  = std::size_t(1) << s;
                    const std::size_t m2 = m >> 1;
                    // w_m_inv = e^{-2πi / m}
                    const double theta = -2.0 * PI / double(m);
                    const std::complex<double> w_m_inv(std::cos(theta), std::sin(theta));

                    for (std::size_t k = 0; k < N; k += m)
                    {
                        std::complex<double> w(1.0, 0.0);
                        for (std::size_t j = 0; j < m2; ++j)
                        {
                            const std::complex<double> t = w * values[k + j + m2];
                            const std::complex<double> u = values[k + j];
                            values[k + j]      = u + t;
                            values[k + j + m2] = u - t;
                            w *= w_m_inv;
                        }
                    }
                }
                // No division by N here; caller must normalize separately if needed.
            }

        private:
            static constexpr double PI = std::acos(-1.0);

            /**
            Performs an in-place bit-reversal reordering on `values[0..N-1]`,
            where N = 2^log_n. After this, the array is in bit-reversed order.
            */
            void bit_reverse_reorder(std::complex<double>* values, int log_n) const
            {
                const std::size_t N = std::size_t(1) << log_n;
                for (std::size_t i = 1, j = 0; i < N; ++i)
                {
                    std::size_t bit = N >> 1;
                    for (; j & bit; bit >>= 1)
                    {
                        j ^= bit;
                    }
                    j |= bit;
                    if (i < j)
                    {
                        std::swap(values[i], values[j]);
                    }
                }
            }
        };
    } // namespace fpga
} // namespace seal
