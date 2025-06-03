// native/src/seal/fpga/fpga_encoder.cpp
#include "seal/fpga/fpga_encoder.h"
#include "seal/fpga/fft.h"
#include "seal/fpga/SYCL_ifft.h"
#include "seal/ckks.h"
#include <random>
#include <stdexcept>

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace fpga
    {
        FPGAEncoder::FPGAEncoder(const SEALContext &context) : context_(context)
        {
        
        }

        void FPGAEncoder::::encode_internal(double value, parms_id_type parms_id, double scale, Plaintext &destination, MemoryPoolHandle pool) const
        {
            // 1. Verify parameters.
            
            // 2. Build an N-length vector of complex numbers and fill it with zeros.
            
            // 3. Embed each slot and its conjugate into bit-reversed positions.
            
            // 4. Inverse FFT (no scaling inside)
            
            // 5. Normalize by (1/N) * (1/scale) in a separate pass.
            
            // 6. Round and RNS-decompose each of the N complex coefficients into Plaintext.
            
            // 7. Finalize the Plaintext object.
        }

        void FPGAEncoder::::encode_internal(int64_t value, parms_id_type parms_id, Plaintext &destination) const
        {

        }



    }// namespace fpga
}// namespace seal