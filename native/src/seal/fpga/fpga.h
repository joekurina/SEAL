#ifndef SEAL_FPGA_H
#define SEAL_FPGA_H

#include <vector>
#include <complex>
#include "seal/plaintext.h"         // For seal::Plaintext
#include "seal/context.h"           // For seal::SEALContext
#include "seal/memorymanager.h"     // For seal::MemoryPoolHandle
#include "seal/util/defines.h"      // For seal::util::parms_id_type (used by seal::parms_id_type)
#include "seal/encryptionparams.h"  // For seal::parms_id_type definition

// SYCL specific headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // If using specific FPGA extensions

namespace seal
{
    namespace fpga
    {
        
    } // namespace fpga
} // namespace seal

#endif // SEAL_FPGA_H