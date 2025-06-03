// native/src/seal/fpga/fpga.cpp
#include "seal/fpga/fpga.h"
#include "seal/fpga/ifft.h" // Should contain CKKSInverseTransformKernel definition

// Essential SEAL headers
#include "seal/encryptionparams.h"
#include "seal/context.h" // Provides SEALContext and SEALContext::ContextData
#include "seal/util/common.h"
#include "seal/util/croots.h"
#include "seal/util/galois.h" // For reverse_bits
#include "seal/util/ntt.h"    // For SmallNTTTables and ntt_negacyclic_harvey
#include "seal/util/polycore.h" // For various poly operations
#include "seal/util/rns.h"      // For RNSTool
#include "seal/util/uintarithsmallmod.h" // For barrett_reduce etc.
#include "seal/memorymanager.h" // For MemoryManager::GetPool()

// SYCL specific headers
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // If using specific FPGA extensions
// #include <sycl/ext/intel/ac_types/ac_complex.hpp> // For ac_complex, if used explicitly by your kernel elsewhere

// Standard library includes
#include <cmath> // For M_PI, std::log2, std::fabs, std::round, std::fmod, std::ceil, std::signbit
#include <limits>
#include <stdexcept>
#include <vector> // Ensure std::vector is included
#include <complex> // Ensure std::complex is included
#include <algorithm> // For std::swap


namespace seal
{
    namespace fpga
    {
        

    } // namespace fpga
} // namespace seal
