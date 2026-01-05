// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Placeholder for future SYCL CKKS symmetric encryption kernel integration.
// This file is compiled as part of the FPGA test suite but the functionality
// is implemented in the modular kernel pipeline (fpga_pipeline.cpp and related).

#ifdef SEAL_USE_FPGA

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace seal
{
    namespace fpga
    {
        namespace sycl_ckks
        {
            // Placeholder - actual implementation uses the modular kernel pipeline
        }
    }
}

#endif // SEAL_USE_FPGA
