// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "fpga_kernel_types.h"

#ifdef SEAL_USE_FPGA
#include <sycl/sycl.hpp>
#include "fpga_pipes.h"
#endif

namespace seal
{
    namespace fpga
    {
#ifdef SEAL_USE_FPGA

        class ExitKernel;

        sycl::event submit_exit_kernel(
            sycl::queue& q,
            std::size_t* n_out,
            std::uint64_t* c0_out,
            std::uint64_t* c1_out);

#endif
    }
}
