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

        class ScaleReduceKernel;
        class NTTForwardKernel;

        sycl::event submit_scale_reduce_kernel(sycl::queue& q);
        sycl::event submit_ntt_forward_kernel(sycl::queue& q);

#endif
    }
}
