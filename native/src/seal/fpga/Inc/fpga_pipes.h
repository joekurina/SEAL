// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#ifdef SEAL_USE_FPGA

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "fpga_kernel_types.h"

namespace seal
{
    namespace fpga
    {
        namespace pipes
        {
            using namespace sycl;

            class EntranceToDWT;
            class DWTToScaleReduce;
            class ScaleReduceToNTT;
            class NTTToEncrypt;
            class EncryptToExit;

            using EntranceToDWTPipe = ext::intel::pipe<EntranceToDWT, DWTPacket, 1>;
            using DWTToScaleReducePipe = ext::intel::pipe<DWTToScaleReduce, ScaleReducePacket, 1>;
            using ScaleReduceToNTTPipe = ext::intel::pipe<ScaleReduceToNTT, NTTPacket, 1>;
            using NTTToEncryptPipe = ext::intel::pipe<NTTToEncrypt, EncryptPacket, 1>;
            using EncryptToExitPipe = ext::intel::pipe<EncryptToExit, CiphertextPacket, 1>;

        } // namespace pipes
    } // namespace fpga
} // namespace seal

#endif // SEAL_USE_FPGA
