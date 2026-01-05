// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>

namespace seal
{
    namespace fpga
    {
        namespace arith
        {
            struct uint128_t
            {
                std::uint64_t lo;
                std::uint64_t hi;
            };

            inline uint128_t mul_u64(std::uint64_t a, std::uint64_t b)
            {
                std::uint64_t a_lo = a & 0xFFFFFFFF;
                std::uint64_t a_hi = a >> 32;
                std::uint64_t b_lo = b & 0xFFFFFFFF;
                std::uint64_t b_hi = b >> 32;

                std::uint64_t p0 = a_lo * b_lo;
                std::uint64_t p1 = a_lo * b_hi;
                std::uint64_t p2 = a_hi * b_lo;
                std::uint64_t p3 = a_hi * b_hi;

                std::uint64_t mid = p1 + (p0 >> 32);
                std::uint64_t carry = (mid < p1) ? 1ULL : 0ULL;
                mid += p2;
                carry += (mid < p2) ? 1ULL : 0ULL;

                uint128_t result;
                result.lo = (mid << 32) | (p0 & 0xFFFFFFFF);
                result.hi = p3 + (mid >> 32) + (carry << 32);
                return result;
            }

            inline std::uint64_t mod_u128(uint128_t val, std::uint64_t modulus)
            {
                if (val.hi == 0)
                {
                    return val.lo % modulus;
                }

                std::uint64_t result = 0;
                std::uint64_t base = 1;

                for (int i = 0; i < 64; i++)
                {
                    if ((val.lo >> i) & 1)
                    {
                        result += base;
                        if (result >= modulus) result -= modulus;
                    }
                    base <<= 1;
                    if (base >= modulus) base -= modulus;
                }

                for (int i = 0; i < 64; i++)
                {
                    if ((val.hi >> i) & 1)
                    {
                        result += base;
                        if (result >= modulus) result -= modulus;
                    }
                    base <<= 1;
                    if (base >= modulus) base -= modulus;
                }

                return result;
            }

            inline std::uint64_t mul_mod_fpga(std::uint64_t a, std::uint64_t b, std::uint64_t modulus)
            {
                uint128_t prod = mul_u64(a, b);
                return mod_u128(prod, modulus);
            }

            inline std::uint64_t mod_reduce(std::uint64_t value, std::uint64_t modulus)
            {
                return value >= modulus ? value - modulus : value;
            }

        } // namespace arith
    } // namespace fpga
} // namespace seal
