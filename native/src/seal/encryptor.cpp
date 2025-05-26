// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/encryptor.h"
#include "seal/modulus.h"
#include "seal/randomtostd.h"
#include "seal/util/common.h"
#include "seal/util/iterator.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/rlwe.h"
#include "seal/util/scalingvariant.h"
#include <algorithm>
#include <stdexcept>

using namespace std;
using namespace seal::util;

namespace seal
{
    Encryptor::Encryptor(const SEALContext &context, const PublicKey &public_key) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        set_public_key(public_key);

        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Quick sanity check
        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }
    }

    Encryptor::Encryptor(const SEALContext &context, const SecretKey &secret_key) : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        set_secret_key(secret_key);

        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Quick sanity check
        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }
    }

    Encryptor::Encryptor(const SEALContext &context, const PublicKey &public_key, const SecretKey &secret_key)
        : context_(context)
    {
        // Verify parameters
        if (!context_.parameters_set())
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }

        set_public_key(public_key);
        set_secret_key(secret_key);

        auto &parms = context_.key_context_data()->parms();
        auto &coeff_modulus = parms.coeff_modulus();
        size_t coeff_count = parms.poly_modulus_degree();
        size_t coeff_modulus_size = coeff_modulus.size();

        // Quick sanity check
        if (!product_fits_in(coeff_count, coeff_modulus_size, size_t(2)))
        {
            throw logic_error("invalid parameters");
        }
    }

    void Encryptor::encrypt_zero_internal(
        parms_id_type parms_id, bool is_asymmetric, bool save_seed, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // Verify parameters.
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw invalid_argument("parms_id is not valid for encryption parameters");
        }

        auto &context_data = *context_.get_context_data(parms_id);
        auto &parms = context_data.parms();
        size_t coeff_modulus_size = parms.coeff_modulus().size();
        size_t coeff_count = parms.poly_modulus_degree();
        bool is_ntt_form = false;

        if (parms.scheme() == scheme_type::ckks || parms.scheme() == scheme_type::bgv)
        {
            is_ntt_form = true;
        }
        else if (parms.scheme() != scheme_type::bfv)
        {
            throw invalid_argument("unsupported scheme");
        }

        // Resize destination and save results
        destination.resize(context_, parms_id, 2);

        // If asymmetric key encryption
        if (is_asymmetric)
        {
            auto prev_context_data_ptr = context_data.prev_context_data();
            if (prev_context_data_ptr)
            {
                // Requires modulus switching
                auto &prev_context_data = *prev_context_data_ptr;
                auto &prev_parms_id = prev_context_data.parms_id();
                auto rns_tool = prev_context_data.rns_tool();

                // Zero encryption without modulus switching
                Ciphertext temp(pool);
                util::encrypt_zero_asymmetric(public_key_, context_, prev_parms_id, is_ntt_form, temp);

                // Modulus switching
                SEAL_ITERATE(iter(temp, destination), temp.size(), [&](auto I) {
                    if (parms.scheme() == scheme_type::ckks)
                    {
                        rns_tool->divide_and_round_q_last_ntt_inplace(
                            get<0>(I), prev_context_data.small_ntt_tables(), pool);
                    }
                    // bfv switch-to-next
                    else if (parms.scheme() == scheme_type::bfv)
                    {
                        rns_tool->divide_and_round_q_last_inplace(get<0>(I), pool);
                    }
                    // bgv switch-to-next
                    else if (parms.scheme() == scheme_type::bgv)
                    {
                        rns_tool->mod_t_and_divide_q_last_ntt_inplace(
                            get<0>(I), prev_context_data.small_ntt_tables(), pool);
                    }
                    set_poly(get<0>(I), coeff_count, coeff_modulus_size, get<1>(I));
                });

                destination.parms_id() = parms_id;
                destination.is_ntt_form() = is_ntt_form;
                destination.scale() = temp.scale();
                destination.correction_factor() = temp.correction_factor();
            }
            else
            {
                // Does not require modulus switching
                util::encrypt_zero_asymmetric(public_key_, context_, parms_id, is_ntt_form, destination);
            }
        }
        else
        {
            // Does not require modulus switching
            util::encrypt_zero_symmetric(secret_key_, context_, parms_id, is_ntt_form, save_seed, destination);
        }
    }

    void Encryptor::encrypt_internal(
        const Plaintext &plain, bool is_asymmetric, bool save_seed, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // Minimal verification that the keys are set
        if (is_asymmetric)
        {
            if (!is_metadata_valid_for(public_key_, context_))
            {
                throw logic_error("public key is not set");
            }
        }
        else
        {
            if (!is_metadata_valid_for(secret_key_, context_))
            {
                throw logic_error("secret key is not set");
            }
        }

        // Verify that plain is valid
        if (!is_valid_for(plain, context_))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }

        auto scheme = context_.key_context_data()->parms().scheme();
        if (scheme == scheme_type::bfv)
        {
            if (plain.is_ntt_form())
            {
                throw invalid_argument("plain cannot be in NTT form");
            }

            encrypt_zero_internal(context_.first_parms_id(), is_asymmetric, save_seed, destination, pool);

            // Multiply plain by scalar coeff_div_plaintext and reposition if in upper-half.
            // Result gets added into the c_0 term of ciphertext (c_0,c_1).
            multiply_add_plain_with_scaling_variant(plain, *context_.first_context_data(), *iter(destination));
        }
        else if (scheme == scheme_type::ckks)
        {
            if (!plain.is_ntt_form())
            {
                throw invalid_argument("plain must be in NTT form");
            }

            auto context_data_ptr = context_.get_context_data(plain.parms_id());
            if (!context_data_ptr)
            {
                throw invalid_argument("plain is not valid for encryption parameters");
            }
            encrypt_zero_internal(plain.parms_id(), is_asymmetric, save_seed, destination, pool);

            auto &parms = context_.get_context_data(plain.parms_id())->parms();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();

            // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
            ConstRNSIter plain_iter(plain.data(), coeff_count);
            RNSIter destination_iter = *iter(destination);
            add_poly_coeffmod(destination_iter, plain_iter, coeff_modulus_size, coeff_modulus, destination_iter);

            destination.scale() = plain.scale();
        }
        else if (scheme == scheme_type::bgv)
        {
            if (plain.is_ntt_form())
            {
                throw invalid_argument("plain cannot be in NTT form");
            }
            encrypt_zero_internal(context_.first_parms_id(), is_asymmetric, save_seed, destination, pool);

            auto &context_data = *context_.first_context_data();
            auto &parms = context_data.parms();
            auto &coeff_modulus = parms.coeff_modulus();
            size_t coeff_modulus_size = coeff_modulus.size();
            size_t coeff_count = parms.poly_modulus_degree();
            size_t plain_coeff_count = plain.coeff_count();
            uint64_t plain_upper_half_threshold = context_data.plain_upper_half_threshold();
            auto plain_upper_half_increment = context_data.plain_upper_half_increment();
            auto ntt_tables = iter(context_data.small_ntt_tables());

            // c_{0} = pk_{0}*u + p*e_{0} + M
            Plaintext plain_copy = plain;
            // Resize to fit the entire NTT transformed (ciphertext size) polynomial
            // Note that the new coefficients are automatically set to 0
            plain_copy.resize(coeff_count * coeff_modulus_size);
            RNSIter plain_iter(plain_copy.data(), coeff_count);
            if (!context_data.qualifiers().using_fast_plain_lift)
            {
                // Allocate temporary space for an entire RNS polynomial
                // Slight semantic misuse of RNSIter here, but this works well
                SEAL_ALLOCATE_ZERO_GET_RNS_ITER(temp, coeff_modulus_size, coeff_count, pool);

                SEAL_ITERATE(iter(plain_copy.data(), temp), plain_coeff_count, [&](auto I) {
                    auto plain_value = get<0>(I);
                    if (plain_value >= plain_upper_half_threshold)
                    {
                        add_uint(plain_upper_half_increment, coeff_modulus_size, plain_value, get<1>(I));
                    }
                    else
                    {
                        *get<1>(I) = plain_value;
                    }
                });

                context_data.rns_tool()->base_q()->decompose_array(temp, coeff_count, pool);

                // Copy data back to plain
                set_poly(temp, coeff_count, coeff_modulus_size, plain_copy.data());
            }
            else
            {
                // Note that in this case plain_upper_half_increment holds its value in RNS form modulo the
                // coeff_modulus primes.

                // Create a "reversed" helper iterator that iterates in the reverse order both plain RNS components and
                // the plain_upper_half_increment values.
                auto helper_iter = reverse_iter(plain_iter, plain_upper_half_increment);
                advance(helper_iter, -safe_cast<ptrdiff_t>(coeff_modulus_size - 1));

                SEAL_ITERATE(helper_iter, coeff_modulus_size, [&](auto I) {
                    SEAL_ITERATE(iter(*plain_iter, get<0>(I)), plain_coeff_count, [&](auto J) {
                        get<1>(J) =
                            SEAL_COND_SELECT(get<0>(J) >= plain_upper_half_threshold, get<0>(J) + get<1>(I), get<0>(J));
                    });
                });
            }
            // Transform to NTT domain
            ntt_negacyclic_harvey(plain_iter, coeff_modulus_size, ntt_tables);

            // The plaintext gets added into the c_0 term of ciphertext (c_0,c_1).
            RNSIter destination_iter = *iter(destination);
            add_poly_coeffmod(destination_iter, plain_iter, coeff_modulus_size, coeff_modulus, destination_iter);
        }
        else
        {
            throw invalid_argument("unsupported scheme");
        }
    }

    template <typename T, typename U> void Encryptor::encode_and_encrypt_ckks(
        const std::vector<T> &values, parms_id_type parms_id, double scale, Ciphertext &destination,
        MemoryPoolHandle pool) const
    {
        // --- Phase 1: Encoding (Adapted from CKKSEncoder::encode_internal) ---
        // Verify parameters for encoding.
        auto context_data_ptr = context_.get_context_data(parms_id);
        if (!context_data_ptr)
        {
            throw std::invalid_argument("parms_id is not valid for encryption parameters");
        }
        if (values.empty()) // Or other checks on values.size() if needed
        {
            throw std::invalid_argument("values cannot be empty");
        }
        if (values.size() > context_data_ptr->parms().poly_modulus_degree() / 2)
        {
            throw std::invalid_argument("values_size is too large");
        }
        if (!pool)
        {
            throw std::invalid_argument("pool is uninitialized");
        }

        auto &context_data = *context_data_ptr;
        auto &parms = context_data.parms();
        auto &coeff_modulus = parms.coeff_modulus();
        std::size_t coeff_modulus_size = coeff_modulus.size();
        std::size_t coeff_count = parms.poly_modulus_degree();
        std::size_t slots = coeff_count / 2;

        // Check scale
        if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count()))
        {
            throw std::invalid_argument("scale out of bounds");
        }

        // Create a CKKSEncoder instance to perform encoding (or replicate its relevant parts)
        // For simplicity here, we instantiate it. In a more optimized version,
        // you might pre-compute/store CKKSEncoder members if Encryptor is long-lived.
        CKKSEncoder encoder(context_); // This will re-initialize roots, etc.

        // Temporary Plaintext to hold the encoded values
        Plaintext encoded_plain(pool);
        // The CKKSEncoder's encode_internal method is private.
        // We'll replicate its core logic here.
        // Or, if you make a helper in CKKSEncoder public/friend, you can call it.

        // --- Start of CKKSEncoder::encode_internal adapted logic ---
        // From native/src/seal/ckks.h encode_internal
        std::size_t n = util::mul_safe(slots, std::size_t(2)); // n = coeff_count
        auto conj_values = util::allocate<std::complex<double>>(n, pool, 0);
        for (std::size_t i = 0; i < values.size(); i++)
        {
            conj_values[encoder.matrix_reps_index_map_[i]] = values[i];
            conj_values[encoder.matrix_reps_index_map_[i + slots]] = std::conj(values[i]);
        }

        double fix = scale / static_cast<double>(n);
        encoder.fft_handler_.transform_from_rev(conj_values.get(), util::get_power_of_two(n), encoder.inv_root_powers_.get(), &fix);

        double max_coeff = 0;
        for (std::size_t i = 0; i < n; i++)
        {
            max_coeff = std::max<>(max_coeff, std::fabs(conj_values[i].real()));
        }
        int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max<>(max_coeff, 1.0)))) + 1;
        if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count())
        {
            throw std::invalid_argument("encoded values are too large for CKKS precision");
        }

        double two_pow_64 = std::pow(2.0, 64);
        encoded_plain.parms_id() = parms_id_zero; // Important before resize
        encoded_plain.resize(util::mul_safe(coeff_count, coeff_modulus_size));

        if (max_coeff_bit_count <= 64) // Simplified decomposition from CKKSEncoder
        {
            for (std::size_t i = 0; i < n; i++) // n is coeff_count here
            {
                double coeffd = std::round(conj_values[i].real());
                bool is_negative = std::signbit(coeffd);
                std::uint64_t coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));
                for (std::size_t j = 0; j < coeff_modulus_size; j++)
                {
                    encoded_plain[i + (j * coeff_count)] = is_negative ?
                        util::negate_uint_mod(util::barrett_reduce_64(coeffu, coeff_modulus[j]), coeff_modulus[j]) :
                        util::barrett_reduce_64(coeffu, coeff_modulus[j]);
                }
            }
        }
        else if (max_coeff_bit_count <= 128)
        {
            // From native/src/seal/ckks.h
            for (std::size_t i = 0; i < n; i++)
            {
                double coeffd = std::round(conj_values[i].real());
                bool is_negative = std::signbit(coeffd);
                coeffd = std::fabs(coeffd);

                std::uint64_t temp_coeffu[2]{ static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64)),
                                        static_cast<std::uint64_t>(coeffd / two_pow_64) };

                if (is_negative)
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        encoded_plain[i + (j * coeff_count)] = util::negate_uint_mod(
                            util::barrett_reduce_128(temp_coeffu, coeff_modulus[j]), coeff_modulus[j]);
                    }
                }
                else
                {
                    for (std::size_t j = 0; j < coeff_modulus_size; j++)
                    {
                        encoded_plain[i + (j * coeff_count)] = util::barrett_reduce_128(temp_coeffu, coeff_modulus[j]);
                    }
                }
            }
        }
        else // Slow case for larger coefficients
        {
            // This part involves RNS tool decomposition from CKKSEncoder::encode_internal
            // See native/src/seal/ckks.h for the full logic
            auto temp_coeffu(util::allocate_uint(coeff_modulus_size, pool));
            for (std::size_t i = 0; i < n; i++)
            {
                double coeffd = std::round(conj_values[i].real());
                bool is_negative = std::signbit(coeffd);
                coeffd = std::fabs(coeffd);

                util::set_zero_uint(coeff_modulus_size, temp_coeffu.get());
                auto temp_coeffu_ptr = temp_coeffu.get();
                while (coeffd >= 1)
                {
                    *temp_coeffu_ptr++ = static_cast<std::uint64_t>(std::fmod(coeffd, two_pow_64));
                    coeffd /= two_pow_64;
                }
                context_data.rns_tool()->base_q()->decompose(temp_coeffu.get(), pool);
                for (std::size_t j = 0; j < coeff_modulus_size; j++)
                {
                    encoded_plain[i + (j * coeff_count)] = is_negative ?
                        util::negate_uint_mod(temp_coeffu[j], coeff_modulus[j]) :
                        temp_coeffu[j];
                }
            }
        }

        // Transform to NTT domain as CKKS encryption expects NTT form plaintext
        auto ntt_tables = context_data.small_ntt_tables();
        for (std::size_t i = 0; i < coeff_modulus_size; i++)
        {
            util::ntt_negacyclic_harvey(encoded_plain.data() + (i * coeff_count), ntt_tables[i]);
        }
        encoded_plain.parms_id() = parms_id;
        encoded_plain.scale() = scale;
        // --- End of CKKSEncoder::encode_internal adapted logic ---


        // --- Phase 2: Encryption (Adapted from Encryptor::encrypt_internal for CKKS) ---
        // Minimal verification that the public key is set
        if (!is_metadata_valid_for(public_key_, context_))
        {
            throw std::logic_error("public key is not set for CKKS combined encode/encrypt");
        }

        // Encrypt zero first (this sets up the structure of the ciphertext)
        // For CKKS, the parms_id of the ciphertext will match the plaintext's.
        // Assuming is_asymmetric=true (public key encryption) and save_seed=false for a standard ciphertext
        encrypt_zero_internal(parms_id, true, false, destination, pool); //

        // Add the encoded plaintext to c0 of the ciphertext
        // The plaintext is already in NTT form
        ConstRNSIter plain_iter(encoded_plain.data(), coeff_count);
        RNSIter destination_c0_iter = *iter(destination); // Gives iterator to c0

        add_poly_coeffmod(destination_c0_iter, plain_iter, coeff_modulus_size, coeff_modulus, destination_c0_iter); //

        // Set the scale for the destination ciphertext
        destination.scale() = encoded_plain.scale(); //
    }

    // Explicit instantiation for T = double
    template void Encryptor::encode_and_encrypt_ckks<double,
        std::enable_if_t<
            std::is_same<std::remove_cv_t<double>, double>::value ||
            std::is_same<std::remove_cv_t<double>, std::complex<double>>::value
        >
    >(
        const std::vector<double> &values, parms_id_type parms_id, double scale, Ciphertext &destination,
        MemoryPoolHandle pool) const;

    // Explicit instantiation for T = std::complex<double>
    template void Encryptor::encode_and_encrypt_ckks<std::complex<double>,
        std::enable_if_t<
            std::is_same<std::remove_cv_t<std::complex<double>>, double>::value ||
            std::is_same<std::remove_cv_t<std::complex<double>>, std::complex<double>>::value
        >
    >(
        const std::vector<std::complex<double>> &values, parms_id_type parms_id, double scale, Ciphertext &destination,
        MemoryPoolHandle pool) const;
} // namespace seal
