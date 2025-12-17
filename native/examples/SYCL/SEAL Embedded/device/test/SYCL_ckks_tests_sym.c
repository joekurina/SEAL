/**
@file SYCL_ckks_test_sym.c
*/

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h> // Needed for calloc, free
#include <string.h> // Needed for memset

#include "ckks_common.h"
#include "ckks_sym.h"
#include "ckks_tests_common.h"
#include "defines.h"
#include "fft.h" // Might need complex type definition if not included elsewhere
#include "fileops.h"
#include "ntt.h"
#include "polymodarith.h"
#include "polymodmult.h"
#include "sample.h"
#include "test_common.h"
#include "util_print.h"

// Newer Combined ENCODE + ENCRYPT test using individual allocations
void SYCL_test_ckks_sym_base(size_t n, size_t nprimes, bool test_message)
{
    Parms parms;
    parms.sample_s      = false;
    parms.is_asymmetric = false;
    parms.small_s       = true;
    bool encode_only    = false; 

    if (!parms.sample_s) se_assert(parms.small_s);

    // Pointers for individually allocated buffers
    complex_double *conj_vals   = NULL; // Used as encoding buffer
    ZZ *c0                      = NULL;
    ZZ *c1                      = NULL;
    uint16_t *index_map         = NULL; // Optional
    ZZ *ntt_roots               = NULL; // Optional
    ZZ *ntt_pte                 = NULL;
    ZZ *s                       = NULL; // Secret key buffer
    flpt *v                     = NULL; // Message buffer
    size_t vlen                 = n / 2;

    // -- Additional pointers required for testing (allocated separately as before)
    ZZ *s_test_save   = NULL;
    ZZ *c1_test_save  = NULL;
    ZZ *temp_test_mem = NULL;

    SE_PRNG prng;
    SE_PRNG shareable_prng;

    // --- Allocate Buffers Individually ---
    bool allocation_success = true;

    conj_vals = (complex_double *)calloc(n, sizeof(complex_double));
    if (!conj_vals) { allocation_success = false; goto cleanup; }

    c0 = (ZZ *)calloc(n, sizeof(ZZ));
    if (!c0) { allocation_success = false; goto cleanup; }

    c1 = (ZZ *)calloc(n, sizeof(ZZ));
    if (!c1) { allocation_success = false; goto cleanup; }

    ntt_pte = (ZZ *)calloc(n, sizeof(ZZ));
    if (!ntt_pte) { allocation_success = false; goto cleanup; }

    // Determine secret key buffer size (assuming ZZ type for buffer)
    size_t s_buf_size = parms.small_s ? (n / 16 + (n % 16 != 0)) : n; // Rough size for small_s, use n for expanded
    // Allocate slightly larger buffer for s if small form, or full ZZ * n if expanded needed
    s = (ZZ *)calloc(s_buf_size, sizeof(ZZ)); // Adjust size accurately if needed
    if (!s) { allocation_success = false; goto cleanup; }

    if (test_message) {
        v = (flpt *)calloc(vlen, sizeof(flpt));
        if (!v) { allocation_success = false; goto cleanup; }
    }

    // Allocate conditional buffers based on defines (mirror logic from ckks_set_ptrs_sym)
#if defined(SE_INDEX_MAP_PERSIST) || defined(SE_INDEX_MAP_LOAD) || \
    defined(SE_INDEX_MAP_LOAD_PERSIST) || \
    defined(SE_INDEX_MAP_LOAD_PERSIST_SYM_LOAD_ASYM)
    index_map = (uint16_t *)calloc(n, sizeof(uint16_t));
    if (!index_map) { allocation_success = false; goto cleanup; }
#endif

#if defined(SE_NTT_ONE_SHOT) || defined(SE_NTT_REG)
    size_t ntt_roots_size = n;
#elif defined(SE_NTT_FAST)
    size_t ntt_roots_size = 2 * n;
#else
    size_t ntt_roots_size = 0; // NTT OTF
#endif
    if (ntt_roots_size > 0) {
        ntt_roots = (ZZ *)calloc(ntt_roots_size, sizeof(ZZ));
        if (!ntt_roots) { allocation_success = false; goto cleanup; }
    }

    // Allocate test-specific buffers
    s_test_save = calloc(n, sizeof(ZZ));
    if (!s_test_save) { allocation_success = false; goto cleanup; }
    c1_test_save = calloc(n, sizeof(ZZ));
    if (!c1_test_save) { allocation_success = false; goto cleanup; }
    temp_test_mem = calloc(4 * n, sizeof(ZZ)); // Used by check_decode_decrypt_inpl
    if (!temp_test_mem) { allocation_success = false; goto cleanup; }

    // --- Setup Parameters ---
    // Ensure index_map is allocated above if needed by ckks_setup config.
    ckks_setup(n, nprimes, index_map, &parms);
    print_test_banner("Symmetric Encryption", &parms);

    // Setup secret key 's' (populates the 's' buffer)
    ckks_setup_s(&parms, NULL, &prng, s);
    // size_t s_size = parms.small_s ? n / 16 : n; // This was from original, use s_buf_size
    if (encode_only) clear(s, s_buf_size);

    // --- Run Tests ---
    for (size_t testnum = 0; testnum < 9; testnum++)
    {
        printf("-------------------- Test %zu -----------------------\n", testnum);
        ckks_reset_primes(&parms);

        // -- Set test values in 'v' buffer
        if (test_message)
        {
            set_encode_encrypt_test(testnum, vlen, v);
            print_poly_flpt("v        ", v, vlen);
        }
        else if (v) // Check if v was allocated (i.e., test_message was true at allocation time)
        {
             // If not testing with a message, use zeros (only if v exists)
             memset(v, 0, vlen * sizeof(flpt));
        }


        // -- Initialize PRNGs for this test run
        prng_randomize_reset(&shareable_prng, NULL);
        prng_randomize_reset(&prng, NULL);

        for (size_t i = 0; i < parms.nprimes; i++)
        {
            print_zz("\n ***** Modulus", parms.curr_modulus->value);

            // -- Call the combined encode/encrypt function using the allocated pointers
            // Note: The SYCL version (SYCL_combined_encrypt) is called *inside*
            // ckks_combined_encode_encrypt_sym. This test function tests the C wrapper.
            ckks_combined_encode_encrypt_sym(
                &parms,
                test_message ? v : NULL,
                vlen,
                &shareable_prng,
                &prng,
                s,          // Pass allocated secret key buffer
                ntt_pte,    // Pass allocated ntt_pte buffer
                ntt_roots,  // Pass allocated ntt_roots buffer (might be NULL)
                c0,         // Pass allocated c0 buffer (output)
                c1,         // Pass allocated c1 buffer (output)
                s_test_save,
                c1_test_save,
                conj_vals,  // Pass allocated conj_vals buffer (used for encoding)
                index_map   // Pass allocated index_map buffer (might be NULL)
            );

            // -- Check that decrypt gives back the pt+err and decode gives back v.
            bool s_test_save_small = false;
            check_decode_decrypt_inpl(c0, c1_test_save, v, vlen, s_test_save, s_test_save_small,
                                      ntt_pte, index_map, &parms, temp_test_mem);

            // -- Done checking this prime. Now try next prime if requested
            bool ret = ckks_next_prime_sym(&parms, s);
            se_assert(ret || (!ret && i + 1 == parms.nprimes));
        }

        // -- Can exit now if rlwe testing only
        if (!test_message) break;
    }

cleanup:
    // --- Cleanup ---
    if (!allocation_success) {
        printf("Aborting test due to memory allocation failure.\n");
    }

    free(conj_vals);
    free(c0);
    free(c1);
    free(ntt_pte);
    free(s);
    free(v); // Free v if it was allocated
    free(index_map); // Free index_map if it was allocated
    free(ntt_roots); // Free ntt_roots if it was allocated

    // Free test-specific buffers
    free(s_test_save);
    free(c1_test_save);
    free(temp_test_mem);

    delete_parameters(&parms);
}

/**
Full encode + symmetric encrypt test
@param[in] n        Polynomial ring degree
@param[in] nprimes  # of modulus primes
*/
void SYCL_test_ckks_encode_encrypt_sym(size_t n, size_t nprimes)
{
    printf("Beginning tests for ckks encode + symmetric encrypt (SYCL Test)...\n");
    bool test_message = 1;
    SYCL_test_ckks_sym_base(n, nprimes, test_message);
}

/**
Symmetric rlwe test only (message is the all-zeros vector)
@param[in] n        Polynomial ring degree
@param[in] nprimes  # of modulus primes
*/
void SYCL_test_enc_zero_sym(size_t n, size_t nprimes)
{
    printf("Beginning tests for rlwe symmetric encryption of 0 (SYCL Test)...\n");
    bool test_message = 0;
    SYCL_test_ckks_sym_base(n, nprimes, test_message);
}