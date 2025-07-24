/**
    SIMD helper functions for various bit lengths.
    NOTE: AVX-512 is required for some operations.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#define sse4_enabled 1
#define avx256_enabled 1
#define avx512_enabled 1
#define REQUIRE_SSE4                                                           \
    static_assert(sse4_enabled, "sse4 is required for this operation.");
#define REQUIRE_AVX256                                                         \
    static_assert(avx256_enabled, "avx256 is required for this operation.");
#define REQUIRE_AVX512                                                         \
    static_assert(avx512_enabled, "avx512 is required for this operation.");

#define ENSURE_LENGTH(BitLength)                                               \
    static_assert(BitLength == 64 || BitLength == 128 || BitLength == 256 ||   \
                      BitLength == 512,                                        \
                  "Unsupported bit length");

#define _inline inline __attribute__((always_inline))

template <unsigned short BitLength> struct avx_vec {
    static_assert(false, "Unsupported bit length");
};
template <> struct avx_vec<512> {
    REQUIRE_AVX512;
    using data_t = __m512i;
    static _inline auto load(const void* addr) {
        return _mm512_loadu_si512(addr);
    }
    static _inline void store(void* addr, const data_t& v) {
        _mm512_storeu_si512(addr, v);
    }
    static _inline bool is_zero(const data_t& v) {
        return _mm512_test_epi64_mask(v, v) == 0;
    }
    /// count logical 1 bits in packed 64-bit integers
    static _inline auto popcnt(const data_t& v) {
        return _mm512_popcnt_epi64(v);
    }
    static _inline auto or_op(const data_t& a, const data_t& b) {
        return _mm512_or_si512(a, b);
    }
    static _inline auto and_op(const data_t& a, const data_t& b) {
        return _mm512_and_si512(a, b);
    }
    /// returns a & ~b
    static _inline auto andnot_op(const data_t& a, const data_t& b) {
        // Intel® Intrinsics Guide:
        // Compute the bitwise NOT of 512 bits (representing integer data) in a
        // and then AND with b, and store the result in dst.
        return _mm512_andnot_si512(b, a);
    }
    static _inline auto add_op(const data_t& a, const data_t& b) {
        return _mm512_add_epi64(a, b);
    }
    static _inline long long int reduce_add(const data_t& v) {
        return _mm512_reduce_add_epi64(v);
    }
    static _inline bool eq_cmp(const data_t& a, const data_t& b) {
        return _mm512_cmpeq_epi64_mask(a, b) == 0xff;
    }
};
template <> struct avx_vec<256> {
    REQUIRE_AVX256;
    using data_t = __m256i;
    static _inline auto load(const void* addr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i_u*>(addr));
    }
    static _inline void store(void* addr, const data_t& v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i_u*>(addr), v);
    }
    static _inline bool is_zero(const data_t& v) {
        return _mm256_testz_si256(v, v) == 1;
    }
    static _inline auto popcnt(const data_t& v) {
        return _mm256_popcnt_epi64(v);
    }
    static _inline auto or_op(const data_t& a, const data_t& b) {
        return _mm256_or_si256(a, b);
    }
    static _inline auto and_op(const data_t& a, const data_t& b) {
        return _mm256_and_si256(a, b);
    }
    static _inline auto andnot_op(const data_t& a, const data_t& b) {
        return _mm256_andnot_si256(b, a);
    }
    static _inline auto add_op(const data_t& a, const data_t& b) {
        return _mm256_add_epi64(a, b);
    }
    static _inline long long int reduce_add(const data_t& v) {
        return _mm256_extract_epi64(v, 0) + _mm256_extract_epi64(v, 1) +
               _mm256_extract_epi64(v, 2) + _mm256_extract_epi64(v, 3);
    }
    static _inline bool eq_cmp(const data_t& a, const data_t& b) {
        REQUIRE_AVX512;
        return _mm256_cmpeq_epi64_mask(a, b) == 0xf;
    }
};
template <> struct avx_vec<128> {
    REQUIRE_SSE4;
    using data_t = __m128i;
    static _inline auto load(const void* addr) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(addr));
    }
    static _inline void store(void* addr, const data_t& v) {
        _mm_storeu_si128(reinterpret_cast<__m128i_u*>(addr), v);
    }
    static _inline bool is_zero(const data_t& v) {
        return _mm_testz_si128(v, v) == 1;
    }
    static _inline auto popcnt(const data_t& v) {
        return _mm_popcnt_epi64(v);
    }
    static _inline auto or_op(const data_t& a, const data_t& b) {
        return _mm_or_si128(a, b);
    }
    static _inline auto and_op(const data_t& a, const data_t& b) {
        return _mm_and_si128(a, b);
    }
    static _inline auto andnot_op(const data_t& a, const data_t& b) {
        return _mm_andnot_si128(b, a);
    }
    static _inline auto add_op(const data_t& a, const data_t& b) {
        return _mm_add_epi64(a, b);
    }
    static _inline uint64_t reduce_add(const data_t& v) {
        return _mm_extract_epi64(v, 0) + _mm_extract_epi64(v, 1);
    }
    static _inline bool eq_cmp(const data_t& a, const data_t& b) {
        REQUIRE_AVX512;
        return _mm_cmpeq_epi64_mask(a, b) == 0x3;
    }
};

/// Returns true if all bits are zero.
template <unsigned short BitLength> _inline bool testz(const void* addr) {
    const auto v = avx_vec<BitLength>::load(addr);
    return avx_vec<BitLength>::is_zero(v);
}

/// Count bits set(logical 1) in a contiguous memory region.
template <unsigned short BitLength>
_inline uint32_t popcnt(const void* addr, size_t count) {
    if (count == 0) return 0;
    const auto v0 = avx_vec<BitLength>::load(addr);
    auto c = avx_vec<BitLength>::popcnt(v0);
    const auto* cur_addr =
        reinterpret_cast<const typename avx_vec<BitLength>::data_t*>(addr);
    for (size_t i = 1; i < count; i++) {
        cur_addr++;
        const auto curv = avx_vec<BitLength>::load(cur_addr);
        const auto curc = avx_vec<BitLength>::popcnt(curv);
        c = avx_vec<BitLength>::add_op(c, curc);
    }
    return avx_vec<BitLength>::reduce_add(c);
}
template <> _inline uint32_t popcnt<64>(const void* addr, size_t size) {
    uint32_t result = 0;
    auto arr = reinterpret_cast<const uint64_t*>(addr);
    for (size_t i = 0; i < size; ++i, ++arr) {
        result += _mm_popcnt_u64(*arr);
    }
    return result;
}

/// Returns true if all bits in v2 are set in v1. (that is, v1 contains v2)
template <unsigned short BitLength>
_inline bool contains(const void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitLength>::load(addr1);
    const auto v2 = avx_vec<BitLength>::load(addr2);
    const auto and_result = avx_vec<BitLength>::and_op(v1, v2);
    return avx_vec<BitLength>::eq_cmp(and_result, v2); // v1 & v2 == v2
}

/// Returns true if v1 and v2 share any bits.
template <unsigned short BitLength>
_inline bool intersects(const void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitLength>::load(addr1);
    const auto v2 = avx_vec<BitLength>::load(addr2);
    const auto and_result = avx_vec<BitLength>::and_op(v1, v2);
    return !avx_vec<BitLength>::is_zero(and_result); // v1 & v2 != 0
}

template <unsigned short BitLength>
_inline bool cmpeq(const void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitLength>::load(addr1);
    const auto v2 = avx_vec<BitLength>::load(addr2);
    return avx_vec<BitLength>::eq_cmp(v1, v2);
}

_inline bool cmpeq(const uint64_t* addr1, const uint64_t* addr2,
                   const size_t size_i64) {
    size_t i = 0;
    for (; size_i64 >= i + 8; i += 8, addr1 += 8, addr2 += 8)
        if (!cmpeq<512>(addr1, addr2)) return false;
    if (size_i64 >= i + 4) {
        if (!cmpeq<256>(addr1, addr2)) return false;
        i += 4, addr1 += 4, addr2 += 4;
    }
    if (size_i64 >= i + 2) {
        if (!cmpeq<128>(addr1, addr2)) return false;
        i += 2, addr1 += 2, addr2 += 2;
    }
    if (size_i64 >= i + 1) {
        if (*addr1 != *addr2) return false;
        // no need to increment now
    }
    return true;
}

/// Bitwise OR operation on two memory regions,
/// stores the result in addr1.
/// Returns true if any bit in addr1 was changed.
template <unsigned short BitLength>
_inline bool or_inplace(void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitLength>::load(addr1);
    const auto v2 = avx_vec<BitLength>::load(addr2);
    const auto or_result = avx_vec<BitLength>::or_op(v1, v2);
    avx_vec<BitLength>::store(addr1, or_result);
    return !avx_vec<BitLength>::eq_cmp(v1,
                                       or_result); // changed := v1 != v1 | v2
}

struct ComposedChangeResult {
    bool changed;
    bool zeroed;
    ComposedChangeResult(bool changed, bool zeroed)
        : changed(changed), zeroed(zeroed) {}
    static ComposedChangeResult changed_to_zero() {
        return ComposedChangeResult(true, true);
    }
    static ComposedChangeResult not_zero(bool changed) {
        return ComposedChangeResult(changed, false);
    }
};

/// Bitwise AND operation on two memory regions,
/// stores the result in addr1.
/// Returns whether any bit in addr1 was changed and whether it was zeroed.
/// v1 and v2 are suppoesed to be both non-zero.
template <unsigned short BitLength>
_inline ComposedChangeResult and_inplace(void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitLength>::load(addr1);
    const auto v2 = avx_vec<BitLength>::load(addr2);
    const auto and_result = avx_vec<BitLength>::and_op(v1, v2);
    avx_vec<BitLength>::store(addr1, and_result);
    if (avx_vec<BitLength>::is_zero(and_result)) // changed to zero
        return ComposedChangeResult::changed_to_zero();
    else // changed := v1 != v1 & v2, zeroed :false
        return ComposedChangeResult::not_zero(
            !avx_vec<BitLength>::eq_cmp(v1, and_result));
}

template <unsigned short BitLength>
_inline ComposedChangeResult diff_inplace(void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitLength>::load(addr1);
    const auto v2 = avx_vec<BitLength>::load(addr2);
    const auto andnot_result = avx_vec<BitLength>::andnot_op(v1, v2);
    avx_vec<BitLength>::store(addr1, andnot_result);
    if (avx_vec<BitLength>::is_zero(andnot_result)) // changed to zero
        return ComposedChangeResult::changed_to_zero();
    else // changed := v1 != v1 & ~v2, zeroed :false
        return ComposedChangeResult::not_zero(
            !avx_vec<BitLength>::eq_cmp(v1, andnot_result));
}