/**
    SIMD helper functions for various bit lengths.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#define _inline inline __attribute__((always_inline))

_inline static uint16_t rol16(const uint16_t x, const int n) {
    return (x << n) | (x >> (16 - n));
}
_inline static uint16_t ror16(const uint16_t x, const int n) {
    return (x >> n) | (x << (16 - n));
}

/// Duplicate bits in 16-bit integer to 32-bit integer.
/// 0bxyz -> 0bxxyyzz
_inline uint32_t duplicate_bits(uint16_t from) {
    static_assert(__BMI2__, "BMI2 is required for this operation");
    const auto even_dep = _pdep_u32(from, 0xAAAAAAAA);
    const auto odd_dep = _pdep_u32(from, 0x55555555);
    return odd_dep | even_dep;
}

/// _mm512_2intersect_epi64 with emuated support.
/// Elements out of bounds are treated as maximum 64-bit integers.
/// "ne" means native or emulated. Requires AVX512F.
_inline void ne_mm512_2intersect_epi32(const __m512i& a, const __m512i& b,
                                       __mmask16& k1, __mmask16& k2) {
#if __AVX512_VP2INTERSECT__ && __AVX512VL__
    _mm512_2intersect_epi32(a, b, &k1, &k2);
#else // From https://arxiv.org/abs/2112.06342
    static_assert(__AVX512F__, "AVX512F is required for this operation");
    __m512i a1 = _mm512_alignr_epi32(a, a, 4);
    __m512i a2 = _mm512_alignr_epi32(a, a, 8);
    __m512i a3 = _mm512_alignr_epi32(a, a, 12);

    __m512i b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    __m512i b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    __m512i b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);

    uint16_t m00 = _mm512_cmpeq_epi32_mask(a, b);
    uint16_t m01 = _mm512_cmpeq_epi32_mask(a, b1);
    uint16_t m02 = _mm512_cmpeq_epi32_mask(a, b2);
    uint16_t m03 = _mm512_cmpeq_epi32_mask(a, b3);
    uint16_t m10 = _mm512_cmpeq_epi32_mask(a1, b);
    uint16_t m11 = _mm512_cmpeq_epi32_mask(a1, b1);
    uint16_t m12 = _mm512_cmpeq_epi32_mask(a1, b2);
    uint16_t m13 = _mm512_cmpeq_epi32_mask(a1, b3);
    uint16_t m20 = _mm512_cmpeq_epi32_mask(a2, b);
    uint16_t m21 = _mm512_cmpeq_epi32_mask(a2, b1);
    uint16_t m22 = _mm512_cmpeq_epi32_mask(a2, b2);
    uint16_t m23 = _mm512_cmpeq_epi32_mask(a2, b3);
    uint16_t m30 = _mm512_cmpeq_epi32_mask(a3, b);
    uint16_t m31 = _mm512_cmpeq_epi32_mask(a3, b1);
    uint16_t m32 = _mm512_cmpeq_epi32_mask(a3, b2);
    uint16_t m33 = _mm512_cmpeq_epi32_mask(a3, b3);

    k1 = m00 | m01 | m02 | m03 | rol16(m10 | m11 | m12 | m13, 4) |
         rol16(m20 | m21 | m22 | m23, 8) | ror16(m30 | m31 | m32 | m33, 4);

    uint16_t m_0 = m00 | m10 | m20 | m30;
    uint16_t m_1 = m01 | m11 | m21 | m31;
    uint16_t m_2 = m02 | m12 | m22 | m32;
    uint16_t m_3 = m03 | m13 | m23 | m33;

    k2 = m_0 | ((0x7777 & m_1) << 1) | ((m_1 >> 3) & 0x1111) |
         ((0x3333 & m_2) << 2) | ((m_2 >> 2) & 0x3333) | ((m_3 >> 1) & 0x7777) |
         ((m_3 & 0x1111) << 3);
#endif
}

/// _mm512_maskz_compress_epi16 with emulated support.
/// Requires AVX512F + AVX512DQ.
_inline __m512i ne_mm512_maskz_compress_epi16(const __mmask32& k,
                                              const __m512i& a) {
#if __AVX512_VBMI2__
    return _mm512_maskz_compress_epi16(k, a); // Latency:6 CPI:2
#else
    // TODO improve
    const auto low_as_u32 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(a));
    const auto low_compressed =
        _mm512_maskz_compress_epi32(k & 0xffff, low_as_u32);
    const auto high_as_u32 =
        _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(a, 1));
    const auto high_compressed =
        _mm512_maskz_compress_epi32((k >> 16) & 0xffff, high_as_u32);
    const auto ztest_mask =
        _mm512_test_epi32_mask(low_compressed, low_compressed);
    const auto nzero_count = 32 - _lzcnt_u32((uint32_t)ztest_mask);
    uint16_t temp[32]{};
    _mm512_mask_cvtepi32_storeu_epi16(temp, 0xffff, low_compressed);
    _mm512_mask_cvtepi32_storeu_epi16(temp + nzero_count, 0xffff,
                                      high_compressed);
    return _mm512_loadu_si512(temp);
#endif
}

template <unsigned short BitWidth> struct avx_vec {
    // static_assert(false, "Unsupported bit length");
};
template <> struct avx_vec<512> {
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
    using data_t = __m256i;
    static _inline auto load(const void* addr) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(addr));
    }
    static _inline void store(void* addr, const data_t& v) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(addr), v);
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
        return _mm256_cmpeq_epi64_mask(a, b) == 0xf;
    }
};
template <> struct avx_vec<128> {
    using data_t = __m128i;
    static _inline auto load(const void* addr) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(addr));
    }
    static _inline void store(void* addr, const data_t& v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(addr), v);
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
        return _mm_cmpeq_epi64_mask(a, b) == 0x3;
    }
};

/// Returns true if all bits are zero.
template <unsigned short BitWidth> _inline bool testz(const void* addr) {
    const auto v = avx_vec<BitWidth>::load(addr);
    return avx_vec<BitWidth>::is_zero(v);
}

/// Count bits set(logical 1) in a contiguous memory region.
template <unsigned short BitWidth>
_inline uint32_t popcnt(const void* addr, size_t count) {
    if (count == 0) return 0;
    const auto v0 = avx_vec<BitWidth>::load(addr);
    auto c = avx_vec<BitWidth>::popcnt(v0);
    const auto* cur_addr =
        reinterpret_cast<const typename avx_vec<BitWidth>::data_t*>(addr);
    for (size_t i = 1; i < count; i++) {
        cur_addr++;
        const auto curv = avx_vec<BitWidth>::load(cur_addr);
        const auto curc = avx_vec<BitWidth>::popcnt(curv);
        c = avx_vec<BitWidth>::add_op(c, curc);
    }
    return avx_vec<BitWidth>::reduce_add(c);
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
template <unsigned short BitWidth>
_inline bool contains(const void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitWidth>::load(addr1);
    const auto v2 = avx_vec<BitWidth>::load(addr2);
    const auto and_result = avx_vec<BitWidth>::and_op(v1, v2);
    return avx_vec<BitWidth>::eq_cmp(and_result, v2); // v1 & v2 == v2
}

/// Returns true if v1 and v2 share any bits.
template <unsigned short BitWidth>
_inline bool intersects(const void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitWidth>::load(addr1);
    const auto v2 = avx_vec<BitWidth>::load(addr2);
    const auto and_result = avx_vec<BitWidth>::and_op(v1, v2);
    return !avx_vec<BitWidth>::is_zero(and_result); // v1 & v2 != 0
}

template <unsigned short BitWidth>
_inline bool cmpeq(const void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitWidth>::load(addr1);
    const auto v2 = avx_vec<BitWidth>::load(addr2);
    return avx_vec<BitWidth>::eq_cmp(v1, v2);
}

_inline bool cmpeq(const uint64_t* addr1, const uint64_t* addr2,
                   const size_t size_i64) {
    size_t i = 0;
    for (; size_i64 >= i + 8; i += 8, addr1 += 8, addr2 += 8)
        if (!cmpeq<512>(addr1, addr2)) return false;
    // avoid branching
    const __mmask8 rest_count = size_i64 - i;
    const __mmask8 rest_eq_mask = _mm512_mask_cmpeq_epi64_mask(
        rest_count, _mm512_loadu_si512(addr1), _mm512_loadu_si512(addr2));
    return rest_count == rest_eq_mask;
}

/// Bitwise OR operation on two memory regions,
/// stores the result in addr1.
/// Returns true if any bit in addr1 was changed.
template <unsigned short BitWidth>
_inline bool or_inplace(void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitWidth>::load(addr1);
    const auto v2 = avx_vec<BitWidth>::load(addr2);
    const auto or_result = avx_vec<BitWidth>::or_op(v1, v2);
    avx_vec<BitWidth>::store(addr1, or_result);
    return !avx_vec<BitWidth>::eq_cmp(v1,
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
template <unsigned short BitWidth>
_inline ComposedChangeResult and_inplace(void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitWidth>::load(addr1);
    const auto v2 = avx_vec<BitWidth>::load(addr2);
    const auto and_result = avx_vec<BitWidth>::and_op(v1, v2);
    avx_vec<BitWidth>::store(addr1, and_result);
    if (avx_vec<BitWidth>::is_zero(and_result)) // changed to zero
        return ComposedChangeResult::changed_to_zero();
    else // changed := v1 != v1 & v2, zeroed :false
        return ComposedChangeResult::not_zero(
            !avx_vec<BitWidth>::eq_cmp(v1, and_result));
}

template <unsigned short BitWidth>
_inline ComposedChangeResult diff_inplace(void* addr1, const void* addr2) {
    const auto v1 = avx_vec<BitWidth>::load(addr1);
    const auto v2 = avx_vec<BitWidth>::load(addr2);
    const auto andnot_result = avx_vec<BitWidth>::andnot_op(v1, v2);
    avx_vec<BitWidth>::store(addr1, andnot_result);
    if (avx_vec<BitWidth>::is_zero(andnot_result)) // changed to zero
        return ComposedChangeResult::changed_to_zero();
    else // changed := v1 != v1 & ~v2, zeroed :false
        return ComposedChangeResult::not_zero(
            !avx_vec<BitWidth>::eq_cmp(v1, andnot_result));
}