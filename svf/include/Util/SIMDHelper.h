/**
    SIMD helper functions for various bit lengths.
    Use avx512 if available & applicable, otherwise fall back to avx2(256
    bits)/sse4(128 bits)/uint64(64 bits).
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <popcntintrin.h>

#define sse4_enabled 1
#define avx256_enabled 1
#define avx512_enabled 1

#define ENSURE_LENGTH(BitLength)                                               \
    static_assert(BitLength == 64 || BitLength == 128 || BitLength == 256 ||   \
                      BitLength == 512,                                        \
                  "Unsupported bit length");

/// Returns true if all bits are zero.
template <unsigned short BitLength> inline bool testz(const void* addr) {
    ENSURE_LENGTH(BitLength);
    if constexpr (BitLength == 512) {
        // AVX512
#if avx512_enabled
        __m512i v = _mm512_loadu_si512(addr);
        return _mm512_test_epi64_mask(v, v) == 0;
#else
        return testz<256>(addr) &&
               testz<256>(reinterpret_cast<const std::byte*>(addr) + 32);
#endif
    } else if constexpr (BitLength == 256) {
        // AVX2(avx256)
#if avx256_enabled
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(addr));
        return _mm256_testz_si256(v, v);
#else
        return testz<128>(addr) &&
               testz<128>(reinterpret_cast<const std::byte*>(addr) + 16);
#endif
    } else if constexpr (BitLength == 128) {
        // SSE4
#if sse4_enabled
        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(addr));
        return _mm_testz_si128(v, v);
#else
        return testz<64>(addr) &&
               testz<64>(reinterpret_cast<const std::byte*>(addr) + 8);
#endif
    } else // 64bit
        return *reinterpret_cast<const uint64_t*>(addr) == 0;
}

/// Count bits set(logical 1) in a contiguous memory region.
template <unsigned short BitLength>
inline uint32_t popcnt(const void* addr, size_t size) {
    ENSURE_LENGTH(BitLength);
    if (size == 0) return 0;

    /// only avx512 supports popcnt
    if constexpr (BitLength == 512 && avx512_enabled) {
        __m512i v0 = _mm512_loadu_si512(addr);
        __m512i c = _mm512_popcnt_epi64(v0);
        const __m512i* cur_addr = reinterpret_cast<const __m512i*>(addr);
        for (size_t i = 1; i < size; i++) {
            cur_addr++;
            __m512i curv = _mm512_loadu_si512(cur_addr);
            __m512i curc = _mm512_popcnt_epi64(curv);
            c = _mm512_add_epi64(c, curc);
        }
        return _mm512_reduce_add_epi64(c);
    }

    const uint64_t* addr64 = reinterpret_cast<const uint64_t*>(addr);
    const auto len = size * (BitLength / 64);
    uint32_t result = 0;
    for (size_t i = 0; i < len; i++, addr64++)
        result += _mm_popcnt_u64(*addr64);
    return result;
}

/// Returns true if all bits in v2 are set in v1. (that is, v1 contains v2)
template <unsigned short BitLength>
inline bool contains(const void* addr1, const void* addr2) {
    ENSURE_LENGTH(BitLength);
    static_assert(BitLength == 512 && avx512_enabled,
                  "FIXME: Implement for other bit length");
    __m512i v1 = _mm512_loadu_si512(addr1);
    __m512i v2 = _mm512_loadu_si512(addr2);
    __m512i r = _mm512_or_si512(v1, v2);
    __mmask8 cmp = _mm512_cmpeq_epi64_mask(r, v2);
    return cmp == 0xFF; // all bits in v2 are set in v1
}

/// Returns true if v1 and v2 share any bits.
template <unsigned short BitLength>
inline bool intersects(const void* addr1, const void* addr2) {
    ENSURE_LENGTH(BitLength);
    static_assert(BitLength == 512 && avx512_enabled,
                  "FIXME: Implement for other bit length");
    __m512i v1 = _mm512_loadu_si512(addr1);
    __m512i v2 = _mm512_loadu_si512(addr2);
    __m512i r = _mm512_and_si512(v1, v2);
    // the result is non-zero if they share any bits
    return _mm512_test_epi64_mask(r, r) != 0;
}

template <unsigned short BitLength>
inline bool cmpeq(const void* addr1, const void* addr2) {
    ENSURE_LENGTH(BitLength);
    static_assert(BitLength == 512 && avx512_enabled,
                  "FIXME: Implement for other bit length");
    __m512i v1 = _mm512_loadu_si512(addr1);
    __m512i v2 = _mm512_loadu_si512(addr2);
    return _mm512_cmpeq_epi64_mask(v1, v2) == 0xFF;
}

/// Bitwise OR operation on two memory regions,
/// stores the result in addr1.
/// Returns true if any bit in addr1 was changed.
template <unsigned short BitLength>
inline bool or_inplace(void* addr1, const void* addr2) {
    ENSURE_LENGTH(BitLength);
    static_assert(BitLength == 512 && avx512_enabled,
                  "FIXME: Implement for other bit length");
    __m512i v1 = _mm512_loadu_si512(addr1);
    __m512i v2 = _mm512_loadu_si512(addr2);
    __m512i r = _mm512_or_si512(v1, v2);
    __mmask8 unchanged = _mm512_cmpeq_epi64_mask(v1, r);
    _mm512_storeu_si512(addr1, r);
    return unchanged != 0xFF;
}

struct ComposedChangeResult {
    bool changed;
    bool zeroed;
    ComposedChangeResult(bool changed, bool zeroed)
        : changed(changed | zeroed), zeroed(zeroed) {}
};

/// Bitwise AND operation on two memory regions,
/// stores the result in addr1.
/// Returns whether any bit in addr1 was changed and whether it was zeroed.
/// v1 and v2 are suppoesed to be both non-zero.
template <unsigned short BitLength>
inline ComposedChangeResult and_inplace(void* addr1, const void* addr2) {
    ENSURE_LENGTH(BitLength);
    static_assert(BitLength == 512 && avx512_enabled,
                  "FIXME: Implement for other bit length");
    __m512i v1 = _mm512_loadu_si512(addr1);
    __m512i v2 = _mm512_loadu_si512(addr2);
    __m512i r = _mm512_and_si512(v1, v2);
    __mmask8 unchanged = _mm512_cmpeq_epi64_mask(v1, r);
    __mmask8 testz = _mm512_test_epi64_mask(r, r);
    _mm512_storeu_si512(addr1, r);
    return ComposedChangeResult(unchanged != 0xFF, testz == 0);
}

template <unsigned short BitLength>
inline ComposedChangeResult diff_inplace(void* addr1, const void* addr2) {
    ENSURE_LENGTH(BitLength);
    static_assert(BitLength == 512 && avx512_enabled,
                  "FIXME: Implement for other bit length");
    __m512i v1 = _mm512_loadu_si512(addr1);
    __m512i v2 = _mm512_loadu_si512(addr2);
    __m512i r = _mm512_andnot_epi64(v2, v1); // v1 & ~v2
    __mmask8 unchanged = _mm512_cmpeq_epi64_mask(v1, r);
    __mmask8 testz = _mm512_test_epi64_mask(r, r);
    _mm512_storeu_si512(addr1, r);
    return ComposedChangeResult(unchanged != 0xFF, testz == 0);
}