/*--------------------------------------------------------------------------------------------
 - Origami: A High-Performance Mergesort Framework
 -
 - Copyright(C) 2021 Arif Arman, Dmitri Loguinov
 -
 - Produced via research carried out by the Texas A&M Internet Research Lab -
 - -
 - This program is free software : you can redistribute it and/or modify -
 - it under the terms of the GNU General Public License as published by -
 - the Free Software Foundation, either version 3 of the License, or -
 - (at your option) any later version. -
 - -
 - This program is distributed in the hope that it will be useful, -
 - but WITHOUT ANY WARRANTY; without even the implied warranty of -
 - MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the -
 - GNU General Public License for more details. -
 - -
 - You should have received a copy of the GNU General Public License -
 - along with this program. If not, see < http://www.gnu.org/licenses/>. -
 --------------------------------------------------------------------------------------------*/

#pragma once

#include <cstdint>
#include <immintrin.h>
using Reg = __m512i;
using Item = uint32_t;
using ui = uint32_t;

static inline void reverse(Reg& a0) {
    a0 = _mm512_permutexvar_epi32(
        _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        a0);
}

// SWAP
static inline void swap(Reg& a0, Reg& a1) {
    Reg vTmp = _mm512_min_epu32(a0, a1);
    a1 = _mm512_max_epu32(a0, a1);
    a0 = vTmp;
}

// MASK_MIN
static inline Reg mask_min(Reg a0, Reg a1, __mmask16 mask) {
    return _mm512_mask_min_epu32(a0, mask, a0, a1);
}

// MASK_MAX
static inline Reg mask_max(Reg a0, Reg a1, __mmask16 mask) {
    return _mm512_mask_max_epu32(a0, mask, a0, a1);
}

// SHUFFLE within REGISTER
template <ui bits> static inline void shuffle(Reg& a0) {
    static_assert(bits == 256 || bits == 128 || bits == 64 || bits == 32,
                  "Works only for 32, 64, 128 and 256 bits\n");
    if constexpr (bits == 256) a0 = _mm512_shuffle_i64x2(a0, a0, _MM_PERM_BADC);
    else if constexpr (bits == 128)
        a0 = _mm512_shuffle_i64x2(a0, a0, _MM_PERM_CDAB);
    else if constexpr (bits == 64) a0 = _mm512_shuffle_epi32(a0, _MM_PERM_BADC);
    else if constexpr (bits == 32) a0 = _mm512_shuffle_epi32(a0, _MM_PERM_CDAB);
}

static inline void sort_reg(Reg& a0) {
    // algorithm for 32-bit keys
    const ui mask1 = 0;
    const ui mask2 = 0xFFFF;
    Reg aux = a0;
    shuffle<256>(aux);

    a0 = mask_min(a0, aux, 0b0000000011111111 ^ mask1);
    a0 = mask_max(a0, aux, 0b0000000011111111 ^ mask2);

    aux = a0;
    shuffle<128>(aux);

    a0 = mask_min(a0, aux, 0b0000111100001111 ^ mask1);
    a0 = mask_max(a0, aux, 0b0000111100001111 ^ mask2);

    aux = a0;
    shuffle<64>(aux);

    a0 = mask_min(a0, aux, 0b0011001100110011 ^ mask1);
    a0 = mask_max(a0, aux, 0b0011001100110011 ^ mask2);

    aux = a0;
    shuffle<32>(aux);

    a0 = mask_min(a0, aux, 0b0101010101010101 ^ mask1);
    a0 = mask_max(a0, aux, 0b0101010101010101 ^ mask2);
}

inline void rswap(Reg& a0, Reg& a1) {
    reverse (a1);
    swap (a0, a1);
    sort_reg(a0);
    sort_reg(a1);
}