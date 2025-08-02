#pragma once

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <vector>

#include "SIMDHelper.h"

#define DO_WHILE0(code)                                                        \
    do {                                                                       \
        code                                                                   \
    } while (0);
#define REPEAT_2(identifier, start_from, block)                                \
    {                                                                          \
        constexpr int identifier = start_from;                                 \
        DO_WHILE0(block)                                                       \
    }                                                                          \
    {                                                                          \
        constexpr int identifier = start_from + 1;                             \
        DO_WHILE0(block)                                                       \
    }
#define REPEAT_4(identifier, start_from, block)                                \
    REPEAT_2(identifier, start_from, block)                                    \
    REPEAT_2(identifier, (start_from) + 2, block)
#define REPEAT_8(identifier, start_from, block)                                \
    REPEAT_4(identifier, start_from, block)                                    \
    REPEAT_4(identifier, (start_from) + 4, block)
#define REPEAT_i_2(block) REPEAT_2(i, 0, block)
#define REPEAT_i_4(block) REPEAT_4(i, 0, block)

/// Returns a __m512i that contains 32*16-bit integers in ascending order,
/// that is, {0, 1, 2, ..., 31} (from e0 to e31).
static inline auto get_asc_indexes() {
    return _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
                            18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                            4, 3, 2, 1, 0);
}

static inline size_t szudzik(size_t a, size_t b) {
    return a > b ? b * b + a : a * a + a + b;
}

namespace SVF {
template <uint16_t SegmentBits = 128> class SegmentBitVector {
    friend class SVFIRWriter;
    friend class SVFIRReader;

public:
    using UnitType = uint64_t;
    static constexpr size_t UnitBits = sizeof(UnitType) * 8;
    static constexpr size_t UnitsPerSegment = SegmentBits / UnitBits;

    struct alignas(uint64_t) Segment {
        UnitType data[UnitsPerSegment];

        Segment() = default;
        /// Initialize segment with only one bit set.
        Segment(size_t index) : data{} {
            set(index);
        }

        /// Returns true if all bits are zero.
        bool empty() const noexcept {
            return testz<SegmentBits>(data);
        }

        bool test(size_t index) const noexcept {
            const size_t unit_index = index / UnitBits;
            const size_t bit_index = index % UnitBits;
            return data[unit_index] & (static_cast<UnitType>(1) << bit_index);
        }

        void set(size_t index) noexcept {
            const size_t unit_index = index / UnitBits;
            const size_t bit_index = index % UnitBits;
            data[unit_index] |= (static_cast<UnitType>(1) << bit_index);
        }

        /// Returns true if the bit was not set before, and set it.
        bool test_and_set(size_t index) noexcept {
            const size_t unit_index = index / UnitBits;
            const size_t bit_index = index % UnitBits;
            const auto mask = static_cast<UnitType>(1) << bit_index;
            const bool prev_set = data[unit_index] & mask;
            data[unit_index] |= mask;
            return !prev_set; // return true if it was not set before
        }

        void reset(size_t index) noexcept {
            const size_t unit_index = index / UnitBits;
            const size_t bit_index = index % UnitBits;
            data[unit_index] &= ~(static_cast<UnitType>(1) << bit_index);
        }

        bool contains(const Segment& rhs) const noexcept {
            return ::contains<SegmentBits>(data, rhs.data);
        }

        bool intersects(const Segment& rhs) const noexcept {
            return ::intersects<SegmentBits>(data, rhs.data);
        }

        bool operator==(const Segment& rhs) const noexcept {
            return cmpeq<SegmentBits>(data, rhs.data);
        }

        bool operator!=(const Segment& rhs) const noexcept {
            return !(*this == rhs);
        }

        bool operator|=(const Segment& rhs) noexcept {
            return ::or_inplace<SegmentBits>(data, rhs.data);
        }

        ComposedChangeResult operator&=(const Segment& rhs) noexcept {
            return ::and_inplace<SegmentBits>(data, rhs.data);
        }

        ComposedChangeResult operator-=(const Segment& rhs) noexcept {
            return ::diff_inplace<SegmentBits>(data, rhs.data);
        }
    };
    static_assert(sizeof(Segment) == SegmentBits / 8);

    using index_t = uint32_t;
    // struct IndexedSegment {
    //     index_t index;
    //     Segment data;
    //     IndexedSegment() = default;
    //     template <typename... Args>
    //     IndexedSegment(index_t index, Args&&... seg)
    //         : index(index), data(std::forward<Args>(seg)...) {}
    // };

protected:
    std::vector<index_t> indexes;
    std::vector<Segment> data;

    // std::vector<IndexedSegment> values;
    /// Returns # of segments.
    inline __attribute__((always_inline)) size_t size() const noexcept {
        return indexes.size();
    }
    inline __attribute__((always_inline)) index_t
    index_at(size_t i) const noexcept {
        return indexes[i];
    }
    inline __attribute__((always_inline)) Segment& data_at(size_t i) noexcept {
        return data[i];
    }
    inline __attribute__((always_inline)) const Segment& data_at(
        size_t i) const noexcept {
        return data[i];
    }
    inline __attribute__((always_inline)) void erase_at(size_t i) noexcept {
        indexes.erase(indexes.begin() + i);
        data.erase(data.begin() + i);
    }
    inline __attribute__((always_inline)) void truncate(
        size_t keep_count) noexcept {
        indexes.resize(keep_count);
        data.resize(keep_count);
    }
    template <typename... Args>
    inline __attribute__((always_inline)) void emplace_at(
        size_t i, const index_t& ind, Args&&... seg) noexcept {
        indexes.emplace(indexes.begin() + i, ind);
        data.emplace(data.begin() + i, std::forward<Args>(seg)...);
    }
    inline __attribute__((always_inline)) void push_back_till_end(
        const SegmentBitVector& rhs, size_t other_offset) noexcept {
        indexes.insert(indexes.end(), rhs.indexes.begin() + other_offset,
                       rhs.indexes.end());
        data.insert(data.end(), rhs.data.begin() + other_offset,
                    rhs.data.end());
    }

public:
    class SegmentBitVectorIterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = index_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const index_t*;
        using reference = const index_t&;
        index_t cur_pos;

    protected:
        typename std::vector<index_t>::const_iterator indexIt;
        typename std::vector<index_t>::const_iterator indexEnd;
        typename std::vector<Segment>::const_iterator valueIt;
        unsigned char unit_index; // unit index in the current segment
        unsigned char bit_index;  // bit index in the current unit
        bool end;
        /// move to next unit, returns false if reached the end
        void incr_unit() {
            bit_index = 0;
            if (++unit_index == UnitsPerSegment) {
                // forward to next segment
                unit_index = 0;
                ++indexIt, ++valueIt;
                if (indexIt == indexEnd)
                    // reached the end
                    end = true;
            }
        }
        /// Increment the bit index (mark the current bit as visited).
        void incr_bit() {
            if (++bit_index == UnitBits) {
                // forward to next unit
                incr_unit();
            }
        }
        /// Start from current position and search for the next set bit
        void search() {
            while (!end) {
                auto mask = ~((static_cast<UnitType>(1) << bit_index) - 1);
                auto masked_unit = valueIt->data[unit_index] & mask;
                static_assert(
                    sizeof(UnitType) == 8,
                    "UnitType must be 64 bits, or _tzcnt_u64 can't be used");
                auto tz_count = _tzcnt_u64(masked_unit);
                if (tz_count < UnitBits) {
                    // found a set bit
                    bit_index = tz_count;
                    cur_pos = *indexIt + unit_index * UnitBits + bit_index;
                    return;
                } else // move to next unit
                    incr_unit();
            }
        }
        /// Step out from the current position and search for the next set bit.
        void forward() {
            incr_bit();
            search();
        }

    public:
        SegmentBitVectorIterator() = delete;
        SegmentBitVectorIterator(const SegmentBitVector& vec, bool end = false)
            : // must be init to identify raw vector
              indexEnd(vec.indexes.end()), end(end | vec.empty()) {
            if (end) return;
            indexIt = vec.indexes.begin();
            valueIt = vec.data.begin();
            unit_index = 0;
            bit_index = 0;
            search();
        }
        SegmentBitVectorIterator(const SegmentBitVectorIterator& other) =
            default;
        SegmentBitVectorIterator& operator=(
            const SegmentBitVectorIterator& other) = default;
        SegmentBitVectorIterator(SegmentBitVectorIterator&& other) noexcept =
            default;
        SegmentBitVectorIterator& operator=(
            SegmentBitVectorIterator&& other) noexcept = default;

        reference operator*() const {
            return cur_pos;
        }
        pointer operator->() const {
            return &cur_pos;
        }
        SegmentBitVectorIterator& operator++() {
            forward();
            return *this;
        }
        SegmentBitVectorIterator operator++(int) {
            SegmentBitVectorIterator temp = *this;
            ++*this;
            return temp;
        }
        bool operator==(const SegmentBitVectorIterator& other) const {
            return
                // created from the same SegmentBitVector, and
                indexEnd == other.indexEnd &&
                (( // both ended, or
                     end && other.end) ||
                 ( // both not ended and pointing to the same position
                     end == other.end && cur_pos == other.cur_pos));
        }
        bool operator!=(const SegmentBitVectorIterator& other) const {
            return !(*this == other);
        }
    };
    using iterator = SegmentBitVectorIterator;

    /// Returns an iterator to the beginning of this SegmentBitVector.
    /// NOTE: If you modify the vector after creating an iterator, the iterator
    /// is not stable and may cause UB if used.
    const SegmentBitVectorIterator begin() const {
        return SegmentBitVectorIterator(*this);
    }
    const SegmentBitVectorIterator end() const {
        return SegmentBitVectorIterator(*this, true);
    }

    /// Construct empty SegmentBitVector
    SegmentBitVector(void) {}

    /// Copy constructor
    SegmentBitVector(const SegmentBitVector& other) = default;

    /// Move constructor
    SegmentBitVector(SegmentBitVector&& other) noexcept = default;

    /// Copy assignment
    SegmentBitVector& operator=(const SegmentBitVector& other) = default;

    /// Move assignment
    SegmentBitVector& operator=(SegmentBitVector&& other) noexcept = default;

    /// Returns true if no bits are set.
    bool empty() const noexcept {
        return indexes.empty();
    }

    /// Returns the count of set bits.
    uint32_t count() const noexcept {
        // TODO: improve
        if (size() == 0) return 0;

#if __AVX512VPOPCNTDQ__ && __AVX512VL__
        auto it = values.begin();
        const auto v0 = avx_vec<SegmentBits>::load(&(it->data));
        auto c = avx_vec<SegmentBits>::popcnt(v0);
        ++it;
        for (; it != values.end(); ++it) {
            const auto curv = avx_vec<SegmentBits>::load(&(it->data));
            const auto curc = avx_vec<SegmentBits>::popcnt(curv);
            c = avx_vec<SegmentBits>::add_op(c, curc);
        }
        return avx_vec<SegmentBits>::reduce_add(c);
#else
        return popcnt<64>(this->data.data(), size() * (sizeof(Segment) / 8));
#endif
    }

    /// Empty the set.
    void clear() noexcept {
        indexes.clear();
        data.clear();
    }

    /// Returns true if n is in this set.
    bool test(uint32_t n) const noexcept {
        const auto target_ind = n - (n % SegmentBits);
        const auto low_pos =
            std::lower_bound(indexes.begin(), indexes.end(), target_ind);
        const auto i = std::distance(indexes.begin(), low_pos);
        if (low_pos == indexes.end() || *low_pos != target_ind) // not found
            return false;
        else return data_at(i).test(n % SegmentBits);
    }

    void set(uint32_t n) noexcept {
        const auto target_ind = n - (n % SegmentBits);
        const auto low_pos =
            std::lower_bound(indexes.begin(), indexes.end(), target_ind);
        const auto i = std::distance(indexes.begin(), low_pos);
        if (low_pos == indexes.end() || *low_pos != target_ind) // not found
            emplace_at(i, target_ind, n % SegmentBits);
        else return data_at(i).set(n % SegmentBits);
    }

    /// Check if bit is set. If it is, returns false.
    /// Otherwise, sets bit and returns true.
    bool test_and_set(uint32_t n) noexcept {
        const auto target_ind = n - (n % SegmentBits);
        const auto low_pos =
            std::lower_bound(indexes.begin(), indexes.end(), target_ind);
        const auto i = std::distance(indexes.begin(), low_pos);
        if (low_pos == indexes.end() || *low_pos != target_ind) { // not found
            emplace_at(i, target_ind, n % SegmentBits);
            return true;
        } else return data_at(i).test_and_set(n % SegmentBits);
    }

    void reset(uint32_t n) noexcept {
        const auto target_ind = n - (n % SegmentBits);
        const auto low_pos =
            std::lower_bound(indexes.begin(), indexes.end(), target_ind);
        if (low_pos == indexes.end() || *low_pos != target_ind)
            return; // not found
        const auto i = std::distance(indexes.begin(), low_pos);
        auto& d = data_at(i);
        d.reset(n % SegmentBits);
        if (d.empty()) erase_at(i); // this segment is empty, remove it
    }

    /// Returns true if this set contains all bits of rhs.
    bool contains_simd(const SegmentBitVector& rhs) const noexcept {
        const auto this_size = size(), rhs_size = rhs.size();
        if (this_size < rhs_size) return false;

        size_t this_i = 0, rhs_i = 0;
        const auto asc_indexes = get_asc_indexes();
        const auto all_zero = _mm512_setzero_si512();
        while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
            /// indexes[this_i..this_i + 16]
            const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                       v_rhs_idx =
                           _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
            /// whether each u32 matches (exist in both vectors)
            uint16_t match_this, match_rhs;
            ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this,
                                      match_rhs);

            /// the maximum index in current range [this_i..=this_i + 15],
            /// spread to vector register
            const auto rangemax_this = _mm512_set1_epi32(index_at(this_i + 15)),
                       rangemax_rhs =
                           _mm512_set1_epi32(rhs.index_at(rhs_i + 15));
            /// whether each u32 index is less than or equal to
            /// the maximum index in current range of the other vector
            const uint16_t lemask_this = _mm512_cmple_epu32_mask(v_this_idx,
                                                                 rangemax_rhs),
                           lemask_rhs = _mm512_cmple_epu32_mask(v_rhs_idx,
                                                                rangemax_this);

            /// count of matched indexes
            const auto n_matched = (unsigned int)(_mm_popcnt_u32(match_this));

            /// the number to increase for this_i / rhs_i
            const auto advance_this = 32 - _lzcnt_u32(lemask_this),
                       advance_rhs = 32 - _lzcnt_u32(lemask_rhs);

            if (advance_rhs > n_matched) return false;

            /// compress the intersected data's offset(/8bytes).
            const auto gather_this_offset_u16x32 =
                           _mm512_maskz_compress_epi32(match_this, asc_indexes),
                       gather_rhs_offset_u16x32 =
                           _mm512_maskz_compress_epi32(match_rhs, asc_indexes);
            const auto gather_this_base_addr = data_at(this_i).data;
            const auto gather_rhs_base_addr = rhs.data_at(rhs_i).data;

            const uint32_t n_matched_bits_dup =
                ((uint64_t)1 << (n_matched * 2)) - 1;

            REPEAT_i_4({ // required for `i` to be const
                const auto cur_gather_this_offset_u16x8 =
                    _mm512_extracti64x2_epi64(gather_this_offset_u16x32, i);
                const auto cur_gather_rhs_offset_u16x8 =
                    _mm512_extracti64x2_epi64(gather_rhs_offset_u16x32, i);

                const auto cur_gather_this_offset_u64x8 =
                    _mm512_cvtepu16_epi64(cur_gather_this_offset_u16x8);
                const auto cur_gather_rhs_offset_u64x8 =
                    _mm512_cvtepu16_epi64(cur_gather_rhs_offset_u16x8);

                const auto intersect_this = _mm512_mask_i64gather_epi64(
                    all_zero, n_matched_bits_dup >> i * 8,
                    cur_gather_this_offset_u64x8, gather_this_base_addr, 8);
                const auto intersect_rhs = _mm512_mask_i64gather_epi64(
                    all_zero, n_matched_bits_dup >> i * 8,
                    cur_gather_rhs_offset_u64x8, gather_rhs_base_addr, 8);

                const auto and_result =
                    _mm512_and_epi64(intersect_this, intersect_rhs);

                if (!avx_vec<512>::eq_cmp(and_result, intersect_rhs))
                    return false;
            });
            this_i += advance_this, rhs_i += advance_rhs;
        }
        while (this_i < size() && rhs_i < rhs.size()) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind > rhs_ind) return false;
            if (this_ind < rhs_ind) ++this_i;
            else {
                if (!data_at(this_i).contains(rhs.data_at(rhs_i))) return false;
                ++this_i, ++rhs_i;
            }
        }
        return rhs_i == rhs.size();
    }

    bool contains_loop(const SegmentBitVector& rhs) const noexcept {
        size_t this_i = 0, rhs_i = 0;
        while (this_i < size() && rhs_i < rhs.size()) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind > rhs_ind) return false;
            if (this_ind < rhs_ind) ++this_i;
            else {
                if (!data_at(this_i).contains(rhs.data_at(rhs_i))) return false;
                ++this_i, ++rhs_i;
            }
        }
        return rhs_i == rhs.size();
    }

    bool contains(const SegmentBitVector& rhs) const noexcept {
        return contains_simd(rhs);
        // auto simd_result = contains_simd(rhs), loop_result =
        // contains_loop(rhs); assert(simd_result == loop_result); return
        // simd_result;
    }

    // TODO: use SIMD to improve perf
    /// Returns true if this set and rhs share any bits.
    bool intersects(const SegmentBitVector& rhs) const noexcept {
        size_t this_i = 0, rhs_i = 0;
        while (this_i < size() && rhs_i < rhs.size()) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind > rhs_ind) ++rhs_i;
            else if (this_ind < rhs_ind) ++this_i;
            else {
                if (this->data_at(this_i).intersects(rhs.data_at(rhs_i)))
                    return true;
                ++this_i, ++rhs_i;
            }
        }
        return false;
    }

    bool operator==(const SegmentBitVector& rhs) const noexcept {
        if (size() != rhs.size()) return false;
        return std::memcmp(indexes.data(), rhs.indexes.data(),
                           sizeof(index_t) * size()) == 0 &&
               std::memcmp(data.data(), rhs.data.data(),
                           sizeof(Segment) * size()) == 0;
    }

    bool operator!=(const SegmentBitVector& rhs) const noexcept {
        return !(*this == rhs);
    }

    // TODO: improve
    bool union_simd(const SegmentBitVector& rhs) {
        const auto this_size = size(), rhs_size = rhs.size();
        size_t valid_count = 0, this_i = 0, rhs_i = 0;
        bool changed = false;
        const auto source_rhs_mark = _mm512_set1_epi32(0b1'0000);
        const auto reserved_extract_mask = _mm512_set1_epi32(0b1'1111);
        const auto offset_extract_mask = _mm512_set1_epi32(0b0'1111);
        /// store into merged vector to mark position
        const auto offset_info_this_u32x16 = _mm512_setr_epi32(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        const auto offset_info_rhs_u32x16 = _mm512_setr_epi32(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        const auto u64x8_dup_from_u64x4 =
            _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3);
        const auto max =
            _mm512_set1_epi32(std::numeric_limits<uint32_t>::max());
        const auto offset_1_permutex2 = _mm512_setr_epi32(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        /// the integers at 1, 3, 5, 7 are set to 8. 0 otherwise
        const auto odd_8 = _mm512_maskz_set1_epi64(0b1010'1010, 8);
        const auto seg_size = _mm512_set1_epi32(sizeof(Segment));

        // std::vector<index_t> this_index_copy = indexes;
        // std::vector<Segment> this_data_copy = data;
        auto this_index_copy_raw =
            (index_t*)std::malloc(this_size * sizeof(index_t));
        std::memcpy(this_index_copy_raw, indexes.data(),
                    this_size * sizeof(index_t));
        auto this_data_copy_raw =
            (Segment*)std::malloc(this_size * sizeof(Segment));
        std::memcpy(this_data_copy_raw, data.data(),
                    this_size * sizeof(Segment));
        // Calling clear() does not affect the result of capacity().
        indexes.clear(), data.clear();
        alignas(64) Segment sorted_data[32];
        while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
            // pack indexes[this_i..this_i + 16), rhs.indexes[rhs_i..rhs_i + 16)
            // into a avx512 vector register (u32x16)
            const auto v_this_idx =
                           _mm512_loadu_epi32(&this_index_copy_raw[this_i]),
                       v_rhs_idx =
                           _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);

            const auto rangemax_this =
                           _mm512_set1_epi32(this_index_copy_raw[this_i + 15]),
                       rangemax_rhs =
                           _mm512_set1_epi32(rhs.indexes[rhs_i + 15]);

            const auto lemask_this =
                           _mm512_cmple_epu32_mask(v_this_idx, rangemax_rhs),
                       lemask_rhs =
                           _mm512_cmple_epu32_mask(v_rhs_idx, rangemax_this);

            // for those elements can't advance now, don't compare (skip
            // processing in this iteration)
            const auto v_this_masked = _mm512_mask_blend_epi32(lemask_this, max,
                                                               v_this_idx),
                       v_rhs_masked =
                           _mm512_mask_blend_epi32(lemask_rhs, max, v_rhs_idx);

            // number of indexes that can advance, each <= 16
            const auto advance_this = 32 - _lzcnt_u32(lemask_this),
                       advance_rhs = 32 - _lzcnt_u32(lemask_rhs);

            // since `index` is multiple of SegmentBits, there are some bits
            // left to store the origin position
            // | 27 bits | 1 bit source | 4 bits offset |
            static_assert(
                SegmentBits >= 1 << 5,
                "5 bits are required to sort the index along with value");

            /// add source, offset info to the indexes
            const auto v_this_marked = _mm512_or_epi32(v_this_masked,
                                                       offset_info_this_u32x16),
                       v_rhs_marked = _mm512_or_epi32(v_rhs_masked,
                                                      offset_info_rhs_u32x16);

            auto sorted_low = v_this_marked, sorted_high = v_rhs_marked;
            rswap(sorted_low, sorted_high);

            /// the count of valid indexes in sorted result, <=32
            const auto n_processed = advance_this + advance_rhs;
            /// the lowest n_valid bits set to 1
            const uint32_t n_processed_bits = ((uint64_t)1 << n_processed) - 1;
            const uint16_t n_processed_bits_low = n_processed_bits,
                           n_processed_bits_high = n_processed_bits >> 16;

            /// set bits[0..=4] to 0
            const auto sorted_real_index_low = _mm512_andnot_epi32(
                           reserved_extract_mask, sorted_low),
                       sorted_real_index_high = _mm512_andnot_epi32(
                           reserved_extract_mask, sorted_high);
            /// shift left u32x1
            const auto sorted_real_index_low_offset1 =
                           _mm512_permutex2var_epi32(sorted_real_index_low,
                                                     offset_1_permutex2,
                                                     sorted_real_index_high),
                       sorted_real_index_high_offset1 =
                           _mm512_permutex2var_epi32(sorted_real_index_high,
                                                     offset_1_permutex2,
                                                     sorted_real_index_high);
            /// whether each index is equal to the next. 0 if invalid
            const uint16_t index_equal_mask_low = _mm512_mask_cmpeq_epi32_mask(
                               n_processed_bits_low, sorted_real_index_low,
                               sorted_real_index_low_offset1),
                           index_equal_mask_high = _mm512_mask_cmpeq_epi32_mask(
                               n_processed_bits_high, sorted_real_index_high,
                               sorted_real_index_high_offset1);
            const uint32_t index_equal_mask =
                index_equal_mask_low | (index_equal_mask_high << 16);
            /// whether each index is valid (0 if merge into prev | skipped)
            const uint32_t index_valid_mask =
                ~((uint64_t)index_equal_mask << 1) & n_processed_bits;

            /// extract offset (4 bits)
            const auto offset_val_low = _mm512_maskz_and_epi32(
                           n_processed_bits_low, sorted_low,
                           offset_extract_mask),
                       offset_val_high = _mm512_maskz_and_epi32(
                           n_processed_bits_high, sorted_high,
                           offset_extract_mask);
            /// offset in bytes
            const auto offset_bytes_low =
                           _mm512_mullo_epi32(offset_val_low, seg_size),
                       offset_bytes_high =
                           _mm512_mullo_epi32(offset_val_high, seg_size);

            static_assert(sizeof(std::byte*) == sizeof(uint64_t),
                          "64-bit address required");

            const auto rhs_base_addr_diff =
                reinterpret_cast<const std::byte*>(&rhs.data_at(rhs_i)) -
                reinterpret_cast<const std::byte*>(&this_data_copy_raw[this_i]);
            const auto rhs_base_addr_diff_u64x8 =
                _mm512_set1_epi64(rhs_base_addr_diff);
            /// bits[i] = 1 iif come from rhs && valid
            const auto come_from_rhs_low = _mm512_mask_test_epi32_mask(
                           n_processed_bits_low, sorted_low, source_rhs_mark),
                       come_from_rhs_high = _mm512_mask_test_epi32_mask(
                           n_processed_bits_high, sorted_high, source_rhs_mark);
            const uint32_t come_from_rhs =
                come_from_rhs_low | (come_from_rhs_high << 16);

            //  Sort the data: Gather from correct mem address, then store
            //  contiguously
            REPEAT_4(i, 0, {
                if (i * 8 < n_processed) {
                    const auto cur_offset_bytes_u32x8 =
                        _mm512_extracti32x8_epi32(i < 2 ? offset_bytes_low
                                                        : offset_bytes_high,
                                                  i % 2);
                    /// we must use u64 for address
                    const auto cur_offset_bytes_u64x8 =
                        _mm512_cvtepu32_epi64(cur_offset_bytes_u32x8);
                    const auto seg_addr_offset_u64x8 = _mm512_mask_add_epi64(
                        cur_offset_bytes_u64x8, come_from_rhs >> (i * 8),
                        cur_offset_bytes_u64x8, rhs_base_addr_diff_u64x8);
                    REPEAT_2(j, 0, {
                        if (i * 8 + j * 4 < n_processed) {
                            const auto data_addr_offset_u64x4 =
                                _mm512_extracti64x4_epi64(seg_addr_offset_u64x8,
                                                          j);
                            /// from a,b,c,d,e,f,g,h to a,a,b,b,...,h,h
                            const auto data_addr_offset_dup_u64x8 =
                                _mm512_permutexvar_epi64(
                                    u64x8_dup_from_u64x4,
                                    _mm512_castsi256_si512(
                                        data_addr_offset_u64x4));
                            /// add 8 (bytes) to odd elements
                            const auto data_addr_offset_u64x8 =
                                _mm512_add_epi64(data_addr_offset_dup_u64x8,
                                                 odd_8);

                            // REPEAT_8(k, 0, {
                            //     std::memcpy(&sorted_data[i * 8 + j * 4 + k],
                            //                 reinterpret_cast<const
                            //                 std::byte*>(
                            //                     &this_data_copy_raw[this_i])
                            //                     + data_addr_offset_u64x8[k],
                            //                 sizeof(Segment));
                            // });
                            const auto data_gathered_u64x8_segx4 =
                                _mm512_i64gather_epi64(
                                    data_addr_offset_u64x8,
                                    &this_data_copy_raw[this_i], 1);
                            _mm512_store_epi64(&sorted_data[i * 8 + j * 4],
                                               data_gathered_u64x8_segx4);
                        }
                    });
                }
            });

            int valid_segs = 0;
            // load sorted data, compute OR, store result
            REPEAT_8(i, 0, {
                if (i * 4 < n_processed) {
                    const auto cur_sorted_data_u64x8 =
                        _mm512_loadu_epi64(&sorted_data[i * 4]);
                    const auto sorted_data_offset_u64x8 =
                        _mm512_loadu_epi64(&sorted_data[i * 4 + 1]);

                    /// whether cur segment should use OR result (otherwise
                    /// ignored or use as-is)
                    const uint8_t use_merged_mask =
                        (index_equal_mask >> (i * 4)) & 0xf;
                    /// whether cur segment is valid (0 if merged into prev)
                    const uint8_t valid_mask =
                        (index_valid_mask >> (i * 4)) & 0xf;
                    if (!changed) {
                        /// whether cur segment comes from rhs
                        const uint8_t rhs_mask =
                            (come_from_rhs >> (i * 4)) & 0xf;
                        // any rhs element is added (without OR)
                        changed = rhs_mask & valid_mask & ~use_merged_mask;
                    }

                    /// OR result of current segment with next (sorted)
                    const auto or_result = _mm512_or_epi64(
                        cur_sorted_data_u64x8, sorted_data_offset_u64x8);
                    if (!changed) {
                        /// whether cur u64 is changed
                        const uint8_t eq_mask = _mm512_cmpeq_epu64_mask(
                            or_result, cur_sorted_data_u64x8);
                        /// bits[2k] := bits[2k] & bits[2k + 1]
                        const uint8_t seg_eq_mask =
                            eq_mask & ((eq_mask & 0b1010'1010) >> 1);
                        /// whether cur segment is unchanged (compress bits[2k]
                        /// into bits[k])
                        const auto seg_eq = _pext_u32(seg_eq_mask, 0b0101'0101);
                        // cur seg is valid && changed && should use OR merged
                        // one
                        changed = valid_mask & use_merged_mask & ~seg_eq;
                    }
                    // NOTE: OR can't produce all-zero Segment
                    const auto data_to_store = _mm512_mask_blend_epi64(
                        duplicate_bits(use_merged_mask), cur_sorted_data_u64x8,
                        or_result);
                    _mm512_mask_compressstoreu_epi64(&sorted_data[valid_segs],
                                                     duplicate_bits(valid_mask),
                                                     data_to_store);
                    valid_segs += _mm_popcnt_u32(valid_mask);
                }
            });
            indexes.resize(indexes.size() + valid_segs);
            const auto n_valid_index_in_low =
                _mm_popcnt_u32(index_valid_mask & 0xffff);
            _mm512_mask_compressstoreu_epi32(
                &indexes[valid_count], index_valid_mask, sorted_real_index_low);
            _mm512_mask_compressstoreu_epi32(
                &indexes[valid_count + n_valid_index_in_low],
                index_valid_mask >> 16, sorted_real_index_high);
            data.resize(data.size() + valid_segs);
            std::memcpy(&data[valid_count], sorted_data,
                        sizeof(Segment) * valid_segs);
            this_i += advance_this, rhs_i += advance_rhs;
            valid_count += valid_segs;
        }
        // deal with remaining elements
        while (this_i < this_size && rhs_i < rhs_size) {
            const auto this_ind = this_index_copy_raw[this_i];
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind < rhs_ind) {
                indexes.emplace_back(this_ind);
                data.emplace_back(this_data_copy_raw[this_i]);
                ++valid_count;
                ++this_i;
            } else if (this_ind > rhs_ind) {
                // copy current rhs segment to this
                indexes.emplace_back(rhs_ind);
                data.emplace_back(rhs.data_at(rhs_i));
                ++valid_count;
                changed = true;
                ++rhs_i;
            } else {
                indexes.emplace_back(this_ind);
                auto& inserted_data =
                    data.emplace_back(this_data_copy_raw[this_i]);
                changed |= (inserted_data |= rhs.data_at(rhs_i));
                ++valid_count;
                ++this_i, ++rhs_i;
            }
        }
        if (rhs_i < rhs_size) {
            // append remaining elements from rhs
            push_back_till_end(rhs, rhs_i);
            changed = true;
        } else if (this_i < this_size) {
            // append remaining elements from this
            indexes.insert(indexes.end(), this_index_copy_raw + this_i,
                           this_index_copy_raw + this_size);
            data.insert(data.end(), this_data_copy_raw + this_i,
                        this_data_copy_raw + this_size);
        }
        std::free(this_index_copy_raw);
        std::free(this_data_copy_raw);
        return changed;
    }

    bool union_simd2(const SegmentBitVector& rhs) {
        // works similar to `diff_simd`.
        // Update `this` inplace, and add extra segments if needed.
        const auto this_size = size(), rhs_size = rhs.size();
        size_t this_i = 0, rhs_i = 0;
        bool changed = false;
        while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
            /// indexes[this_i..this_i + 16]
            const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                       v_rhs_idx =
                           _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
            /// whether each u32 matches (exist in both vectors)
            uint16_t match_this, match_rhs;
            ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this,
                                      match_rhs);

            /// the maximum index in current range [this_i..=this_i + 15],
            /// spread to vector register
            const auto rangemax_this = _mm512_set1_epi32(index_at(this_i + 15)),
                       rangemax_rhs =
                           _mm512_set1_epi32(rhs.index_at(rhs_i + 15));
            /// whether each u32 index is less than or equal to
            /// the maximum index in current range of the other vector
            const uint16_t lemask_this = _mm512_cmple_epu32_mask(v_this_idx,
                                                                 rangemax_rhs),
                           lemask_rhs = _mm512_cmple_epu32_mask(v_rhs_idx,
                                                                rangemax_this);
            /// the number to increase for this_i / rhs_i
            const auto advance_this = 32 - _lzcnt_u32(lemask_this),
                       advance_rhs = 32 - _lzcnt_u32(lemask_rhs);

            // align matched data of rhs to the shape of this.
            // we don't care unmatched position
            auto matched_this_temp = match_this, matched_rhs_temp = match_rhs;
            Segment rhs_data_temp[512 / (sizeof(index_t) * 8)];
            auto rhs_data_temp_addr = rhs_data_temp,
                 rhs_data_addr = &rhs.data_at(rhs_i);
            while (matched_this_temp) {
                const auto this_pad = _tzcnt_u32(matched_this_temp),
                           rhs_pad = _tzcnt_u32(matched_rhs_temp);
                rhs_data_temp_addr += this_pad, rhs_data_addr += rhs_pad;
                *rhs_data_temp_addr = *rhs_data_addr;
                matched_this_temp >>= this_pad + 1,
                    matched_rhs_temp >>= rhs_pad + 1;
                ++rhs_data_temp_addr, ++rhs_data_addr;
            }

            /// each u32 index match => 2*u64 (data) to store & compute
            auto dup_this = duplicate_bits(match_this);
            const uint16_t advance_this_to_bits =
                ((uint32_t)1 << advance_this) - 1;
            const auto dup_advance_this = duplicate_bits(advance_this_to_bits);

            // compute OR result of matched segments
            REPEAT_i_4({
                /// matched & ordered 4 segments (8 u64) from memory. zero in
                /// case of out of bounds
                const auto v_this = _mm512_maskz_loadu_epi64(
                    dup_advance_this >> (i * 8), &data_at(this_i) + i * 4);
                const auto v_rhs = _mm512_maskz_loadu_epi64(
                    dup_this >> (i * 8), &rhs_data_temp[i * 4]);
                const auto or_result = _mm512_or_epi64(v_this, v_rhs);

                if (!changed) // compute `changed` if not already set
                    changed = !avx_vec<512>::eq_cmp(v_this, or_result);

                _mm512_mask_storeu_epi64(data.data() + this_i + i * 4,
                                         dup_advance_this >> (i * 8),
                                         or_result);
            });
            _mm512_mask_storeu_epi32(indexes.data() + this_i, dup_advance_this,
                                     v_this_idx);
            this_i += advance_this, rhs_i += advance_rhs;
        }

        size_t unprocessed_this_idx = this_i;
        // deal with remaining | extra elements
        size_t this_j = 0, rhs_j = 0;
        while (this_j < size() && rhs_j < rhs_size) {
            const auto this_ind = index_at(this_j);
            const auto rhs_ind = rhs.index_at(rhs_j);
            if (this_ind < rhs_ind) ++this_j;
            else if (this_ind > rhs_ind) {
                // copy current rhs segment to this
                emplace_at(this_j, rhs_ind, rhs.data_at(rhs_j));
                changed = true;
                if (this_j <= unprocessed_this_idx) ++unprocessed_this_idx;
                ++this_j, ++rhs_j;
            } else {                                // this_ind == rhs_ind
                if (this_j >= unprocessed_this_idx) // not processed by SIMD
                    changed |= (data_at(this_j) |= rhs.data_at(rhs_j));
                ++this_j, ++rhs_j;
            }
        }
        if (rhs_j < rhs_size) {
            // append remaining elements from rhs
            push_back_till_end(rhs, rhs_j);
            changed = true;
        }
        return changed;
    }

    /// Inplace union with rhs.
    /// Returns true if this set changed.
    bool operator|=(const SegmentBitVector& rhs) {
        return union_simd2(rhs);

        auto copy = *this;
        auto simd_result = copy.union_simd2(rhs);

        const auto rhs_size = rhs.size();
        indexes.reserve(rhs_size);
        data.reserve(rhs_size);
        size_t this_i = 0, rhs_i = 0;
        bool changed = false;
        while (this_i < size() && rhs_i < rhs_size) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind < rhs_ind) ++this_i;
            else if (this_ind > rhs_ind) {
                // copy current rhs segment to this
                emplace_at(this_i, rhs_ind, rhs.data_at(rhs_i));
                changed = true;
                ++this_i, ++rhs_i;
            } else {
                changed |= (data_at(this_i) |= rhs.data_at(rhs_i));
                ++this_i, ++rhs_i;
            }
        }
        if (rhs_i < rhs_size) {
            // append remaining elements from rhs
            push_back_till_end(rhs, rhs_i);
            changed = true;
        }
        assert(simd_result == changed);
        assert(*this == copy);
        return changed;
    }

    SegmentBitVector operator|(const SegmentBitVector& rhs) const {
        SegmentBitVector copy(*this);
        copy |= rhs;
        return copy;
    }

    /// Inplace intersection with rhs.
    /// Returns true if this set changed.
    /// Optimized using AVX512 intrinsics, requires AVX512F inst set.
    bool intersect_simd(const SegmentBitVector& rhs) {
        const auto this_size = size(), rhs_size = rhs.size();
        size_t valid_count = 0, this_i = 0, rhs_i = 0;
        bool changed = false;
        const auto asc_indexes = get_asc_indexes();
        const auto all_zero = _mm512_setzero_si512();
        while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
            /// indexes[this_i..this_i + 16]
            const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                       v_rhs_idx =
                           _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
            /// whether each u32 matches (exist in both vectors)
            uint16_t match_this, match_rhs;
            ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this,
                                      match_rhs);

            /// the maximum index in current range [this_i..=this_i + 15],
            /// spread to vector register
            const auto rangemax_this = _mm512_set1_epi32(index_at(this_i + 15)),
                       rangemax_rhs =
                           _mm512_set1_epi32(rhs.index_at(rhs_i + 15));
            /// whether each u32 index is less than or equal to
            /// the maximum index in current range of the other vector
            const uint16_t lemask_this = _mm512_cmple_epu32_mask(v_this_idx,
                                                                 rangemax_rhs),
                           lemask_rhs = _mm512_cmple_epu32_mask(v_rhs_idx,
                                                                rangemax_this);
            /// the number to increase for this_i / rhs_i
            const auto advance_this = 32 - _lzcnt_u32(lemask_this),
                       advance_rhs = 32 - _lzcnt_u32(lemask_rhs);

            /// compress the intersected data's offset(/8bytes).
            const auto gather_this_offset_u16x32 =
                           _mm512_maskz_compress_epi32(match_this, asc_indexes),
                       gather_rhs_offset_u16x32 =
                           _mm512_maskz_compress_epi32(match_rhs, asc_indexes);
            const auto gather_this_base_addr = data_at(this_i).data;
            const auto gather_rhs_base_addr = rhs.data_at(rhs_i).data;

            const auto ordered_indexes =
                _mm512_maskz_compress_epi32(match_this, v_this_idx);

            const auto n_matched = _mm_popcnt_u32(match_this);
            const uint32_t n_matched_bits_dup =
                ((uint64_t)1 << (n_matched * 2)) - 1;

            // compute AND result of matched segments
            REPEAT_i_4({
                const auto cur_gather_this_offset_u16x8 =
                    _mm512_extracti64x2_epi64(gather_this_offset_u16x32, i);
                const auto cur_gather_rhs_offset_u16x8 =
                    _mm512_extracti64x2_epi64(gather_rhs_offset_u16x32, i);

                const auto cur_gather_this_offset_u64x8 =
                    _mm512_cvtepu16_epi64(cur_gather_this_offset_u16x8);
                const auto cur_gather_rhs_offset_u64x8 =
                    _mm512_cvtepu16_epi64(cur_gather_rhs_offset_u16x8);

                /// matched & ordered 4 segments (8 u64) from memory. zero in
                /// case of out of bounds
                const auto intersect_this = _mm512_mask_i64gather_epi64(
                    all_zero, n_matched_bits_dup >> i * 8,
                    cur_gather_this_offset_u64x8, gather_this_base_addr, 8);
                const auto intersect_rhs = _mm512_mask_i64gather_epi64(
                    all_zero, n_matched_bits_dup >> i * 8,
                    cur_gather_rhs_offset_u64x8, gather_rhs_base_addr, 8);
                const auto and_result =
                    _mm512_and_epi64(intersect_this, intersect_rhs);

                if (!changed) // compute `changed` if not already set
                    changed = !avx_vec<512>::eq_cmp(intersect_this, and_result);

                /// zero-test for each u64, 1 means nonzero
                const auto nzero_mask =
                    _mm512_test_epi64_mask(and_result, and_result);
                // _bit[2k] := bit[2k] | bit[2k+1]
                uint8_t nzero_mask_by_segment =
                    nzero_mask | ((nzero_mask & 0b10101010) >> 1);
                // _bit[k] := bit[2k] | bit[2k+1]
                const auto nzero_compressed =
                    _pext_u32(nzero_mask_by_segment, 0b01010101);
                // _bit[2k+1] := bit[2k] | bit[2k+1],
                // compute here to improve pipeline perf
                nzero_mask_by_segment |= ((nzero_mask & 0b01010101) << 1);
                // store new index & data for 4 segments (remove empty segments)
                _mm512_mask_compressstoreu_epi32(indexes.data() + valid_count,
                                                 nzero_compressed << (i * 4),
                                                 ordered_indexes);
                _mm512_mask_compressstoreu_epi64(data.data() + valid_count,
                                                 nzero_mask_by_segment,
                                                 and_result);

                const auto nzero_count = _mm_popcnt_u32(nzero_compressed);
                valid_count += nzero_count;
            });
            this_i += advance_this, rhs_i += advance_rhs;
        }
        // use trival loop for the rest
        // TODO: improve
        while (this_i < this_size && rhs_i < rhs_size) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind < rhs_ind) {
                // ignore this segment (not copied to valid position)
                ++this_i;
                changed = true;
            } else if (this_ind > rhs_ind) ++rhs_i;
            else { // this_ind == rhs_ind
                const auto vcur_this =
                               avx_vec<SegmentBits>::load(data_at(this_i).data),
                           vcur_rhs = avx_vec<SegmentBits>::load(
                               rhs.data_at(rhs_i).data);
                const auto and_result =
                    avx_vec<SegmentBits>::and_op(vcur_this, vcur_rhs);
                if (avx_vec<SegmentBits>::is_zero(and_result))
                    // no bits in common, skip this segment
                    changed = true;
                else {
                    if (!changed) // compute `changed` if not already set
                        changed = !avx_vec<SegmentBits>::eq_cmp(vcur_this,
                                                                and_result);

                    // store the result
                    avx_vec<SegmentBits>::store(data_at(valid_count).data,
                                                and_result);
                    indexes[valid_count] = this_ind;
                    ++valid_count; // increment valid count
                }
                ++this_i, ++rhs_i;
            }
        }
        truncate(valid_count);
        changed |= (valid_count != this_size);
        return changed;
    }
    /// Inplace intersection with rhs.
    /// Returns true if this set changed.
    bool operator&=(const SegmentBitVector& rhs) {
        return intersect_simd(rhs);

        // // old implementation without SIMD
        // bool changed = false;
        // size_t this_i = 0, rhs_i = 0;
        // while (this_i < size() && rhs_i < rhs.size()) {
        //     const auto this_ind = index_at(this_i);
        //     const auto rhs_ind = rhs.index_at(rhs_i);
        //     if (this_ind < rhs_ind) {
        //         erase_at(this_i);
        //         changed = true;
        //     } else if (this_ind > rhs_ind) ++rhs_i;
        //     else {
        //         const auto [cur_changed, zeroed] =
        //             (data_at(this_i) &= rhs.data_at(rhs_i));
        //         changed |= cur_changed;
        //         if (zeroed) erase_at(this_i); // remove empty segment
        //         else ++this_i;
        //         ++rhs_i;
        //     }
        // }
        // if (this_i < size()) {
        //     // remove remaining elements from this
        //     truncate(this_i);
        //     changed = true;
        // }
        // return changed;
    }
    /// Inplace difference with rhs.
    /// Returns true if this set changed.
    bool diff_simd(const SegmentBitVector& rhs) {
        const auto this_size = size(), rhs_size = rhs.size();
        size_t valid_count = 0, this_i = 0, rhs_i = 0;
        bool changed = false;
        while (this_i + 16 <= this_size && rhs_i + 16 <= rhs_size) {
            /// indexes[this_i..this_i + 16]
            const auto v_this_idx = _mm512_loadu_epi32(indexes.data() + this_i),
                       v_rhs_idx =
                           _mm512_loadu_epi32(rhs.indexes.data() + rhs_i);
            /// whether each u32 matches (exist in both vectors)
            uint16_t match_this, match_rhs;
            ne_mm512_2intersect_epi32(v_this_idx, v_rhs_idx, match_this,
                                      match_rhs);

            /// the maximum index in current range [this_i..=this_i + 15],
            /// spread to vector register
            const auto rangemax_this = _mm512_set1_epi32(index_at(this_i + 15)),
                       rangemax_rhs =
                           _mm512_set1_epi32(rhs.index_at(rhs_i + 15));
            /// whether each u32 index is less than or equal to
            /// the maximum index in current range of the other vector
            const uint16_t lemask_this = _mm512_cmple_epu32_mask(v_this_idx,
                                                                 rangemax_rhs),
                           lemask_rhs = _mm512_cmple_epu32_mask(v_rhs_idx,
                                                                rangemax_this);
            /// the number to increase for this_i / rhs_i
            const auto advance_this = 32 - _lzcnt_u32(lemask_this),
                       advance_rhs = 32 - _lzcnt_u32(lemask_rhs);

            // align matched data of rhs to the shape of this.
            // we don't care unmatched position
            auto matched_this_temp = match_this, matched_rhs_temp = match_rhs;
            Segment rhs_data_temp[512 / (sizeof(index_t) * 8)];
            auto rhs_data_temp_addr = rhs_data_temp,
                 rhs_data_addr = &rhs.data_at(rhs_i);
            while (matched_this_temp) {
                const auto this_pad = _tzcnt_u32(matched_this_temp),
                           rhs_pad = _tzcnt_u32(matched_rhs_temp);
                rhs_data_temp_addr += this_pad, rhs_data_addr += rhs_pad;
                *rhs_data_temp_addr = *rhs_data_addr;
                matched_this_temp >>= this_pad + 1,
                    matched_rhs_temp >>= rhs_pad + 1;
                ++rhs_data_temp_addr, ++rhs_data_addr;
            }

            /// each u32 index match => 2*u64 (data) to store & compute
            auto dup_this = duplicate_bits(match_this);
            const uint16_t advance_this_to_bits =
                ((uint32_t)1 << advance_this) - 1;
            const auto dup_advance_this = duplicate_bits(advance_this_to_bits);

            // compute AND result of matched segments
            REPEAT_i_4({
                /// matched & ordered 4 segments (8 u64) from memory. zero in
                /// case of out of bounds
                const auto v_this = _mm512_maskz_loadu_epi64(
                    dup_advance_this >> (i * 8), &data_at(this_i) + i * 4);
                const auto v_rhs = _mm512_maskz_loadu_epi64(
                    dup_this >> (i * 8), &rhs_data_temp[i * 4]);
                // _mm512_andnot_epi64 intrinsic: NOT of 512 bits (composed of
                // packed 64-bit integers) in a and then AND with b
                const auto andnot_result = _mm512_andnot_epi64(v_rhs, v_this);

                if (!changed) // compute `changed` if not already set
                    changed = !avx_vec<512>::eq_cmp(v_this, andnot_result);

                /// zero-test for each u64, 1 means nonzero
                const auto nzero_mask =
                    _mm512_test_epi64_mask(andnot_result, andnot_result);
                // _bit[2k] := bit[2k] | bit[2k+1]
                uint8_t nzero_mask_by_segment =
                    nzero_mask | ((nzero_mask & 0b10101010) >> 1);
                // _bit[k] := bit[2k] | bit[2k+1]
                const auto nzero_compressed =
                    _pext_u32(nzero_mask_by_segment, 0b01010101);
                // _bit[2k+1] := bit[2k] | bit[2k+1],
                // compute here to improve pipeline perf
                nzero_mask_by_segment |= ((nzero_mask & 0b01010101) << 1);
                // store new index & data for 4 segments (remove empty segments)
                _mm512_mask_compressstoreu_epi32(indexes.data() + valid_count,
                                                 nzero_compressed << (i * 4),
                                                 v_this_idx);
                _mm512_mask_compressstoreu_epi64(data.data() + valid_count,
                                                 nzero_mask_by_segment,
                                                 andnot_result);

                const auto nzero_count = _mm_popcnt_u32(nzero_compressed);
                valid_count += nzero_count;
            });
            this_i += advance_this, rhs_i += advance_rhs;
        }
        while (this_i < this_size && rhs_i < rhs_size) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind < rhs_ind) { // keep this segment
                indexes[valid_count] = this_ind;
                data_at(valid_count) = data_at(this_i);
                ++this_i;
                ++valid_count;
            } else if (this_ind > rhs_ind) ++rhs_i;
            else { // compute ANDNOT
                const auto v_this =
                    avx_vec<SegmentBits>::load(&data_at(this_i));
                const auto v_rhs =
                    avx_vec<SegmentBits>::load(&rhs.data_at(rhs_i));
                const auto andnot_result =
                    avx_vec<SegmentBits>::andnot_op(v_this, v_rhs);
                if (avx_vec<SegmentBits>::is_zero(
                        andnot_result)) // changed to zero
                    changed = true;
                else {
                    if (!changed)
                        changed = !avx_vec<SegmentBits>::eq_cmp(v_this,
                                                                andnot_result);
                    indexes[valid_count] = this_ind;
                    avx_vec<SegmentBits>::store(&data_at(valid_count),
                                                andnot_result);
                    valid_count++;
                }
                ++this_i, ++rhs_i;
            }
        }
        // the rest element is kept
        const auto rest_count = this_size - this_i;
        std::memmove(&indexes[valid_count], &indexes[this_i],
                     sizeof(index_t) * rest_count);
        std::memmove(&data_at(valid_count), &data_at(this_i),
                     sizeof(Segment) * rest_count);
        valid_count += rest_count;
        truncate(valid_count);
        changed |= (valid_count != this_size);
        return changed;
    }
    bool operator-=(const SegmentBitVector& rhs) {
        return diff_simd(rhs);

        // bool changed = false;
        // size_t this_i = 0, rhs_i = 0;
        // while (this_i < size() && rhs_i < rhs.size()) {
        //     const auto this_ind = index_at(this_i);
        //     const auto rhs_ind = rhs.index_at(rhs_i);
        //     if (this_ind < rhs_ind) ++this_i;
        //     else if (this_ind > rhs_ind) ++rhs_i;
        //     else {
        //         const auto [cur_changed, zeroed] =
        //             (data_at(this_i) -= rhs.data_at(rhs_i));
        //         changed |= cur_changed;
        //         if (zeroed) erase_at(this_i); // remove empty segment
        //         else ++this_i;
        //         ++rhs_i;
        //     }
        // }
        // return changed;
    }
    bool intersectWithComplement(const SegmentBitVector& rhs) {
        return *this -= rhs;
    }

    void intersectWithComplement(const SegmentBitVector& lhs,
                                 const SegmentBitVector& rhs) {
        // TODO: inefficient!
        *this = lhs;
        intersectWithComplement(rhs);
    }

    size_t hash() const noexcept {
        return szudzik(count(), szudzik(size(), size() > 0 ? *begin() : -1));
    }
};
} // namespace SVF
