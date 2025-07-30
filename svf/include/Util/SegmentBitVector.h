#pragma once

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "SIMDHelper.h"
#include "Util/GeneralType.h"

/// Macro to loop 4 times, useful for unrolling loops.
#define LOOP_i_4(block)                                                        \
    {                                                                          \
        constexpr int i = 0;                                                   \
        block                                                                  \
    }                                                                          \
    {                                                                          \
        constexpr int i = 1;                                                   \
        block                                                                  \
    }                                                                          \
    {                                                                          \
        constexpr int i = 2;                                                   \
        block                                                                  \
    }                                                                          \
    {                                                                          \
        constexpr int i = 3;                                                   \
        block                                                                  \
    }

/// Returns a __m512i that contains 32*16-bit integers in ascending order,
/// that is, {0, 1, 2, ..., 31} (from e0 to e31).
static inline auto get_asc_indexes() {
    return _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
                            18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5,
                            4, 3, 2, 1, 0);
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
            const auto prev_set = data[unit_index] & mask;
            data[unit_index] |= mask;
            return prev_set == 0; // return true if it was not set before
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
        const SegmentBitVector rhs, size_t other_offset) noexcept {
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
        for (size_t i = 0; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // early return
            if (ind == target_ind) return data_at(i).test(n % SegmentBits);
        }
        return false;
    }

    void set(uint32_t n) noexcept {
        // TODO binary search?
        size_t i = 0;
        const auto target_ind = n - (n % SegmentBits);
        for (; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // need to insert before this index
            if (ind == target_ind) return data_at(i).set(n % SegmentBits);
        }
        emplace_at(i, target_ind, n % SegmentBits);
    }

    bool test_and_set(uint32_t n) noexcept {
        size_t i = 0;
        const auto target_ind = n - (n % SegmentBits);
        for (; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // need to insert before this index
            if (ind == target_ind)
                return data_at(i).test_and_set(n % SegmentBits);
        }
        emplace_at(i, target_ind, n % SegmentBits);
        return true;
    }

    void reset(uint32_t n) noexcept {
        // TODO binary search?
        const auto target_ind = n - (n % SegmentBits);
        for (size_t i = 0; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // early return
            if (ind == target_ind) {
                auto& d = data_at(i);
                d.reset(n % SegmentBits);
                if (d.empty()) erase_at(i); // this segment is empty, remove it
                break;
            }
        }
    }

    /// Returns true if this set contains all bits of rhs.
    bool contains(const SegmentBitVector& rhs) const noexcept {
        const auto this_size = size(), rhs_size = rhs.size();
        if (this_size < rhs_size) return false;

        size_t this_i = 0, rhs_i = 0;
        const auto asc_indexes = get_asc_indexes();
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
            const auto match_count = (unsigned int)(_mm_popcnt_u32(match_this));

            /// the number to increase for this_i / rhs_i
            const auto advance_this = 32 - _lzcnt_u32(lemask_this),
                       advance_rhs = 32 - _lzcnt_u32(lemask_rhs);

            if (advance_this > match_count) return false;

            /// each u32 index match => 2*u64 (data) to store & compute
            auto dup_this = duplicate_bits(match_this),
                 dup_rhs = duplicate_bits(match_rhs);

            /// compress the intersected data's offset(/8bytes).
            const auto gather_this_offset_u16x32 =
                           ne_mm512_maskz_compress_epi16(dup_this, asc_indexes),
                       gather_rhs_offset_u16x32 =
                           ne_mm512_maskz_compress_epi16(dup_rhs, asc_indexes);
            const auto gather_this_base_addr = data_at(this_i).data;
            const auto gather_rhs_base_addr = rhs.data_at(rhs_i).data;

            LOOP_i_4({ // required for `i` to be const
                const auto cur_gather_this_offset_u16x8 =
                    _mm512_extracti64x2_epi64(gather_this_offset_u16x32, i);
                const auto cur_gather_rhs_offset_u16x8 =
                    _mm512_extracti64x2_epi64(gather_rhs_offset_u16x32, i);

                const auto cur_gather_this_offset_u64x8 =
                    _mm512_cvtepu16_epi64(cur_gather_this_offset_u16x8);
                const auto cur_gather_rhs_offset_u64x8 =
                    _mm512_cvtepu16_epi64(cur_gather_rhs_offset_u16x8);

                const auto intersect_this = _mm512_i64gather_epi64(
                    cur_gather_this_offset_u64x8, gather_this_base_addr, 8);
                const auto intersect_rhs = _mm512_i64gather_epi64(
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

    /// Inplace union with rhs.
    /// Returns true if this set changed.
    bool operator|=(const SegmentBitVector& rhs) {
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
        return changed;
    }

    /// Inplace intersection with rhs.
    /// Returns true if this set changed.
    /// Optimized using AVX512 intrinsics, requires AVX512F inst set.
    bool intersect_fast(const SegmentBitVector& rhs) {
        const auto this_size = size(), rhs_size = rhs.size();
        size_t valid_count = 0, this_i = 0, rhs_i = 0;
        bool changed = false;
        const auto asc_indexes = get_asc_indexes();
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

            /// each u32 index match => 2*u64 (data) to store & compute
            auto dup_this = duplicate_bits(match_this),
                 dup_rhs = duplicate_bits(match_rhs);

            /// compress the intersected data's offset(/8bytes).
            const auto gather_this_offset_u16x32 =
                           ne_mm512_maskz_compress_epi16(dup_this, asc_indexes),
                       gather_rhs_offset_u16x32 =
                           ne_mm512_maskz_compress_epi16(dup_rhs, asc_indexes);
            const auto gather_this_base_addr = data_at(this_i).data;
            const auto gather_rhs_base_addr = rhs.data_at(rhs_i).data;

            const auto ordered_indexes =
                _mm512_maskz_compress_epi32(match_this, v_this_idx);

            // compute AND result of matched segments
            LOOP_i_4({
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
                const auto intersect_this = _mm512_i64gather_epi64(
                    cur_gather_this_offset_u64x8, gather_this_base_addr, 8);
                const auto intersect_rhs = _mm512_i64gather_epi64(
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
        return intersect_fast(rhs);

        // old implementation without SIMD
        bool changed = false;
        size_t this_i = 0, rhs_i = 0;
        while (this_i < size() && rhs_i < rhs.size()) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind < rhs_ind) {
                erase_at(this_i);
                changed = true;
            } else if (this_ind > rhs_ind) ++rhs_i;
            else {
                const auto [cur_changed, zeroed] =
                    (data_at(this_i) &= rhs.data_at(rhs_i));
                changed |= cur_changed;
                if (zeroed) erase_at(this_i); // remove empty segment
                else ++this_i;
                ++rhs_i;
            }
        }
        if (this_i < size()) {
            // remove remaining elements from this
            truncate(this_i);
            changed = true;
        }
        return changed;
    }
    /// Inplace difference with rhs.
    /// Returns true if this set changed.
    bool quick_diff(const SegmentBitVector& rhs) {
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
            LOOP_i_4({
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
        return quick_diff(rhs);
        
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
        SVF::Hash<std::pair<std::pair<size_t, size_t>, size_t>> h;
        return h(std::make_pair(std::make_pair(count(), size()),
                                size() > 0 ? *begin() : -1));
    }
};
} // namespace SVF
