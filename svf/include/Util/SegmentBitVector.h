#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "SIMDHelper.h"
#include "Util/GeneralType.h"

namespace SVF {
template <size_t SegmentBits = 128> class SegmentBitVector {
    friend class SVFIRWriter;
    friend class SVFIRReader;

public:
    using UnitType = uint64_t;
    static constexpr size_t UnitBits = sizeof(UnitType) * 8;
    static constexpr size_t UnitsPerSegment = SegmentBits / UnitBits;

    struct Segment {
        UnitType data[UnitsPerSegment]{};

        Segment() = default;
        /// Initialize segment with only one bit set.
        Segment(size_t index) {
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
            if (data[unit_index] & mask) return false; // already set
            data[unit_index] |= mask;                  // set the bit
            return true;                               // it was not set before
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
    } __attribute__((packed));

    using index_t = size_t;

protected:
    std::vector<index_t> indexes;
    std::vector<Segment> data;
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
    inline __attribute__((always_inline)) void erase_till_end(
        size_t start) noexcept {
        indexes.erase(indexes.begin() + start, indexes.end());
        data.erase(data.begin() + start, data.end());
    }
    template <typename... Args>
    inline __attribute__((always_inline)) void emplace_at(
        size_t i, const index_t& ind, Args&&... seg) noexcept {
        indexes.emplace(indexes.begin() + i, ind);
        data.emplace(data.begin() + i, std::forward<Args>(seg)...);
    }
    inline __attribute__((always_inline)) void insert_many(
        size_t this_offset, const SegmentBitVector rhs, size_t other_offset,
        size_t count) noexcept {
        indexes.insert(indexes.begin() + this_offset,
                       rhs.indexes.begin() + other_offset,
                       rhs.indexes.begin() + other_offset + count);
        data.insert(data.begin() + this_offset, rhs.data.begin() + other_offset,
                    rhs.data.begin() + other_offset + count);
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
        typename std::vector<index_t>::const_iterator indexesIt;
        typename std::vector<index_t>::const_iterator indexesEnd;
        typename std::vector<Segment>::const_iterator dataIt;
        unsigned char unit_index; // unit index in the current segment
        unsigned char bit_index;  // bit index in the current unit
        bool end;
        /// move to next unit, returns false if reached the end
        void incr_unit() {
            bit_index = 0;
            if (++unit_index == UnitsPerSegment) {
                // forward to next segment
                unit_index = 0;
                ++indexesIt, ++dataIt;
                if (indexesIt == indexesEnd) {
                    // reached the end
                    end = true;
                }
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
                auto masked_unit = dataIt->data[unit_index] & mask;
                auto tz_count = __tzcnt_u64(masked_unit);
                if (tz_count < UnitBits) {
                    // found a set bit
                    bit_index = tz_count;
                    cur_pos = *indexesIt + unit_index * UnitBits + bit_index;
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
              indexesEnd(vec.indexes.end()), end(end | vec.empty()) {
            if (end) return;
            indexesIt = vec.indexes.begin();
            dataIt = vec.data.begin();
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
                indexesEnd == other.indexesEnd &&
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
        return popcnt<SegmentBits>(data.data(), size());
    }

    /// Empty the set.
    void clear() noexcept {
        indexes.clear();
        data.clear();
    }

    /// Returns true if n is in this set.
    bool test(uint32_t n) const noexcept {
        for (size_t i = 0; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // early return
            if (n >= ind && n < ind + SegmentBits)
                return data_at(i).test(n % SegmentBits);
        }
        return false;
    }

    void set(uint32_t n) noexcept {
        // TODO binary search
        size_t i = 0;
        for (; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // need to insert before this index
            if (n >= ind && n < ind + SegmentBits)
                return data_at(i).set(n % SegmentBits);
        }
        emplace_at(i, n - (n % SegmentBits), n % SegmentBits);
    }

    bool test_and_set(uint32_t n) noexcept {
        size_t i = 0;
        for (; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // need to insert before this index
            if (n >= ind && n < ind + SegmentBits)
                return data_at(i).test_and_set(n % SegmentBits);
        }
        emplace_at(i, n - (n % SegmentBits), n % SegmentBits);
        return true;
    }

    void reset(uint32_t n) noexcept {
        // TODO binary search
        for (size_t i = 0; i < size(); ++i) {
            const auto ind = index_at(i);
            if (ind > n) break; // early return
            if (n >= ind && n < ind + SegmentBits) {
                const auto& d = data_at(i);
                d.reset(n % SegmentBits);
                if (d.empty()) erase_at(i); // this segment is empty, remove it
                break;
            }
        }
    }

    /// Returns true if this set contains all bits of rhs.
    bool contains(const SegmentBitVector& rhs) const noexcept {
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
        for (size_t i = 0; i < size(); ++i)
            if (index_at(i) != rhs.index_at(i) ||
                data_at(i) != rhs.data_at(i))
                return false;
        return true;
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
            insert_many(size(), rhs, rhs_i, rhs_size - rhs_i);
            changed = true;
        }
        return changed;
    }
    /// Inplace intersection with rhs.
    /// Returns true if this set changed.
    bool operator&=(const SegmentBitVector& rhs) {
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
            erase_till_end(this_i);
            changed = true;
        }
        return changed;
    }
    /// Inplace difference with rhs.
    /// Returns true if this set changed.
    bool operator-=(const SegmentBitVector& rhs) {
        bool changed = false;
        size_t this_i = 0, rhs_i = 0;
        while (this_i < size() && rhs_i < rhs.size()) {
            const auto this_ind = index_at(this_i);
            const auto rhs_ind = rhs.index_at(rhs_i);
            if (this_ind < rhs_ind) ++this_i;
            else if (this_ind > rhs_ind) ++rhs_i;
            else {
                const auto [cur_changed, zeroed] =
                    (data_at(this_i) -= rhs.data_at(rhs_i));
                changed |= cur_changed;
                if (zeroed) erase_at(this_i); // remove empty segment,
                else ++this_i;
                ++rhs_i;
            }
        }
        return changed;
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
