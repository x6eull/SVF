#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

#include "SIMDHelper.h"

namespace SVF {
using std::vector;
class SegmentBitVector {
    friend class SVFIRWriter;
    friend class SVFIRReader;

public:
    static constexpr size_t SegmentBits = 512;
    static constexpr size_t SegmentSize = SegmentBits / 8;

    using UnitType = unsigned long long;
    static constexpr size_t UnitSize = sizeof(UnitType);
    static constexpr size_t UnitBits = UnitSize * 8;
    static constexpr size_t UnitsPerSegment = SegmentSize / UnitSize;

    /// A structure to hold 512 bits
    struct Segment {
        UnitType data[UnitsPerSegment];

        /// Returns true if all bits are zero.
        bool empty() const noexcept {
            return testz<SegmentBits>(data);
        }

        bool test(size_t index) const noexcept {
            const size_t unit_index = index / UnitBits;
            const size_t bit_index = index % UnitBits;
            return data[unit_index] & (1u << bit_index);
        }

        void set(size_t index) noexcept {
            const size_t unit_index = index / UnitBits;
            const size_t bit_index = index % UnitBits;
            data[unit_index] |= (1u << bit_index);
        }

        void reset(size_t index) noexcept {
            const size_t unit_index = index / UnitBits;
            const size_t bit_index = index % UnitBits;
            data[unit_index] &= ~(1u << bit_index);
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

    using indice_t = size_t;

protected:
    vector<indice_t> indices;
    vector<Segment> data;
    /// Returns # of segments.
    size_t size() const noexcept {
        return indices.size();
    }

public:
    class SegmentBitVectorIterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = indice_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const indice_t*;
        using reference = const indice_t&;
        indice_t cur_pos;

    protected:
        typename vector<indice_t>::const_iterator indicesIt;
        typename vector<indice_t>::const_iterator indicesEnd;
        typename vector<Segment>::const_iterator dataIt;
        unsigned char unit_index; // unit index in the current segment
        unsigned char bit_index;  // bit index in the current unit
        bool end;
        /// move to next unit, returns false if reached the end
        void incr_unit() {
            bit_index = 0;
            if (++unit_index == UnitsPerSegment) {
                // forward to next segment
                unit_index = 0;
                ++indicesIt, ++dataIt;
                if (indicesIt == indicesEnd) {
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
                    cur_pos = *indicesIt + unit_index * UnitBits + bit_index;
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
              indicesEnd(vec.indices.end()), end(end) {
            if (end) return;
            indicesIt = vec.indices.begin();
            dataIt = vec.data.begin();
            unit_index = 0;
            bit_index = 0;
            search();
        }
        SegmentBitVectorIterator(const SegmentBitVectorIterator& other) =
            default;
        SegmentBitVectorIterator& operator=(
            const SegmentBitVectorIterator& other) = default;

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
                indicesEnd == other.indicesEnd &&
                (( // both ended, or
                     end && other.end) ||
                 ( // both not ended and pointing to the same position
                     end == other.end && cur_pos == other.cur_pos));
        }
        bool operator!=(const SegmentBitVectorIterator& other) const {
            return !(*this == other);
        }
    };

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
    SegmentBitVector(const SegmentBitVector& other)
        : indices(other.indices), data(other.data) {}

    /// Move constructor
    SegmentBitVector(SegmentBitVector&& other) noexcept
        : indices(std::move(other.indices)), data(std::move(other.data)) {}

    /// Copy assignment
    SegmentBitVector& operator=(const SegmentBitVector& other) {
        if (this != &other) {
            indices = other.indices;
            data = other.data;
        }
        return *this;
    }

    /// Move assignment
    SegmentBitVector& operator=(SegmentBitVector&& other) noexcept {
        if (this != &other) {
            indices = std::move(other.indices);
            data = std::move(other.data);
        }
        return *this;
    }

    /// Returns true if no bits are set.
    bool empty() const noexcept {
        return indices.empty();
    }

    /// Returns the count of set bits.
    uint32_t count() const noexcept {
        return popcnt<SegmentBits>(data.data(), data.size());
    }

    /// Empty the set.
    void clear() noexcept {
        indices.clear();
        data.clear();
    }

    /// Returns true if n is in this set.
    bool test(uint32_t n) const noexcept {
        for (int i = 0; i < indices.size(); ++i) {
            const auto ind = indices[i];
            if (ind > n) break; // early return
            if (n >= ind && n < ind + SegmentBits)
                return data[i].test(n % SegmentBits);
        }
        return false;
    }

    void set(uint32_t n) noexcept {
        int i = 0;
        for (; i < indices.size(); ++i) {
            const auto ind = indices[i];
            if (ind > n) break; // early return
            if (n >= ind && n < ind + SegmentBits)
                return data[i].set(n % SegmentBits);
        }
        indices.emplace(indices.begin() + i, n - (n % SegmentBits));
        data.emplace(data.begin() + i);
        return data[i].set(n % SegmentBits);
    }

    bool test_and_set(uint32_t n) noexcept {
        // TODO: improve
        const auto result = test(n);
        set(n);
        return result;
    }

    void reset(uint32_t n) noexcept {
        for (int i = 0; i < indices.size(); ++i) {
            const auto ind = indices[i];
            if (ind > n) break; // early return
            if (n >= ind && n < ind + SegmentBits) {
                data[i].reset(n % SegmentBits);
                if (data[i].empty()) { // this segment is empty, remove it
                    indices.erase(indices.begin() + i);
                    data.erase(data.begin() + i);
                }
                return;
            }
        }
    }

    /// Returns true if this set contains all bits of rhs.
    bool contains(const SegmentBitVector& rhs) const noexcept {
        size_t this_i = 0, rhs_i = 0;
        while (this_i < indices.size() && rhs_i < rhs.indices.size()) {
            const auto this_ind = indices[this_i];
            const auto rhs_ind = rhs.indices[rhs_i];
            if (this_ind > rhs_ind) return false;
            if (this_ind < rhs_ind) ++this_i;
            else {
                if (!data[this_i].contains(rhs.data[rhs_i])) return false;
                ++this_i;
                ++rhs_i;
            }
        }
        return rhs_i == rhs.indices.size();
    }

    /// Returns true if this set and rhs share any bits.
    bool intersects(const SegmentBitVector& rhs) const noexcept {
        size_t this_i = 0, rhs_i = 0;
        while (this_i < indices.size() && rhs_i < rhs.indices.size()) {
            const auto this_ind = indices[this_i];
            const auto rhs_ind = rhs.indices[rhs_i];
            if (this_ind > rhs_ind) ++rhs_i;
            else if (this_ind < rhs_ind) ++this_i;
            else {
                if (this->data[this_i].intersects(rhs.data[rhs_i])) return true;
                ++this_i;
                ++rhs_i;
            }
        }
        return false;
    }

    bool operator==(const SegmentBitVector& rhs) const noexcept {
        if (indices.size() != rhs.indices.size()) return false;
        for (size_t i = 0; i < indices.size(); ++i)
            if (indices[i] != rhs.indices[i] || data[i] != rhs.data[i])
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
        indices.reserve(rhs_size);
        data.reserve(rhs_size);
        size_t this_i = 0, rhs_i = 0;
        bool changed = false;
        while (this_i < indices.size() && rhs_i < rhs_size) {
            const auto this_ind = indices[this_i];
            const auto rhs_ind = rhs.indices[rhs_i];
            if (this_ind < rhs_ind) ++this_i;
            else if (this_ind > rhs_ind) {
                indices.insert(indices.begin() + this_i, rhs_ind);
                data.insert(data.begin() + this_i, rhs.data[rhs_i]);
                ++this_i, ++rhs_i;
                changed = true;
            } else {
                changed |= (data[this_i] |= rhs.data[rhs_i]);
                ++this_i;
                ++rhs_i;
            }
        }
        if (rhs_i < rhs_size) {
            // append remaining elements from rhs
            indices.insert(indices.end(), rhs.indices.begin() + rhs_i,
                           rhs.indices.end());
            data.insert(data.end(), rhs.data.begin() + rhs_i, rhs.data.end());
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
            const auto this_ind = indices[this_i];
            const auto rhs_ind = rhs.indices[rhs_i];
            if (this_ind < rhs_ind) {
                indices.erase(indices.begin() + this_i);
                data.erase(data.begin() + this_i);
                changed = true;
            } else if (this_ind > rhs_ind) ++rhs_i;
            else {
                const auto [cur_changed, zeroed] =
                    (data[this_i] &= rhs.data[rhs_i]);
                changed |= cur_changed;
                if (zeroed) { // remove empty segment, this_i don't change
                    indices.erase(indices.begin() + this_i);
                    data.erase(data.begin() + this_i);
                } else ++this_i;
                ++rhs_i;
            }
        }
        if (this_i < size()) {
            // remove remaining elements from this
            indices.erase(indices.begin() + this_i, indices.end());
            data.erase(data.begin() + this_i, data.end());
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
            const auto this_ind = indices[this_i];
            const auto rhs_ind = rhs.indices[rhs_i];
            if (this_ind < rhs_ind) ++this_i;
            else if (this_ind > rhs_ind) ++rhs_i;
            else {
                const auto [cur_changed, zeroed] =
                    (data[this_i] -= rhs.data[rhs_i]);
                changed |= cur_changed;
                if (zeroed) { // remove empty segment, this_i don't change
                    indices.erase(indices.begin() + this_i);
                    data.erase(data.begin() + this_i);
                } else ++this_i;
                ++rhs_i;
            }
        }
        return changed;
    }
    bool intersectWithComplement(const SegmentBitVector& rhs) {
        return *this -= rhs;
    }

    bool intersectWithComplement(const SegmentBitVector& lhs,
                                 const SegmentBitVector& rhs) {
        // TODO: inefficient!
        *this = lhs;
        return intersectWithComplement(rhs);
    }
};
} // namespace SVF
