#pragma once
#include <vector>
#include <unordered_map>
#include <boost/functional/hash.hpp>


struct HashVector {
    std::size_t operator()(const std::vector<double>& v) const {
        return boost::hash_range(v.begin(), v.end());
    }
};


struct HashVectorIndex {
    std::size_t operator()(const std::vector<int>& v) const {
        return boost::hash_range(v.begin(), v.end());
    }
};

template <std::size_t DIM>
struct HashArrayIndex {
    std::size_t operator()(const std::array<int, DIM>& v) const {
        return boost::hash_range(v.begin(), v.end());
    }
};
