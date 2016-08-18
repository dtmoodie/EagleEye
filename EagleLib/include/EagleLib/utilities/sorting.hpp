#pragma once

#include <algorithm>
#include <vector>

// Sort value and key based on values, key will be reorganized such that 
/*template<typename T, typename U> void sort_key_value(std::vector<T>& value, std::vector<U>& key)
{
    std::sort(key.begin(), key.end(), [&key](size_t i1, size_t i2) {return )
}*/


template<typename T> std::vector<size_t> sort_index_ascending(const std::vector<T>& value)
{
    std::vector<size_t> indecies;
    indecies.resize(value.size());
    for(size_t i = 0; i < value.size(); ++i)
    {
        indecies[i] = i;
    }
    std::sort(indecies.begin(), indecies.end(), [&value](size_t i1, size_t i2){ return value[i1] < value[i2];});
    return indecies;
}

template<typename T> std::vector<size_t> sort_index_descending(const std::vector<T>& value)
{
    std::vector<size_t> indecies;
    indecies.resize(value.size());
    for (size_t i = 0; i < value.size(); ++i)
    {
        indecies[i] = i;
    }
    std::sort(indecies.begin(), indecies.end(), [&value](size_t i1, size_t i2) { return value[i1] > value[i2]; });
    return indecies;
}
