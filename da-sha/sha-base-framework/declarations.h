#pragma once

#include <boost/filesystem.hpp>
#include <boost/range.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/counting_range.hpp>
#include <boost/range/numeric.hpp>

namespace da {
using fs_path = boost::filesystem::path;
using size_t  = ::size_t;
using index_t = size_t;
}  // namespace da
