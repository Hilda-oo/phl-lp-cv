#pragma once
#include <cmath>
#include <functional>
namespace da {
namespace sha {

namespace FieldFunctions {
using FieldFunction = std::function<double(double x, double y, double z)>;
inline FieldFunction NoField() {
    return [=](double x, double y, double z) -> double {
        return 0;
    };
}

inline FieldFunction F1() {
    return [=](double x, double y, double z) -> double {
        return (x * x + y * y);
    };
}

inline FieldFunction F2() {
    return [=](double x, double y, double z) -> double {
        return ((x-1)*(x-1)+(y-1)*(y-1)+(z-1)*(z-1));
    };
}

inline FieldFunction F3() {
    return [=](double x, double y, double z) -> double {
        return x + 1;
    };
}

}  // namespace FieldFunctions
}  // namespace sha
}  // namespace da
