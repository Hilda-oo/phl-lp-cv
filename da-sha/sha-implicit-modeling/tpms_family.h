#pragma once

#include <cmath>
#include <functional>

namespace da {
namespace sha {

namespace TPMSFunctions {
using TPMSFunction = std::function<double(double x, double y, double z)>;
inline TPMSFunction Schwarzp(double coeff = 0, double t = 0) {
  return [=](double x, double y, double z) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    return sin(x) * sin(y) * sin(z) + sin(x) * cos(y) * cos(z) + cos(x) * sin(y) * cos(z) +
           cos(x) * cos(y) * sin(z) + t;
  };
}

inline TPMSFunction DoubleP(double coeff = 0, double t = 0) {
  return [=](double x, double y, double z) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    return cos(x) * cos(y) + cos(y) * cos(z) + cos(x) * cos(z) +
           0.35 * (cos(2 * x) + cos(2 * y) + cos(2 * z)) + t;
  };
}

inline TPMSFunction Schwarzd(double coeff = 0, double t = 0) {
  return [=](double x, double y, double z) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    return sin(x) * sin(y) * sin(z) + sin(x) * cos(y) * cos(z) + cos(x) * sin(y) * cos(z) +
           cos(x) * cos(y) * sin(z) + t;
  };
}

inline TPMSFunction DoubleD(double coeff = 0, double t = 0) {
  return [=](double x, double y, double z) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    return -1 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(x) * cos(z)) -
           1 * (sin(x) * sin(y) * sin(z)) + t;
  };
}

}  // namespace TPMSFunctions
}  // namespace sha
}  // namespace da
