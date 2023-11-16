/*
 * @Author: lab pc yjxkwp@foxmail.com
 * @Date: 2023-05-04 12:57:27
 * @LastEditors: lab pc yjxkwp@foxmail.com
 * @LastEditTime: 2023-05-08 19:15:53
 * @FilePath: /designauto/da-sha/sha-implicit-modeling/field_tpms.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include <cmath>
#include <functional>
namespace da {
namespace sha {

namespace FieldTPMSFunctions {
using FieldTPMSFunction = std::function<double(double x, double y, double z, double coeff)>;
inline FieldTPMSFunction G(double t = 0) {
  return [=](double x, double y, double z, double coeff) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    double r = sqrt(x * x + y * y);
    double theta = 32 * atan(y / x);
    x = r;
    y = theta;
    return sin(x) * cos(y) + sin(z) * cos(x) + sin(y) * cos(z);
  };
}

inline FieldTPMSFunction G_rec(double t = 0) {
  return [=](double x, double y, double z, double coeff) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    return sin(x) * cos(y) + sin(z) * cos(x) + sin(y) * cos(z);
  };
}

inline FieldTPMSFunction Schwarzp(double t = 0) {
  return [=](double x, double y, double z, double coeff) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;

    return sin(x) * sin(y) * sin(z) + sin(x) * cos(y) * cos(z) + cos(x) * sin(y) * cos(z) +
           cos(x) * cos(y) * sin(z) + t;
  };
}

inline FieldTPMSFunction DoubleP(double t = 0) {
  return [=](double x, double y, double z, double coeff) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    return cos(x) * cos(y) + cos(y) * cos(z) + cos(x) * cos(z) +
           0.35 * (cos(2 * x) + cos(2 * y) + cos(2 * z)) + t;
  };
}

inline FieldTPMSFunction Schwarzd(double t = 0) {
  return [=](double x, double y, double z, double coeff) -> double {
    x = x * coeff;
    y = y * coeff;
    z = (z + 0.5) * coeff;

    return -(sin(x) * sin(y) * sin(z) + sin(x) * cos(y) * cos(z) + cos(x) * sin(y) * cos(z) +
           cos(x) * cos(y) * sin(z) + t);
  };
}

inline FieldTPMSFunction DoubleD(double t = 0) {
  return [=](double x, double y, double z, double coeff) -> double {
    x = x * coeff;
    y = y * coeff;
    z = z * coeff;
    return -1 * (cos(x) * cos(y) + cos(y) * cos(z) + cos(x) * cos(z)) -
           1 * (sin(x) * sin(y) * sin(z)) + t;
  };
}

}  // namespace FieldTPMSFunctions
}  // namespace sha
}  // namespace da