#pragma once

#include "sha-base-framework/frame.h"
#include "sha-surface-mesh/matmesh.h"

namespace da {
namespace sha {
double LinearInterpolateFor1D(double v1, double v2, double x1, double x2, double x);

double LinearInterpolateFor2D(double v1, double v2, double v3, double v4, double x1, double x2,
                              double x, double y1, double y2, double y);

double LinearInterpolateFor3D(double v1, double v2, double v3, double v4, double v5, double v6,
                              double v7, double v8, double x1, double x2, double x, double y1,
                              double y2, double y, double z1, double z2, double z);

MatMesh2 LoadStructureVF(index_t struct_type, const fs_path &microstructure_base_path);

auto ReadTrianglePatchFromMicrostructureBase(const std::vector<index_t> &types,
                                             const fs_path &microstructure_base_path)
    -> std::map<index_t, MatMesh3>;
}  // namespace sha
}  // namespace da
