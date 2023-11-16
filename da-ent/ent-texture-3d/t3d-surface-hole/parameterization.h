#pragma once

#include <set>
#include <vector>

#include "sha-surface-mesh/mesh3.h"
#include "sha-surface-mesh/matmesh.h"

namespace da {
void ParameterizeByTuttesMethod(const SurfaceMesh3 &mesh3, Eigen::MatrixXd &uv);

void ParameterizeByArapMethod(const SurfaceMesh3 &mesh3, Eigen::MatrixXd &uv);
}  // namespace da
