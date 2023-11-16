#pragma once

#include <Eigen/Eigen>
#include "matmesh.h"

#include <utility>

namespace da {
namespace sha {
/**
 * @description:
 * @param {MatMesh3} &matmesh
 * @param {size_t} num_samples
 * @return {sample point coordiniates, triangles points from}
 */
auto SamplePointsOnMeshSurfaceUniformly(const MatMesh3 &matmesh, size_t num_samples)
    -> std::pair<Eigen::MatrixXd, Eigen::VectorXi>;

Eigen::MatrixXd SamplePointsInMeshVolumeUniformly(const MatMesh3 &matmesh, size_t num_samples);

Eigen::MatrixXd SamplePointsInAlignedBox3dUniformly(const Eigen::AlignedBox3d &aligned_box, size_t num_samples);
}  // namespace sha
}  // namespace da
