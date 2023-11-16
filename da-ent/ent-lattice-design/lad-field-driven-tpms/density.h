#pragma once

#include <Eigen/Eigen>
#include "sha-base-framework/frame.h"
#include "sha-volume-mesh/matmesh.h"

namespace da {
void ComputeRhosForBackgroundMesh(fs_path backgroundmesh_dir);

auto ComputeRhos(const std::vector<TetrahedralMatMesh> &tetrahedrons)
    -> std::vector<Eigen::VectorXd>;

// TODO: provide ComputeMinDistance
/**
 * compute minimum distance d from point to model
 * d > 0 if point inside model
 * d < 0 if point outside model
 * @param point
 * @return minimum distance d
 */
double ComputeMinDistance(const Eigen::Vector3d &point);
}  // namespace da
