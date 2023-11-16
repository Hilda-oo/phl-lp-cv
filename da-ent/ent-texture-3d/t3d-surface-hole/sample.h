#pragma once

#include "sha-surface-mesh/matmesh.h"

namespace da {
void SampleInBoundingSquare(const MatMesh3 &matmesh, const Eigen::MatrixXd &uv,
                            size_t num_on_long_axis, Eigen::MatrixXd &mat_sample_points,
                            Eigen::MatrixXd &mat_sample_normals);

void SampleInBoundingSquareByCurvatureField(const MatMesh3 &matmesh, const Eigen::MatrixXd &uv,
                            size_t num_on_long_axis, Eigen::MatrixXd &mat_sample_points,
                            Eigen::MatrixXd &mat_sample_normals, Eigen::MatrixXd &mat_sample_principal_curvatures);
}
