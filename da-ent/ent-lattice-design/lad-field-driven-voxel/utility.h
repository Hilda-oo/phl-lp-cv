#pragma once

#include "sha-surface-mesh/matmesh.h"

namespace da {
MatMesh2 ComputeCommonLinesFromTwoMeshes(const MatMesh3 &mesh1, const MatMesh3 &mesh2);
}
