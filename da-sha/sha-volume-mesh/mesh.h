#pragma once

#include <optional>
#include <OpenVolumeMesh/Mesh/HexahedralMesh.hh>
#include <OpenVolumeMesh/Mesh/PolyhedralMesh.hh>
#include <OpenVolumeMesh/Mesh/TetrahedralMesh.hh>

#include "sha-surface-mesh/mesh3.h"

namespace da {
namespace sha {
using HexahedralMesh  = OpenVolumeMesh::GeometricHexahedralMeshV3d;
using TetrahedralMesh = OpenVolumeMesh::GeometricTetrahedralMeshV3d;

// Topologic Mesh
using HexahedralTopoMesh  = OpenVolumeMesh::TopologicHexahedralMesh;
using TetrahedralTopoMesh = OpenVolumeMesh::TopologicTetrahedralMesh;

SurfaceMesh3 CreateSurfaceMesh3FromTetrahedralMesh(const TetrahedralMesh &tetrahedral_mesh);
SurfaceMesh3 CreateSurfaceMesh3FromHexahedralMesh(const HexahedralMesh &hexahedral_mesh);

SurfaceTopoMesh3 CreateSurfaceTopoMesh3FromTetrahedralTopoMesh(
    const TetrahedralTopoMesh &tetrahedral_topomesh);
SurfaceTopoMesh3 CreateSurfaceTopoMesh3FromHexahedralTopoMesh(
    const HexahedralTopoMesh &hexahedral_topomesh);
}  // namespace sha
using sha::HexahedralMesh;
using sha::HexahedralTopoMesh;
using sha::TetrahedralMesh;
using sha::TetrahedralTopoMesh;
}  // namespace da
