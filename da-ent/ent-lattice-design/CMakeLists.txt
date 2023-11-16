AddEntProject(lad-voxel-structure  sha-io-foundation sha-implicit-modeling sha-surface-mesh)
AddEntProject(lad-tpms-structure  sha-io-foundation sha-implicit-modeling sha-surface-mesh)

AddEntProject(lad-voronoi-structure  sha-io-foundation sha-implicit-modeling sha-surface-mesh sha-voronoi-foundation geogram)
target_include_directories(lad-voronoi-structure PRIVATE /usr/local/include/geogram1)

AddEntProject(lad-hexahedron-structure  sha-io-foundation sha-implicit-modeling tet sha-hexahedron-generation)

AddEntProject(lad-field-driven-voxel  sha-io-foundation sha-implicit-modeling sha-surface-mesh)
AddEntProject(lad-field-driven-tpms  sha-io-foundation sha-implicit-modeling sha-surface-mesh)
