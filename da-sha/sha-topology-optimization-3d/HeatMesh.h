//
// Created by cflin on 5/4/23.
//

#ifndef DESIGNAUTO_HEATMESH_H
#define DESIGNAUTO_HEATMESH_H

#include "Mesh.h"

namespace da::sha {
    namespace top {

        class HeatMesh : public Mesh {
        public:
            HeatMesh(int len_x, int len_y, int len_z) : Mesh(len_x, len_y, len_z, 1) {
            }
        };

    } // top
} // da::sha

#endif //DESIGNAUTO_HEATMESH_H
