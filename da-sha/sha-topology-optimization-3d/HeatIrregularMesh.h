//
// Created by cflin on 5/4/23.
//

#ifndef DESIGNAUTO_HEATIRREGULARMESH_H
#define DESIGNAUTO_HEATIRREGULARMESH_H

#include "IrregularMesh.h"

namespace da {
    namespace sha {
        namespace top {
            class HeatIrregularMesh : public IrregularMesh {
            public:
                HeatIrregularMesh(const fs_path &arbitrary_stl_path, const fs_path &chosen_stl_path,
                                  double relative_length_of_voxel) : IrregularMesh(arbitrary_stl_path, chosen_stl_path,
                                                                                   relative_length_of_voxel, 1) {}
            };

        } // da
    } // sha
} // top

#endif //DESIGNAUTO_HEATIRREGULARMESH_H
