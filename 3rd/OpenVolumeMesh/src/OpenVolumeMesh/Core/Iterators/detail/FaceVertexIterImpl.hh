#pragma once

#include <OpenVolumeMesh/Core/Handles.hh>
#include <OpenVolumeMesh/Config/Export.hh>
#include <OpenVolumeMesh/Core/Iterators/BaseCirculator.hh>
#include <OpenVolumeMesh/Core/Iterators/HalfFaceVertexIter.hh>

#ifndef NDEBUG
#include <iostream>
#endif

namespace OpenVolumeMesh::detail {

class OVM_EXPORT FaceVertexIterImpl : public HalfFaceVertexIter {
public:

    typedef FaceHandle CenterEntityHandle;
    FaceVertexIterImpl(const FaceHandle& _ref_h, const TopologyKernel* _mesh, int _max_laps = 1);
};


} // namespace OpenVolumeMesh::detail
