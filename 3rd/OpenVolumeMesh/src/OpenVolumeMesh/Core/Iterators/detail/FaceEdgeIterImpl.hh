#pragma once

#include <OpenVolumeMesh/Core/Handles.hh>
#include <OpenVolumeMesh/Config/Export.hh>
#include <OpenVolumeMesh/Core/Iterators/BaseCirculator.hh>

#ifndef NDEBUG
#include <iostream>
#endif

namespace OpenVolumeMesh::detail {

class OVM_EXPORT FaceEdgeIterImpl : public BaseCirculator<FaceHandle, EdgeHandle> {
public:
    using BaseIter = BaseCirculator<FaceHandle, EdgeHandle>;

    typedef FaceHandle CenterEntityHandle;

    FaceEdgeIterImpl(const FaceHandle& _ref_h, const TopologyKernel* _mesh, int _max_laps = 1);
    FaceEdgeIterImpl& operator++();
    FaceEdgeIterImpl& operator--();

private:
    std::vector<HalfEdgeHandle> const& halfedges_;
    size_t cur_index_ = 0;
};



} // namespace OpenVolumeMesh::detail
