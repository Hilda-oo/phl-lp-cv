#pragma once

#include <Eigen/Eigen>

#include <memory>

#include <array>
#include <queue>

#include <geogram/mesh/mesh.h>
#include "sha-base-framework/declarations.h"
#include "sha-base-framework/frame.h"
#include "sha-base-framework/logger.h"
#include "sha-surface-mesh/matmesh.h"
#include "sha-surface-mesh/mesh3.h"

namespace da::sha {

class OctreeNode : public std::enable_shared_from_this<OctreeNode> {
 public:
  explicit OctreeNode(const Eigen::AlignedBox3d &domain, int depth)
      : domain_(domain), depth_(depth) {
    for (index_t idx = 0; idx < 8; idx++) {
      sub_nodes_[idx] = nullptr;
    }
  }

  bool IsLeaf() const { return depth_ == 0; }

  void CollectIntersectedLeaves(const Eigen::AlignedBox3d &boundingbox,
                                std::vector<std::shared_ptr<OctreeNode>> &leaf_nodes) {
    if (!boundingbox.intersects(domain_)) return;
    if (IsLeaf()) {
      leaf_nodes.push_back(this->shared_from_this());
    } else {
      for (auto sub_node : sub_nodes_) {
        if (sub_node != nullptr) {
          sub_node->CollectIntersectedLeaves(boundingbox, leaf_nodes);
        }
      }
    }
  }

  void InsertBoundingBoxWithIndex(const Eigen::AlignedBox3d &boundingbox, const index_t index) {
    if (!boundingbox.intersects(domain_)) return;
    if (IsLeaf()) {
      boxes_with_index_.emplace_back(boundingbox, index);
    } else {
      for (auto sub_node : sub_nodes_) {
        if (sub_node != nullptr) {
          sub_node->InsertBoundingBoxWithIndex(boundingbox, index);
        }
      }  // end for
    }    // end if
  }

  void FindIndicesByBoundingBox(const Eigen::AlignedBox3d &boundingbox,
                                std::set<index_t> &indices) {
    if (!boundingbox.intersects(domain_)) return;
    if (IsLeaf()) {
      for (auto &[sub_box, index] : boxes_with_index_) {
        indices.insert(index);
      }
    } else {
      for (auto sub_node : sub_nodes_) {
        if (sub_node != nullptr) {
          sub_node->FindIndicesByBoundingBox(boundingbox, indices);
        }
      }  // end for
    }    // end if
  }

 public:
  std::array<std::shared_ptr<OctreeNode>, 8> sub_nodes_;

  Eigen::AlignedBox3d domain_;
  int depth_;

  std::vector<std::pair<Eigen::AlignedBox3d, index_t>> boxes_with_index_;
};

template <typename DateType = index_t>
class DynamicBoundingBoxOctree {
 public:
  explicit DynamicBoundingBoxOctree(const Eigen::AlignedBox3d &domain, size_t max_depth)
      : max_depth_(max_depth), root_(std::make_shared<OctreeNode>(domain, max_depth)) {
    // make tree
    std::queue<std::shared_ptr<OctreeNode>> nodes_queue;
    nodes_queue.push(root_);
    while (!nodes_queue.empty()) {
      auto node = nodes_queue.front();
      nodes_queue.pop();
      if (node == nullptr) continue;
      if (node->depth_ <= 0) continue;
      // subdivide
      auto half_size = node->domain_.sizes() / 2.0;
      for (int idx = 0; idx < 8; ++idx) {
        Eigen::Vector3d sub_min = {node->domain_.min().x() + (((idx & 4) > 0) ? half_size.x() : 0),
                                   node->domain_.min().y() + (((idx & 2) > 0) ? half_size.y() : 0),
                                   node->domain_.min().z() + (((idx & 1) > 0) ? half_size.z() : 0)};
        Eigen::Vector3d sub_max = sub_min + half_size;
        Eigen::AlignedBox3d sub_domain(sub_min, sub_max);
        auto new_sub_node     = std::make_shared<OctreeNode>(sub_domain, node->depth_ - 1);
        node->sub_nodes_[idx] = new_sub_node;
        nodes_queue.push(new_sub_node);
      }
    }
  }

  void InsertData(const Eigen::AlignedBox3d &boundingbox, const DateType &data) {
    index_t index = data_vector.size();
    data_vector.push_back(data);
    root_->InsertBoundingBoxWithIndex(boundingbox, index);
  }

  std::set<index_t> FindDataIndicesByBoundingBox(const Eigen::AlignedBox3d &boundingbox) {
    std::set<index_t> data_indices;
    root_->FindIndicesByBoundingBox(boundingbox, data_indices);
    return data_indices;
  }

  DateType &Data(index_t data_idx) { return data_vector[data_idx]; }
  const DateType &Data(index_t data_idx) const { return data_vector[data_idx]; }

 public:
  size_t max_depth_;
  std::shared_ptr<OctreeNode> root_;

 public:
  std::vector<DateType> data_vector;
};

template class DynamicBoundingBoxOctree<index_t>;

void ConvertMatmesh3ToGeoMesh(const MatMesh3 &mesh, GEO::Mesh &geo_mesh);
void ConvertGeoMeshToMatmesh3(const GEO::Mesh &geo_mesh, MatMesh3 &mesh);

}  // namespace da::sha