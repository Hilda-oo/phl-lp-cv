#include "lp_cvt_wrapper.h"
#include <LpCVT/algebra/F_Lp.h>
#include <LpCVT/combinatorics/clipped_VD.h>
#include <LpCVT/combinatorics/delaunay_CGAL.h>
#include <LpCVT/combinatorics/mesh.h>

namespace da {
static bool has_edge(const std::vector<unsigned int> &facet_vertex_index, unsigned int facet_begin,
                     unsigned int facet_end, unsigned int v1, unsigned int v2) {
  unsigned int facet_n = facet_end - facet_begin;
  for (unsigned int i = 0; i < facet_n; i++) {
    unsigned int w1 = facet_vertex_index[facet_begin + i];
    unsigned int w2 = facet_vertex_index[facet_begin + ((i + 1) % facet_n)];
    if (w1 == v1 && w2 == v2) {
      return true;
    }
  }
  return false;
}

int ConvertMatMesh3ToGeexMesh(const MatMesh3 &mesh, Geex::Mesh &geex_mesh) {
  geex_mesh.clear();
  const size_t num_vertices = mesh.NumVertices();
  const size_t num_faces    = mesh.NumFaces();
  std::vector<Geex::vec3> vertex;
  std::vector<std::vector<unsigned int> > star;

  for (index_t vtx_idx = 0; vtx_idx < num_vertices; vtx_idx++) {
    Geex::vec3 point(mesh.mat_coordinates(vtx_idx, 0), mesh.mat_coordinates(vtx_idx, 1),
                     mesh.mat_coordinates(vtx_idx, 2));
    vertex.push_back(point);
    star.push_back(std::vector<unsigned int>());
  }
  for (index_t face_idx = 0; face_idx < num_faces; face_idx++) {
    std::vector<int> cur_facet = {mesh.mat_faces(face_idx, 0), mesh.mat_faces(face_idx, 1),
                                  mesh.mat_faces(face_idx, 2)};
    unsigned int f             = geex_mesh.nb_facets();
    geex_mesh.begin_facet();
    for (unsigned int i = 0; i < cur_facet.size(); i++) {
      unsigned int v = cur_facet[i];
      geex_mesh.add_vertex(Geex::VertexEdge(vertex[v]));
      geex_mesh.top_vertex().set_flag(Geex::VertexEdge::ORIGINAL);
      geex_mesh.vertex_index_.push_back(v);
      star[v].push_back(f);
    }
    geex_mesh.end_facet();
  }
  geex_mesh.original_vertices_.resize(vertex.size());
  std::copy(vertex.begin(), vertex.end(), geex_mesh.original_vertices_.begin());
  // Step 2: compute facet adjacencies
  for (unsigned int f = 0; f < geex_mesh.nb_facets(); f++) {
    unsigned int facet_base = geex_mesh.facet_begin(f);
    unsigned int facet_n    = geex_mesh.facet_size(f);

    for (unsigned int i = 0; i < facet_n; i++) {
      unsigned int v1                    = facet_base + i;
      unsigned int v2                    = facet_base + ((i + 1) % facet_n);
      unsigned int gv1                   = geex_mesh.vertex_index_[v1];
      unsigned int gv2                   = geex_mesh.vertex_index_[v2];
      const std::vector<unsigned int> &S = star[gv1];
      for (unsigned int k = 0; k < S.size(); k++) {
        unsigned int g = S[k];
        if (g != f && has_edge(geex_mesh.vertex_index_, geex_mesh.facet_begin(g),
                               geex_mesh.facet_end(g), gv2, gv1)) {
          geex_mesh.vertex(v1).f = g;
          break;
        }
      }
    }
  }

  // Step 3: assign facet ids
  for (unsigned int f = 0; f < geex_mesh.nb_facets(); f++) {
    geex_mesh.facet_info(f).id = f;
  }

  // Step 4: initialize symbolic information
  geex_mesh.init_symbolic_vertices();

  // Just for checking
  unsigned int nb_borders = 0;
  for (unsigned int i = 0; i < geex_mesh.nb_vertices(); i++) {
    if (geex_mesh.vertex(i).f < 0) {
      nb_borders++;
    }
  }

  double vol             = geex_mesh.signed_volume();
  geex_mesh.orientation_ = (vol > 0.0);
  std::cerr << "Mesh loaded, nb_facets = " << geex_mesh.nb_facets()
            << " nb_borders = " << nb_borders << " signed volume = " << vol << std::endl;
  if (!geex_mesh.orientation_ && nb_borders == 0) {
    std::cerr << " WARNING ! orientation is negative" << std::endl;
  }
  return nb_borders;
}

void ConvertMatVertexToGeexVec3(const Eigen::MatrixXd &mat_vertex,
                                std::vector<Geex::vec3> &vec_vertex) {
  const int vertex_num = mat_vertex.rows();
  vec_vertex.resize(0);
  vec_vertex.resize(vertex_num);
  for (index_t vI = 0; vI < vertex_num; ++vI) {
    Eigen::Vector3d vertex = mat_vertex.row(vI);
    vec_vertex.emplace_back(Geex::vec3(vertex(0), vertex(1), vertex(2)));
  }
}

void GetCombinatorics(Geex::Mesh *M, const std::vector<Geex::vec3> &pts, std::vector<int> &I,
                      std::vector<Geex::vec3> &C, std::vector<int> &F, bool volume) {
  Geex::Delaunay *delaunay = Geex::Delaunay::create("CGAL");
  delaunay->set_vertices(pts);
  Geex::ClippedVoronoiDiagram CVD(delaunay, M);

  class MemorizeIndices {
   public:
    MemorizeIndices(std::vector<int> &I_in, std::vector<Geex::vec3> &C_in) : I(I_in), C(C_in) {
      I.resize(0);
      C.resize(0);
    }

    void operator()(unsigned int i, int j, const Geex::VertexEdge &v1, const Geex::VertexEdge &v2,
                    const Geex::VertexEdge &v3) const {
      I.push_back(i);
      I.push_back(v1.sym[2]);
      I.push_back(v1.sym[1]);
      I.push_back(v1.sym[0]);
      I.push_back(v2.sym[2]);
      I.push_back(v2.sym[1]);
      I.push_back(v2.sym[0]);
      I.push_back(v3.sym[2]);
      I.push_back(v3.sym[1]);
      I.push_back(v3.sym[0]);
      C.push_back(v1);
      C.push_back(v2);
      C.push_back(v3);
    }

   private:
    std::vector<int> &I;
    std::vector<Geex::vec3> &C;
  };

  CVD.for_each_triangle(MemorizeIndices(I, C));
  delete delaunay;
}

}  // namespace da

// for generating Clipped Voronoi Diagram with shrink
namespace da {
class SavePrimalTriangle {
 public:
  SavePrimalTriangle(std::ofstream &out) : out_(&out) {}
  void operator()(unsigned int i, unsigned int j, unsigned int k) const {
    (*out_) << "f " << i + 1 << " " << j + 1 << " " << k + 1 << std::endl;
  }

 private:
  std::ofstream *out_;
};

void save_RDT(Geex::RestrictedVoronoiDiagram &RVD, const std::string &filename) {
  std::ofstream out(filename.c_str());
  for (unsigned int i = 0; i < RVD.delaunay()->nb_vertices(); i++) {
    out << "v " << RVD.delaunay()->vertex(i) << std::endl;
  }
  RVD.for_each_primal_triangle(SavePrimalTriangle(out));
  out.close();
}

class SaveRVDFacets {
 public:
  SaveRVDFacets(std::ostream &out) : out_(out), cur_v_(1), cur_f_(1) {
    out << "# attribute chart facet integer" << std::endl;
  }
  void operator()(unsigned int iv, Geex::Mesh *M) const {
    for (unsigned int f = 0; f < M->nb_facets(); f++) {
      for (unsigned int i = M->facet_begin(f); i < M->facet_end(f); i++) {
        const Geex::vec3 &v = M->vertex(i);
        out_ << "v " << v << std::endl;
      }
      out_ << "f ";
      for (unsigned int i = M->facet_begin(f); i < M->facet_end(f); i++) {
        out_ << cur_v_ << " ";
        cur_v_++;
      }
      out_ << std::endl;
      out_ << "# attrs f " << cur_f_ << " " << iv << std::endl;
      cur_f_++;
    }
  }

 private:
  std::ostream &out_;
  mutable unsigned int cur_v_;
  mutable unsigned int cur_f_;
};

void save_RVD(Geex::RestrictedVoronoiDiagram &RVD, const std::string &filename) {
  std::ofstream out(filename.c_str());
  if (!out) {
    std::cerr << "could not open file." << std::endl;
    return;
  }
  bool sym_backup = RVD.symbolic();
  RVD.set_symbolic(true);
  RVD.for_each_facet(SaveRVDFacets(out));
  RVD.set_symbolic(sym_backup);
}

class SaveClippedVDFacets {
 public:
  SaveClippedVDFacets(Geex::Delaunay *delaunay, std::ostream &out, double shrink)
      : delaunay_(delaunay), out_(out), shrink_(shrink), cur_(1) {
    out << "# attribute chart facet integer" << std::endl;
  }
  void operator()(unsigned int i, int j, const Geex::vec3 &p1, const Geex::vec3 &p2,
                  const Geex::vec3 &p3) const {
    Geex::vec3 x0 = delaunay_->vertex(i);
    out_ << "v " << x0 + shrink_ * (p1 - x0) << std::endl;
    out_ << "v " << x0 + shrink_ * (p2 - x0) << std::endl;
    out_ << "v " << x0 + shrink_ * (p3 - x0) << std::endl;
    out_ << "f " << cur_ << " " << cur_ + 1 << " " << cur_ + 2 << std::endl;
    cur_ += 3;
    out_ << "# attrs f " << ((cur_ - 1) / 3) << " " << i << std::endl;
  }

 private:
  Geex::Delaunay *delaunay_;
  std::ostream &out_;
  double shrink_;
  mutable unsigned int cur_;
};

void save_clippedVD(Geex::ClippedVoronoiDiagram &CVD, const std::string &filename, double shrink) {
  std::ofstream out(filename.c_str());
  if (!out) {
    std::cerr << "could not open file." << std::endl;
    return;
  }

  CVD.for_each_triangle(SaveClippedVDFacets(CVD.delaunay(), out, shrink));
}

}  // namespace da

namespace da {
LpNormCVTWrapper::LpNormCVTWrapper(const MatMesh3 &mesh, const int P) : P_(P) {
  boundary_mesh_          = new Geex::Mesh;
  unsigned int nb_borders = ConvertMatMesh3ToGeexMesh(mesh, *boundary_mesh_);
  Q.resize(boundary_mesh_->nb_facets());
  for (unsigned int i = 0; i < boundary_mesh_->nb_facets(); i++) {
    Q[i] = boundary_mesh_->facet_plane(i);
  }
}

LpNormCVTWrapper::~LpNormCVTWrapper() {
  delete this->boundary_mesh_;
  this->boundary_mesh_ = nullptr;
}

double LpNormCVTWrapper::EvaluateLpCVT(const Eigen::MatrixXd &mat_seeds,
                                       const std::vector<Eigen::Matrix3d> &anisotropyMatrix,
                                       Eigen::VectorXd &dL) {
  const size_t num_seeds = mat_seeds.rows();
  std::vector<Geex::vec3> variables(num_seeds);

  for (index_t seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
    variables[seed_idx].x = mat_seeds(seed_idx, 0);
    variables[seed_idx].y = mat_seeds(seed_idx, 1);
    variables[seed_idx].z = mat_seeds(seed_idx, 2);
  }

  std::vector<int> I;
  std::vector<Geex::vec3> C;
  std::vector<int> F;

  GetCombinatorics(boundary_mesh_, variables, I, C, F, true);
  unsigned int nb_integration_simplices = (unsigned int)I.size() / 10;

  // load matrix
  std::vector<Geex::mat3> anisotropy_matries(nb_integration_simplices);
  for (index_t simplix_idx = 0; simplix_idx < nb_integration_simplices; ++simplix_idx) {
    Eigen::Matrix3d mat_anisotropy = anisotropyMatrix.at(simplix_idx);

    // for (int i_idx = 0; i_idx < 3; i_idx++) {
    //   for (int j_idx = 0; j_idx < 3; j_idx++) {
    //     anisotropy_matries[simplix_idx](i_idx, j_idx) = mat_anisotropy(i_idx, j_idx);
    //   }
    // }
      Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat_anisotropy, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d sigma = svd.singularValues().asDiagonal();
      sigma                 = sigma.cwiseSqrt();
      Eigen::Matrix3d UU  = svd.matrixU();
      Eigen::Matrix3d VV = svd.matrixV();
      Eigen::Matrix3d MM = sigma * 4;
      for (int i_idx = 0; i_idx < 3; i_idx++) {
        for (int j_idx = 0; j_idx < 3; j_idx++) {
          anisotropy_matries[simplix_idx](i_idx, j_idx) = MM(i_idx, j_idx);
        }
      }
  }

  // // load oval M
  // for (unsigned int i = 0; i < nb_integration_simplices; i++) {
  //   anisotropy_matries[i].load_identity();
  //   Geex::vec3 seed = variables[I[i * 10]];
  //   // double l = seed.length();
  //   double p = ::sqrt((seed.x * seed.x) + (seed.y * seed.y) + (seed.z * seed.z));
  //   // double theta = ::acos(p / l);
  //   Eigen::Matrix3d G;
  //   G << p * p, 0, 0,  //
  //       0, p * p, 0,   //
  //       0, 0, 1;       //
  //   Eigen::JacobiSVD<Eigen::Matrix3d> svd(G, Eigen::ComputeFullU | Eigen::ComputeFullV);
  //   Eigen::Matrix3d sigma = svd.singularValues().asDiagonal();
  //   sigma                 = sigma.cwiseSqrt();
  //   // sigma = sigma.cwiseSqrt();
  //   Eigen::Matrix3d W  = svd.matrixU();
  //   Eigen::Matrix3d MM = sigma * 4;
  //   for (int i_idx = 0; i_idx < 3; i_idx++) {
  //     for (int j_idx = 0; j_idx < 3; j_idx++) {
  //       anisotropy_matries[i](i_idx, j_idx) = MM(i_idx, j_idx);
  //     }
  //   }
  //   // M[i](0, 0) = p * p;
  //   // M[i](0, 1) = 0;
  //   // M[i](0, 2) = 0;
  //   // M[i](1, 0) = 0;
  //   // M[i](1, 1) = p * p;
  //   // M[i](1, 2) = 0;
  //   // M[i](2, 0) = 0;
  //   // M[i](2, 1) = 0;
  //   // M[i](2, 2) = 1;
  // }
  std::vector<double> gradients(num_seeds * 3);

  double f = Geex::compute_F_Lp(true, P_, boundary_mesh_, I, C, variables, Q, anisotropy_matries,
                                gradients);
  dL.resize(num_seeds * 3);
  std::copy_n(gradients.begin(), num_seeds * 3, dL.data());
  return f;
}

auto LpNormCVTWrapper::GetQuerySeeds(const Eigen::MatrixXd &mat_seeds) -> Eigen::MatrixXd {
  // spdlog::info("get query points");
  const size_t num_seeds = mat_seeds.rows();
  std::vector<Geex::vec3> variables(num_seeds);

  for (index_t seed_idx = 0; seed_idx < num_seeds; seed_idx++) {
    variables[seed_idx].x = mat_seeds(seed_idx, 0);
    variables[seed_idx].y = mat_seeds(seed_idx, 1);
    variables[seed_idx].z = mat_seeds(seed_idx, 2);
  }

  std::vector<int> I;
  std::vector<Geex::vec3> C;
  std::vector<int> F;

  GetCombinatorics(boundary_mesh_, variables, I, C, F, true);
  unsigned int nb_integration_simplices = (unsigned int)I.size() / 10;

  Eigen::MatrixXd seeds(nb_integration_simplices, 3);
  Eigen::MatrixXd c_vertex(nb_integration_simplices, 3);

  // get seeds
  for (index_t simplix_idx = 0; simplix_idx < nb_integration_simplices; ++simplix_idx) {
    Geex::vec3 seed           = variables[I[simplix_idx * 10]];
    Geex::vec3 vertex         = C[simplix_idx * 3];
    c_vertex.row(simplix_idx) = Eigen::Vector3d(vertex.x, vertex.y, vertex.z);
    seeds.row(simplix_idx)    = Eigen::Vector3d(seed.x, seed.y, seed.z);
  }
  return seeds;
}

void LpNormCVTWrapper::WriteShrinkCVD(const fs_path &fpath, const Eigen::MatrixXd &seeds,
                                      double shrink) {
  Geex::Delaunay *delaunay = Geex::Delaunay::create("CGAL");
  // Geex::RestrictedVoronoiDiagram RVD(delaunay, boundary_mesh_);
  Geex::ClippedVoronoiDiagram CVD(delaunay, boundary_mesh_);

  std::vector<Geex::vec3> variables;
  ConvertMatVertexToGeexVec3(seeds, variables);
  delaunay->set_vertices(variables);

  // const std::string rvd_path = fpath.string() + "/rvd.obj";
  // save_RVD(RVD, rvd_path);

  // const std::string rdt_path = fpath.string() + "/rdt.obj";
  // save_RDT(RVD, rdt_path);

  const std::string cvd_path   = fpath.string() + "/CVD.obj";
  save_clippedVD(CVD, cvd_path, shrink);
  delete delaunay;
}
}  // namespace da