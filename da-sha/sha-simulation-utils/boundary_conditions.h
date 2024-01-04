#pragma once

#include <Eigen/Eigen>
#include <utility>
#include <limits>

namespace da::sha {

struct DirichletBC {
  DirichletBC(Eigen::Vector3d p_relMinBBox, Eigen::Vector3d p_relMaxBBox,
              const std::array<double, 2> &p_timeRange = {0.0,
                                                          std::numeric_limits<double>::infinity()})
      : relMinBBox(std::move(p_relMinBBox)),
        relMaxBBox(std::move(p_relMaxBBox)),
        timeRange(p_timeRange) {}

  DirichletBC(Eigen::Vector3d p_relMinBBox, Eigen::Vector3d p_relMaxBBox,
              double p_temperature,
              const std::array<double, 2> &p_timeRange =
                  {0.0, std::numeric_limits<double>::infinity()})
      : relMinBBox(std::move(p_relMinBBox)),
        relMaxBBox(std::move(p_relMaxBBox)), temperature(p_temperature),
        timeRange(p_timeRange) {}        

  void calcAbsBBox(const Eigen::Vector3d &modelMinBBox, const Eigen::Vector3d &modelMaxBBox) {
    Eigen::Vector3d modelLength = modelMaxBBox - modelMinBBox;
    absMinBBox = modelMinBBox + (modelLength.array() * relMinBBox.array()).matrix();
    absMaxBBox = modelMinBBox + (modelLength.array() * relMaxBBox.array()).matrix();
  }

  bool inDBC(const Eigen::Vector3d &p) {
    return (p.array() >= absMinBBox.array()).all() && (p.array() <= absMaxBBox.array()).all();
  }

  bool isActive(double stepStartTime) const {
    return stepStartTime >= timeRange[0] && stepStartTime < timeRange[1];
  }

  std::array<double, 2> timeRange = {0.0, std::numeric_limits<double>::infinity()};

  Eigen::Vector3d relMinBBox;
  Eigen::Vector3d relMaxBBox;

  Eigen::Vector3d absMinBBox;
  Eigen::Vector3d absMaxBBox;
  double temperature;

};

struct NeumannBC {
  NeumannBC(Eigen::Vector3d p_relMinBBox, Eigen::Vector3d p_relMaxBBox, Eigen::Vector3d p_force,
            const std::array<double, 2> &p_timeRange = {0.0,
                                                        std::numeric_limits<double>::infinity()})
      : relMinBBox(std::move(p_relMinBBox)),
        relMaxBBox(std::move(p_relMaxBBox)),
        force(std::move(p_force)),
        timeRange(p_timeRange) {}
  NeumannBC(Eigen::Vector3d p_relMinBBox, Eigen::Vector3d p_relMaxBBox,
            double p_flux,
            const std::array<double, 2> &p_timeRange =
                {0.0, std::numeric_limits<double>::infinity()})
      : relMinBBox(std::move(p_relMinBBox)),
        relMaxBBox(std::move(p_relMaxBBox)), flux(p_flux),
        timeRange(p_timeRange) {}        

  void calcAbsBBox(const Eigen::Vector3d &modelMinBBox, const Eigen::Vector3d &modelMaxBBox) {
    Eigen::Vector3d modelLength = modelMaxBBox - modelMinBBox;
    absMinBBox = modelMinBBox + (modelLength.array() * relMinBBox.array()).matrix();
    absMaxBBox = modelMinBBox + (modelLength.array() * relMaxBBox.array()).matrix();
  }

  bool inNBC(const Eigen::Vector3d &p) {
    return (p.array() >= absMinBBox.array()).all() && (p.array() <= absMaxBBox.array()).all();
  }

  bool isActive(double stepStartTime) const {
    return stepStartTime >= timeRange[0] && stepStartTime < timeRange[1];
  }

  std::array<double, 2> timeRange = {0.0, std::numeric_limits<double>::infinity()};

  Eigen::Vector3d relMinBBox;
  Eigen::Vector3d relMaxBBox;

  Eigen::Vector3d absMinBBox;
  Eigen::Vector3d absMaxBBox;

  Eigen::Vector3d force;
  double flux;
};

}  // namespace da::sha
