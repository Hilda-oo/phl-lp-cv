#include "config.h"
#include "spdlog/spdlog.h"
#include "json.hpp"

using json = nlohmann::json;

namespace da {

void hasObject(const json &data, const std::string &key) {
  if (!data.count(key)) {
    spdlog::error("no \"{}\" in json file", key);
    exit(-1);
  }
}

bool Config::loadFromJSON(const std::string &p_filePath) {
  filePath = p_filePath;
  std::ifstream f(filePath);
  if (!f) {
    return false;
  }

  json data = json::parse(f);
  if (data.count("OptimizationExample")) {
    condition = data["OptimizationExample"];
  }

  if (data.count("level")) {
    level = data["level"];
  }

  if (data.count("path")) { // Path
    outputPath = data["path"]["outputPath"];
    meshFilePath = data["path"]["meshFilePath"];
    seedsPath = data["path"]["seedsPath"];
    backgroundCellsPath = data["path"]["backgroundCellsPath"];
    cellTetsPath = data["path"]["cellTetsPath"];
    cellPolyhedronPath = data["path"]["cellPolyhedronPath"];
  }

  if (data.count("material")) { // material
    YM = data["material"]["youngs_modulus"];
    PR = data["material"]["poisson_ratio"];
    TC = data["material"]["thermal_conductivity"];
    TEC = data["material"]["thermal_expansion_coefficient"];
  }
  if (data.count("model")) { // material
    radius[0] = data["model"]["radius"]["init"];
    radius[1] = data["model"]["radius"]["min"];
    radius[2] = data["model"]["radius"]["max"];
    shell = data["model"]["shell"];
    cellNum = data["model"]["cells"];
  }
  if (data.count("topology")) { // material
    para->E = data["topology"]["E"];
    para->max_loop = data["topology"]["max_loop"];
    para->volfrac = data["topology"]["volfrac"];
    para->r_min = data["topology"]["r_min"];
    para->penal = data["topology"]["penal"];
    para->T_ref = data["topology"]["T_ref"];
    para->T_limit = data["topology"]["T_limit"];
    para->R_E = data["topology"]["R_E"];
    para->R_lambda = data["topology"]["R_lambda"];
    para->R_beta = data["topology"]["R_beta"];
  }
  // mechanical boundary conditions
  if (data.count("mechanical_boundary_condition")) {
    const auto &mechBC = data["mechanical_boundary_condition"];
    // NBC
    hasObject(mechBC, "NBC");
    int NBCNum = mechBC["NBC"].size();
    for (int _i = 0; _i < NBCNum; ++_i) {
      const auto &NBCI = mechBC["NBC"][_i];
      hasObject(NBCI, "min");
      Eigen::Vector3d minBBox(NBCI["min"][0], NBCI["min"][1], NBCI["min"][2]);
      hasObject(NBCI, "max");
      Eigen::Vector3d maxBBox(NBCI["max"][0], NBCI["max"][1], NBCI["max"][2]);
      hasObject(NBCI, "val");
      Eigen::Vector3d force(NBCI["val"][0], NBCI["val"][1], NBCI["val"][2]);
      std::array<double, 2> timeRange = {
          0.0, std::numeric_limits<double>::infinity()}; // default value
      if (NBCI.count("timeRange")) {
        timeRange[0] = NBCI["timeRange"][0];
        timeRange[1] = NBCI["timeRange"][1];
      }
      mechanicalNeumannBCs.emplace_back(
          sha::NeumannBC(minBBox, maxBBox, force, timeRange));
    }

    // DBC
    hasObject(mechBC, "DBC");
    int DBCNum = mechBC["DBC"].size();
    for (int _i = 0; _i < DBCNum; ++_i) {
      const auto &DBCI = mechBC["DBC"][_i];
      hasObject(DBCI, "min");
      Eigen::Vector3d minBBox(DBCI["min"][0], DBCI["min"][1], DBCI["min"][2]);
      hasObject(DBCI, "max");
      Eigen::Vector3d maxBBox(DBCI["max"][0], DBCI["max"][1], DBCI["max"][2]);
      std::array<double, 2> timeRange = {
          0.0, std::numeric_limits<double>::infinity()}; // default value
      if (DBCI.count("timeRange")) {
        timeRange[0] = DBCI["timeRange"][0];
        timeRange[1] = DBCI["timeRange"][1];
      }
      mechanicalDirichletBCs.emplace_back(
          sha::DirichletBC(minBBox, maxBBox, timeRange));
    }
  }

  // thermal boundary conditions
  if (data.count("thermal_boundary_condition")) {
    const auto &thermalBC = data["thermal_boundary_condition"];
    hasObject(thermalBC, "NBC");
    int NBCNum = thermalBC["NBC"].size();
    for (int _i = 0; _i < NBCNum; ++_i) {
      const auto &NBCI = thermalBC["NBC"][_i];
      hasObject(NBCI, "min");
      Eigen::Vector3d minBBox(NBCI["min"][0], NBCI["min"][1], NBCI["min"][2]);
      hasObject(NBCI, "max");
      Eigen::Vector3d maxBBox(NBCI["max"][0], NBCI["max"][1], NBCI["max"][2]);
      hasObject(NBCI, "heat_flux");
      double heat_flux = NBCI["heat_flux"];
      std::array<double, 2> timeRange = {
          0.0, std::numeric_limits<double>::infinity()}; // default value
      if (NBCI.count("timeRange")) {
        timeRange[0] = NBCI["timeRange"][0];
        timeRange[1] = NBCI["timeRange"][1];
      }
      thermalNeumannBCs.emplace_back(
          sha::NeumannBC(minBBox, maxBBox, heat_flux, timeRange));
    }

    hasObject(thermalBC, "DBC");
    int DBCNum = thermalBC["DBC"].size();
    for (int _i = 0; _i < DBCNum; ++_i) {
      const auto &DBCI = thermalBC["DBC"][_i];
      hasObject(DBCI, "min");
      Eigen::Vector3d minBBox(DBCI["min"][0], DBCI["min"][1], DBCI["min"][2]);
      hasObject(DBCI, "max");
      Eigen::Vector3d maxBBox(DBCI["max"][0], DBCI["max"][1], DBCI["max"][2]);
      hasObject(DBCI, "temperature");
      double temperature = DBCI["temperature"];
      std::array<double, 2> timeRange = {
          0.0, std::numeric_limits<double>::infinity()}; // default value
      if (DBCI.count("timeRange")) {
        timeRange[0] = DBCI["timeRange"][0];
        timeRange[1] = DBCI["timeRange"][1];
      }
      thermalDirichletBCs.emplace_back(
          sha::DirichletBC(minBBox, maxBBox, temperature, timeRange));
    }
  }
  return true;
}

void Config::backUpConfig(const std::string &p_filePath) {
  std::ifstream src(filePath, std::ios::binary);
  std::ofstream dst(p_filePath, std::ios::binary);
  dst << src.rdbuf();
}

} // namespace da
