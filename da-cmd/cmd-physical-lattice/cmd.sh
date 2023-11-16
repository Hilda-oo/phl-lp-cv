#!/usr/bin/env bash

project_name=cmd-lattice-design

asset_dir_path=../../da-cmd/${project_name}/assets
workplace_dir_path=../Workplace

if [ ! -d "${workplace_dir_path}" ];then
  mkdir "${workplace_dir_path}"
fi

if [ -d "${workplace_dir_path}/results" ];then
  rm -rf "${workplace_dir_path}/results"
fi

if [ ! -d "${workplace_dir_path}/assets" ];then
  ln -s ${asset_dir_path} ${workplace_dir_path}/assets
fi

mkdir ${workplace_dir_path}/results

./lad-voxel-structure --type 8 --shell_radius 0.06 --lattice_radius 0.06 --cell_x 4 --cell_y 4 --cell_z 4 --num_samples 130
./lad-voronoi-structure --num_seeds 80 --lattice_radius 0.06 --num_samples 130
./lad-hexahedron-structure --alpha 4 --beta 1 --iterations 20 --lattice_radius 0.06  --cell_x 4 --cell_y 4 --cell_z 4 --num_samples 130
