#!/usr/bin/env bash

project_name=cmd-topology-optimation

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

./top-arbitrary-model
