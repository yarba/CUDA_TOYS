#!/bin/bash
#
# gcc
# 
source /g4/yarba_j/scisoft-installs/externals/setup
setup gcc v7_3_0
setup cmake v3_12_1
#
# setup PATH
#
export PATH=/usr/local/cuda-10.0/bin:${PATH}
#
# message
#
gcc_version=`gcc --version |head -1 | awk '{print $3}'`
nvcc_version=`nvcc --version |tail -1 |awk '{print $6}'`
echo "... Using gcc version ${gcc_version} and nvcc version ${nvcc_version} ..."
