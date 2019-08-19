#!/bin/bash
#
# gcc
# 
# On tev.fnal.gov it'll set gcc to /usr/local/gcc-7.1.0/bin/gcc
# (as opposed to default /usr/bin/gcc)
#
module load gcc/7.1.0 
#
# Alternatively, I can set up gcc 7.3 which was downloaded
# as "relacatable ups" from scisoft.fnal.gov
# 
source /g4/yarba_j/scisoft-installs/externals/setup
setup gcc v7_3_0 
#
# add cuda to the PATH (nvcc is part of it)
# (as of tev.fnal.gov; cuda 10.1 is also available)
#
export PATH=/usr/local/cuda-10.0/bin:${PATH}
#
# check what we're using
#
gcc_version=`gcc --version |head -1 | awk '{print $3}'`
nvcc_version=`nvcc --version |tail -1 |awk '{print $6}'`
echo "... Using gcc version ${gcc_version} and nvcc version ${nvcc_version} ..."
