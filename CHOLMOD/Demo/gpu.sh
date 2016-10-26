#!/bin/bash
export CHOLMOD_USE_GPU=1
#./cholmod_l_demo < ~/Temp/Matrices/Muu.mtx
#./cholmod_l_demo < ~/Temp/Matrices/nd3k.mtx
#./cholmod_l_demo < ~/Temp/Matrices/nd6k.mtx
./cholmod_l_demo < ~/Temp/Matrices/Fault_639.mtx
