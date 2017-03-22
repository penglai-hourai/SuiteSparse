#!/bin/bash

export CHOLMOD_USE_GPU=1

./cholmod_l_batched_demo \
    ~/Temp/Matrices/Emilia_923.mtx   \
    ~/Temp/Matrices/Fault_639.mtx    \
    ~/Temp/Matrices/Flan_1565.mtx    \
    ~/Temp/Matrices/G3_circuit.mtx   \
    ~/Temp/Matrices/Geo_1438.mtx     \
    ~/Temp/Matrices/Hook_1498.mtx    \
    ~/Temp/Matrices/Serena.mtx       \
    ~/Temp/Matrices/StocF-1465.mtx   \
    ~/Temp/Matrices/apache2.mtx      \
    ~/Temp/Matrices/audikw_1.mtx     \
    ~/Temp/Matrices/bone010.mtx      \
    ~/Temp/Matrices/boneS10.mtx      \
    ~/Temp/Matrices/ecology2.mtx     \
    ~/Temp/Matrices/ldoor.mtx        \
    ~/Temp/Matrices/thermal2.mtx     \
    ~/Temp/Matrices/tmt_sym.mtx      \
