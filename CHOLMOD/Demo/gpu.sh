#!/bin/bash

export CHOLMOD_USE_GPU=1

./cholmod_l_batched_demo \
    /scratch/fantamon/Matrices/Emilia_923.mtx   \
    /scratch/fantamon/Matrices/Fault_639.mtx    \
    /scratch/fantamon/Matrices/Flan_1565.mtx    \
    /scratch/fantamon/Matrices/G3_circuit.mtx   \
    /scratch/fantamon/Matrices/Geo_1438.mtx     \
    /scratch/fantamon/Matrices/Hook_1498.mtx    \
    /scratch/fantamon/Matrices/Serena.mtx       \
    /scratch/fantamon/Matrices/StocF-1465.mtx   \
#    /scratch/fantamon/Matrices/apache2.mtx      \
#    /scratch/fantamon/Matrices/audikw_1.mtx     \
#    /scratch/fantamon/Matrices/bone010.mtx      \
#    /scratch/fantamon/Matrices/boneS10.mtx      \
#    /scratch/fantamon/Matrices/ecology2.mtx     \
#    /scratch/fantamon/Matrices/ldoor.mtx        \
#    /scratch/fantamon/Matrices/thermal2.mtx     \
#    /scratch/fantamon/Matrices/tmt_sym.mtx      \
