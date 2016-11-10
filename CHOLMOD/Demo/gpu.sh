#!/bin/bash

export CHOLMOD_USE_GPU=1                                # Type      Rows        Cols        Nonzeros    flop_count

#./cholmod_l_demo < ~/Temp/Matrices/Emilia_923.mtx      # Real      923136      923136      40373538     1.3e+14
#./cholmod_l_demo < ~/Temp/Matrices/Fault_639.mtx       # Real      638802      638802      27245944     6.4e+13
#./cholmod_l_demo < ~/Temp/Matrices/Flan_1565.mtx       # Real      1564794     1564794     114165372    2.2e_13
./cholmod_l_demo < ~/Temp/Matrices/G3_circuit.mtx      # Real      1585478     1585478     7660826      3.0e+11
#./cholmod_l_demo < ~/Temp/Matrices/Geo_1438.mtx        # Real      1437960     1437960     60236322     1.2e+14
#./cholmod_l_demo < ~/Temp/Matrices/Hook_1498.mtx       # Real      1498023     1498023     59374451     2.1e+13
#./cholmod_l_demo < ~/Temp/Matrices/Serena.mtx          # Real      1391349     1391349     64131971     2.2e+14
#./cholmod_l_demo < ~/Temp/Matrices/StocF-1465.mtx      # Real      1465137     1465137     21005389     4.3e+13
#./cholmod_l_demo < ~/Temp/Matrices/apache2.mtx         # Real      715176      715176      4817870      2.8e+11
#./cholmod_l_demo < ~/Temp/Matrices/audikw_1.mtx        # Real      943695      943695      77651847     2.1e+13
#./cholmod_l_demo < ~/Temp/Matrices/bone010.mtx         # Real      986703      986703      47851783     2.5e+13
#./cholmod_l_demo < ~/Temp/Matrices/boneS10.mtx         # Real      914898      914898      40878708     7.7e+11
#./cholmod_l_demo < ~/Temp/Matrices/ecology2.mtx        # Real      999999      999999      4995991      2.0e+10
#./cholmod_l_demo < ~/Temp/Matrices/ldoor.mtx           # Real      952203      952203      42493817     1.1e+11
#./cholmod_l_demo < ~/Temp/Matrices/thermal2.mtx        # Real      1228045     1228045     8580313      2.8e+10
#./cholmod_l_demo < ~/Temp/Matrices/tmt_sym.mtx         # Real      726713      726713      5080961      2.4e+10
