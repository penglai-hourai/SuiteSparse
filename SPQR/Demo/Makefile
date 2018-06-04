#===============================================================================
# SuiteSparseQR/Demo/Makefile
#===============================================================================

default: all

ccode: all

include ../../SuiteSparse_config/SuiteSparse_config.mk

#-------------------------------------------------------------------------------
# the optional SPQRGPU module requires CUDA, SuiteSparse_GPURuntime, GPUQREngine
ifneq ($(GPU_CONFIG),)
    LIB_WITH_SPQRGPU = $(CUDART_LIB) $(CUBLAS_LIB) \
        -lSuiteSparse_GPURuntime -lGPUQREngine
    I_WITH_SPQRGPU = -I../../SuiteSparse_GPURuntime/Include \
        -I../../GPUQREngine/Include $(CUDA_INC)
else
    LIB_WITH_SPQRGPU = 
    I_WITH_SPQRGPU = 
endif

#-------------------------------------------------------------------------------

CLIB = $(LDFLAGS) -L../../lib -lspqr -lsuitesparseconfig -lcholmod -lamd \
        -lcolamd $(LIB_WITH_PARTITION) $(LIB_WITH_SPQRGPU) $(LDLIBS)

# use the BLAS and LAPACK defined by SuiteSparse_config.mk; do not use valgrind 
FLIB = $(LAPACK) $(BLAS)
V =

# To use Valgrind and the plain BLAS and plain LAPACK (non-optimized):
# FLIB = -lgfortran -llapack_plain -lblas_plain -lg2c
# V = valgrind --quiet

all: library qrdemo qrsimple qrsimplec gpu

cdemo: qrdemoc

ifneq ($(GPU_CONFIG),)
GPU_DEMOS = gpu2 gpu3 gpu4 gpu1
else
GPU_DEMOS =
endif

gpu: $(GPU_DEMOS)

gpu1: qrdemo_gpu

gpu2: qrdemo_gpu2

gpu3: qrdemo_gpu3

gpu4: qrdemo_gpu

library: metis
	( cd ../Lib ; $(MAKE) )
	( cd ../../AMD ; $(MAKE) library )
	( cd ../../SuiteSparse_config ; $(MAKE) library )
	- ( cd ../../CHOLMOD && $(MAKE) library )
	- ( cd ../../COLAMD && $(MAKE) library )
	- ( cd ../../CCOLAMD && $(MAKE) library )
	- ( cd ../../CAMD && $(MAKE) library )
ifneq ($(GPU_CONFIG),)
	- ( cd ../../SuiteSparse_GPURuntime && $(MAKE) library )
	- ( cd ../../GPUQREngine && $(MAKE) library )
endif

metis: ../../include/metis.h

../../include/metis.h:
	- ( cd ../.. && $(MAKE) metis )

purge: distclean

distclean: clean
	- $(RM) qrdemo qrdemo_gpu qrdemoc qrsimple qrsimplec X.mtx
	- $(RM) R.mtx C.mtx E.txt gpu_results.txt qrdemo_gpu2 qrdemo_gpu3
	- $(RM) *.dot pfile tfile
	- $(RM) -r $(PURGE)

clean:
	- $(RM) -r $(CLEAN)

INC = ../Include/spqr.hpp ../Include/SuiteSparseQR_C.h \
	../Include/SuiteSparseQR_definitions.h \
	../Include/SuiteSparseQR.hpp Makefile

I = -I../../include $(I_WITH_SPQRGPU)

C = $(CXX) $(CF) $(SPQR_CONFIG) $(CONFIG_PARTITION) $(CONFIG_GPU) $(I) \
	$(CHOLMOD_CONFIG)

LIBS = $(CLIB) $(FLIB) $(TBB) $(GPULIB)

# With the CUDA BLAS
ifneq ($(GPU_CONFIG),)
LIBS += $(CUDART_LIB) $(CUBLAS_LIB)
endif

qrsimple: qrsimple.cpp $(INC)
	$(C) qrsimple.cpp -o qrsimple $(LIBS)

qrdemo: qrdemo.cpp $(INC)
	$(C) qrdemo.cpp -o qrdemo $(LIBS)

qrdemo_gpu: qrdemo_gpu.cpp $(INC)
ifneq ($(GPU_CONFIG),)
	$(C) qrdemo_gpu.cpp -o qrdemo_gpu $(LIBS)
else
	echo
endif

qrdemo_gpu2: qrdemo_gpu2.cpp $(INC)
ifneq ($(GPU_CONFIG),)
	$(C) qrdemo_gpu2.cpp -o qrdemo_gpu2 $(LIBS)
else
	echo
endif

qrdemo_gpu3: qrdemo_gpu3.cpp $(INC)
ifneq ($(GPU_CONFIG),)
	$(C) qrdemo_gpu3.cpp -o qrdemo_gpu3 $(LIBS)
else
	echo
endif

# compile the C code with gcc, but link with g++ to use SuiteSparseQR:

qrdemoc.o: qrdemoc.c $(INC)
	$(CC) $(CF) $(SPQR_CONFIG) $(I) -c $<

qrdemoc: qrdemoc.o $(INC)
	$(C) -o qrdemoc qrdemoc.o $(LIBS)

qrsimplec.o: qrsimplec.c $(INC)
	$(CC) $(CF) $(SPQR_CONFIG) $(CHOLMOD_CONFIG) $(I) -c $<

qrsimplec: qrsimplec.o $(INC)
	$(C) -o qrsimplec qrsimplec.o $(LIBS)

