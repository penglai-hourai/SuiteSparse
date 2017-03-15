/* ========================================================================== */
/* === Include/cholmod_gpu_kernels.h ======================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Include/cholmod_gpu_kernels.h.
 * Copyright (C) 2014, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * CUDA kernel support routines for CHOLMOD
 * -------------------------------------------------------------------------- */

#ifndef CHOLMODGPUKERNELS_H
#define CHOLMODGPUKERNELS_H

#include "SuiteSparse_config.h"

/* make it easy for C++ programs to include CHOLMOD */
#ifdef __cplusplus
extern "C" {
#endif

int createMapOnDevice ( Int *d_Map, Int *d_Ls, Int psi, Int nsrow ); 

int createRelativeMapOnDevice ( Int *d_Map, Int *d_Ls, Int *d_RelativeMap,
                           Int pdi1, Int ndrow, cudaStream_t *astream ); 

int addUpdateOnDevice ( double *d_A, double *devPtrC, Int *d_RelativeMap,
    Int ndrow1, Int ndrow2, Int nsrow, cudaStream_t *astream );

int addComplexUpdateOnDevice ( double *d_A, double *devPtrC, Int *d_RelativeMap,
    Int ndrow1, Int ndrow2, Int nsrow, cudaStream_t *astream );

int sumAOnDevice ( double *a1, double *a2, const double alpha, int nsrow,
    int nscol );

int sumComplexAOnDevice ( double *a1, double *a2, const double alpha,
    int nsrow, int nscol );

#ifdef __cplusplus
}
#endif

#endif
