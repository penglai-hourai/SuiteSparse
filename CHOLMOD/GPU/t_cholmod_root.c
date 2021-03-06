/* ========================================================================== */
/* === GPU/t_cholmod_root =================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/GPU Module.  Copyright (C) 2005-2012, Timothy A. Davis
 * The CHOLMOD/GPU Module is licensed under Version 2.0 of the GNU
 * General Public License.  See gpl.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/*
 * File:
 *   t_cholmod_root
 *
 * Description:
 *   Contains functions for root algorithm.
 *
 */


/* includes */
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


/* macros */
#undef L_ENTRY
#ifdef REAL
#define L_ENTRY 1
#else
#define L_ENTRY 2
#endif










/*
 * Function:
 *   gpu_init_root
 *
 * Description:
 *   Performs required initialization for GPU computing.
 *   Returns 0 if there is an error, disabling GPU computing (useGPU = 0)
 *
 */
int TEMPLATE2 (CHOLMOD (gpu_init_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   cholmod_factor *L,
   Int *Lpi,
   Int nsuper,
   Int n,
   int gpuid
   )
{
  /* local variables */
  Int i, k, nls;
  cublasStatus_t cublasError;
  cudaError_t cudaErr;

  void *base_root;

#ifdef USE_NVTX
  nvtxRangeId_t range1 = nvtxRangeStartA("gpu_init_root");
#endif


  /* set cuda device */
  //cudaSetDevice(gpuid / Common->numGPU_parallel);


  /* compute nls */
  nls =  Lpi[nsuper]-Lpi[0];


  /* make buffer size is large enough */
  if ( (nls + n * (1 + CHOLMOD_DEVICE_C_BUFFERS)) * sizeof(Int) > Common->devBuffSize ) {
    ERROR (CHOLMOD_GPU_PROBLEM,"\n\n"
           "GPU Memory allocation error.  Ls, Map and RelativeMap exceed\n"
           "devBuffSize.  It is not clear if this is due to insufficient\n"
           "device or host memory or both.  You can try:\n"
           "     1) increasing the amount of GPU memory requested\n"
           "     2) reducing CHOLMOD_NUM_HOST_BUFFERS\n"
           "     3) using a GPU & host with more memory\n"
           "This issue is a known limitation and should be fixed in a \n"
           "future release of CHOLMOD.\n") ;
    return (0) ;
  }




  /* set gpu memory pointers */

  /* type double */
  base_root = Common->dev_mempool[gpuid / Common->numGPU_parallel] + (gpuid % Common->numGPU_parallel * CHOLMOD_DEVICE_SUPERNODE_BUFFERS) * Common->devBuffSize;
  for (k = 0; k < CHOLMOD_DEVICE_LX_BUFFERS; k++)
      gpu_p->d_Lx_root[gpuid][k] = base_root + k * Common->devBuffSize;
  for (k = 0; k < CHOLMOD_DEVICE_C_BUFFERS; k++)
      gpu_p->d_C_root[gpuid][k]  = base_root + (CHOLMOD_DEVICE_LX_BUFFERS + k) * Common->devBuffSize;
  gpu_p->d_A_root[gpuid][0]  = base_root + (CHOLMOD_DEVICE_LX_BUFFERS + CHOLMOD_DEVICE_C_BUFFERS + 0) * Common->devBuffSize;
  gpu_p->d_A_root[gpuid][1]  = base_root + (CHOLMOD_DEVICE_LX_BUFFERS + CHOLMOD_DEVICE_C_BUFFERS + 1) * Common->devBuffSize;

  /* type Int */
  gpu_p->d_Ls_root[gpuid]    = base_root + (CHOLMOD_DEVICE_LX_BUFFERS + CHOLMOD_DEVICE_C_BUFFERS + 2) * Common->devBuffSize;
  gpu_p->d_Map_root[gpuid] = (void*) gpu_p->d_Ls_root[gpuid] + sizeof(Int) * nls;
  for (k = 0; k < CHOLMOD_DEVICE_C_BUFFERS; k++)
      gpu_p->d_RelativeMap_root[gpuid][k] = (void*) gpu_p->d_Ls_root[gpuid] + sizeof(Int) * nls + sizeof(Int) * n * (1 + k);




  /* copy Ls and Lpi to device */
  //if (gpuid % Common->numGPU_parallel == 0)
  {
  cudaErr = cudaMemcpyAsync ( gpu_p->d_Ls_root[gpuid], L->s, nls*sizeof(Int), cudaMemcpyHostToDevice, Common->gpuStream[gpuid][0] );
  CHOLMOD_HANDLE_CUDA_ERROR(cudaErr,"cudaMemcpy(d_Ls_root)");
  }




  /* set pinned memory pointers */
  /*
  gpu_p->h_Lx_root[gpuid][0]
      = ((void*) Common->host_pinned_mempool[gpuid / Common->numGPU_parallel])
      + (gpuid % Common->numGPU_parallel * CHOLMOD_HOST_SUPERNODE_BUFFERS) * Common->devBuffSize;
  */

  for (k = 0; k <= CHOLMOD_HOST_SUPERNODE_BUFFERS; k++) {
    gpu_p->h_Lx_root[gpuid][k]
        = ((void*) Common->host_pinned_mempool[gpuid / Common->numGPU_parallel])
        + (gpuid % Common->numGPU_parallel * (CHOLMOD_HOST_SUPERNODE_BUFFERS+1)) * Common->devBuffSize + k * Common->devBuffSize;
  }


#ifdef USE_NVTX
  nvtxRangeEnd(range1);
#endif


  return (1);  /* initialization successfull, useGPU = 1 */

}










/*
 * Function:
 *   gpu_reorder_descendants_root
 *
 * Description:
 *   Reorders descendants in a supernode by size (ndrow2*ndcol)
 *
 */
void TEMPLATE2 (CHOLMOD (gpu_reorder_descendants_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   Int k1,
   Int k2,
   Int *Ls,
   Int *Lpi,
   Int *Lpos,
   Int *Super,
   Int *Head,
   Int *tail,
   Int *Next,
   Int *Previous,
   Int *ndescendants,
   Int *mapCreatedOnGpu,
   Int locals,
   int gpuid
   )
{
  /* local variables */
  Int d, k, p, kd1, kd2, ndcol, ndrow1, ndrow2, pdi, pdend, pdi1, pdi2, nextd, dnext, n_descendant = 0;
  Int previousd, nreverse = 1, numThreads;
  double score;

  /* store GPU-eligible descendants in h_Lx[0] */
  struct cholmod_descendant_score_t* scores = (struct cholmod_descendant_score_t*) gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS];

  //cudaSetDevice(gpuid / Common->numGPU_parallel);


  /* initialize variables */
  d = Head[locals];
  *mapCreatedOnGpu = 0;
  numThreads	= Common->ompNumThreads;



  /* loop until reach last descendant in supernode */
  while ( d != EMPTY ) {

      /* get dimensions for the current descendant */
      kd1 = Super [d] ;       /* d contains cols kd1 to kd2-1 of L */
      kd2 = Super [d+1] ;
      ndcol = kd2 - kd1 ;     /* # of columns in all of d */
      pdi = Lpi [d] ;         /* pointer to first row of d in Ls */
      pdend = Lpi [d+1] ;     /* pointer just past last row of d in Ls */
      p = Lpos [d] ;          /* offset of 1st row of d affecting s */
      pdi1 = pdi + p ;        /* ptr to 1st row of d affecting s in Ls */
      for (pdi2 = pdi1 ; pdi2 < pdend && Ls [pdi2] < k2 ; (pdi2)++) ;
      ndrow1 = pdi2 - pdi1 ;
      ndrow2 = pdend - pdi1 ;

      nextd = Next[d];

      /* compute the descendant's rough flops 'score' */
#if 1
      if ( (L_ENTRY * ndcol >= CHOLMOD_ND_COL_LIMIT) && (L_ENTRY * ndrow2 >= CHOLMOD_ND_ROW_LIMIT) )
      {
          score = ndcol * ndrow2;
      }
      else
      {
          score = - ndcol * ndrow2;
      }
#else
          score = ndcol * ndrow1 * ndrow2;
#endif

      /* store descendant in list */
      scores[n_descendant].score = score;
      scores[n_descendant].d = d;
      n_descendant++;

      d = nextd;

  }



  /* sort the GPU-eligible supernodes in descending size (flops) */
  qsort ( scores, n_descendant, sizeof(struct cholmod_descendant_score_t), (__compar_fn_t) CHOLMOD(score_comp) );



  /* place sorted data back in descendant supernode linked list */
  if ( n_descendant > 0 ) {

    Head[locals] = scores[0].d;
    if ( n_descendant > 1 ) {

      #pragma omp parallel for num_threads(numThreads) if (n_descendant > 64)
      for ( k=1; k<n_descendant; k++ ) {
        Next[scores[k-1].d] = scores[k].d;
      }

    }

    Next[scores[n_descendant-1].d] = EMPTY;
  }



  /* reverse the first CHOLMOD_HOST_SUPERNODE_BUFFERS descendants to better hide PCIe communications */
  if ( Head[locals] != EMPTY && Next[Head[locals]] != EMPTY ) {

    previousd = Head[locals];
    d = Next[Head[locals]];

    /* loop through the first CHOLMOD_HOST_SUPERNODE_BUFFERS descendants */
    while ( d!=EMPTY && nreverse < CHOLMOD_HOST_SUPERNODE_BUFFERS ) {

      /* get descendant dimensions */
      kd1 = Super [d] ;       /* d contains cols kd1 to kd2-1 of L */
      kd2 = Super [d+1] ;
      ndcol = kd2 - kd1 ;     /* # of columns in all of d */
      pdi = Lpi [d] ;         /* pointer to first row of d in Ls */
      pdend = Lpi [d+1] ;     /* pointer just past last row of d in Ls */
      p = Lpos [d] ;          /* offset of 1st row of d affecting s */
      pdi1 = pdi + p ;        /* ptr to 1st row of d affecting s in Ls */
      ndrow2 = pdend - pdi1;

      nextd = Next[d];

      nreverse++;

      /* place descendant at the front of the list */
#if 1
      if ( (L_ENTRY * ndcol >= CHOLMOD_ND_COL_LIMIT) && (L_ENTRY * ndrow2 >= CHOLMOD_ND_ROW_LIMIT) )
      {
          Next[previousd] = Next[d];
          Next[d] = Head[locals];
          Head[locals] = d;
      }
      else 
      {
          previousd = d;
      }
#else
      Next[previousd] = Next[d];
      Next[d] = Head[locals];
      Head[locals] = d;
#endif

      d = nextd;
    } /* end while loop */

  }



  /* create a 'previous' list so we can traverse backwards */
  n_descendant = 0;

  if ( Head[locals] != EMPTY ) {

    Previous[Head[locals]] = EMPTY;

    /* loop over descendants */
    for (d = Head [locals] ; d != EMPTY ; d = dnext) {

      n_descendant++;
      dnext = Next[d];

      if ( dnext != EMPTY ) {
        Previous[dnext] = d;
      }
      else {
        *tail = d;
      }

    } /* end loop over descendants */

  }


  /* store descendant dimension */
  *ndescendants = n_descendant;

}










/*
 *  Function:
 *    gpu_initialize_supernode_root
 *
 *  Description:
 *    Initializes a supernode.
 *    Performs two tasks:
 *      1. clears A buffer for assembly
 *      2. creates map on device
 *
 */
void TEMPLATE2 (CHOLMOD (gpu_initialize_supernode_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   Int nscol,
   Int nsrow,
   Int psi,
   Int psx,
   int gpuid
   )
{
  /* local variables */
  Int iidx, i, j;
  int numThreads;
  cudaError_t cudaErr;

  //cudaSetDevice(gpuid / Common->numGPU_parallel);
  numThreads = Common->ompNumThreads;

  /* initialize the device supernode assemby memory to zero */
  cudaErr = cudaMemsetAsync ( gpu_p->d_A_root[gpuid][0], 0, nscol*nsrow*L_ENTRY*sizeof(double), Common->gpuStream[gpuid][0] );
  CHOLMOD_HANDLE_CUDA_ERROR(cudaErr,"cudaMemset(d_A_root)");


  /* create the map for supernode on the device */
  createMapOnDevice ( (Int *)(gpu_p->d_Map_root[gpuid]), (Int *)(gpu_p->d_Ls_root[gpuid]), psi, nsrow, Common->gpuStream[gpuid][0] );
  cudaErr = cudaGetLastError();
  CHOLMOD_HANDLE_CUDA_ERROR(cudaErr,"createMapOnDevice error");
}










/*
 *  Function:
 *    gpu_updateC_root
 *
 *  Description:
 *    Computes the schur compliment (DSYRK & DGEMM) of a batch of supernodes and maps
 *    it back (addUpdate).
 *    Performs three tasks:
 *      1. DSYRK
 *      2. DGEMM
 *      3. addUpdate (map schur comp. to supernode)
 */
int TEMPLATE2 (CHOLMOD (gpu_updateC_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   double *Lx,
   Int ndrow1,
   Int ndrow2,
   Int ndrow,
   Int ndcol,
   Int nsrow,
   Int pdx1,
   Int pdi1,
   int iHostBuff,
   int iDevBuff,
   int iDevCBuff,
   int gpuid
   )
{
  /* local variables */
  Int icol, irow, numThreads;
  Int ndrow3;
  double alpha, beta;
  double *devPtrLx, *devPtrC;
  cublasStatus_t cublasStatus;
  cudaError_t cudaErr;

  //cudaSetDevice(gpuid / Common->numGPU_parallel);

  numThreads = Common->ompNumThreads;



  /* initialize variables */
  ndrow3 = ndrow2 - ndrow1 ;
  alpha  = 1.0 ;
  beta   = 0.0 ;

  /* initialize poitners */
  devPtrLx = (double *)(gpu_p->d_Lx_root[gpuid][iDevBuff]);
  devPtrC = (double *)(gpu_p->d_C_root[gpuid][iDevCBuff]);


  //cudaStreamWaitEvent ( Common->gpuStream[gpuid][iDevBuff], Common->updateCBuffersFree[gpuid][iHostBuff], 0 ) ;


  /*
   * Copy Lx to the device:
   * First copy to pinned buffer, then to the device for
   * better H2D bandwidth.
   */
  /* copy host data to pinned buffer */
#pragma omp parallel for num_threads(numThreads) private (icol, irow) if (ndcol > 32)
  for ( icol=0; icol<ndcol; icol++ ) {
    for ( irow=0; irow<ndrow2*L_ENTRY; irow++ ) {
      gpu_p->h_Lx_root[gpuid][iHostBuff][icol*ndrow2*L_ENTRY+irow] =
        Lx[pdx1*L_ENTRY+icol*ndrow*L_ENTRY + irow];
    }
  }



  /* make the current stream wait for kernels in previous streams */
#ifndef QUERY_LX_EVENTS
  cudaStreamWaitEvent ( Common->gpuStream[gpuid][iDevBuff], Common->updateCDevBuffersFree[gpuid][iDevBuff], 0 ) ;
#endif



  /* copy pinned buffer to device */
  cudaErr = cudaMemcpyAsync (
          devPtrLx,
          gpu_p->h_Lx_root[gpuid][iHostBuff],
          ndrow2*ndcol*L_ENTRY*sizeof(devPtrLx[0]),
          cudaMemcpyHostToDevice,
          Common->gpuStream[gpuid][iDevBuff]);

  if ( cudaErr ) {
    CHOLMOD_HANDLE_CUDA_ERROR(cudaErr,"cudaMemcpyAsync H-D");
    return (0);
  }

  cudaEventRecord ( Common->updateCBuffersFree[gpuid][iHostBuff], Common->gpuStream[gpuid][iDevBuff] );

  /* make the current stream wait for kernels in previous streams */
  cudaStreamWaitEvent ( Common->gpuStream[gpuid][iDevBuff], Common->updateCKernelsComplete[gpuid], 0 ) ;
  //cudaStreamWaitEvent ( Common->gpuStream[gpuid][iDevBuff], Common->updateCDevCBuffersFree[gpuid][iDevCBuff], 0 ) ;


  /* create relative map for the descendant */
  createRelativeMapOnDevice (
          (Int *)(gpu_p->d_Map_root[gpuid]),
          (Int *)(gpu_p->d_Ls_root[gpuid]),
          (Int *)(gpu_p->d_RelativeMap_root[gpuid][iDevCBuff]),
          pdi1,
          ndrow2,
          &(Common->gpuStream[gpuid][iDevBuff]));

  cudaErr = cudaGetLastError();
  if (cudaErr) {
    CHOLMOD_HANDLE_CUDA_ERROR(cudaErr,"createRelativeMapOnDevice");
  }



  /* set cuBlas stream  */
  cublasStatus = cublasSetStream (Common->cublasHandle[gpuid / Common->numGPU_parallel], Common->gpuStream[gpuid][iDevBuff]) ;
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS stream") ;
    return(0);
  }


  /*
   * Perform DSYRK on GPU for current descendant
   */

  alpha  = 1.0 ;
  beta   = 0.0 ;

#ifdef REAL
  cublasStatus = cublasDsyrk (
          Common->cublasHandle[gpuid / Common->numGPU_parallel],
          CUBLAS_FILL_MODE_LOWER,
          CUBLAS_OP_N,
          (int) ndrow1,
          (int) ndcol,    				/* N, K: L1 is ndrow1-by-ndcol */
          &alpha,         				/* ALPHA:  1 */
          devPtrLx,
          ndrow2,         				/* A, LDA: L1, ndrow2 */
          &beta,          				/* BETA:   0 */
          devPtrC,
          ndrow2) ;       				/* C, LDC: C1 */
#else
  cublasStatus = cublasZherk (
          Common->cublasHandle[gpuid / Common->numGPU_parallel],
          CUBLAS_FILL_MODE_LOWER,
          CUBLAS_OP_N,
          (int) ndrow1,
          (int) ndcol,    				/* N, K: L1 is ndrow1-by-ndcol*/
          &alpha,         				/* ALPHA:  1 */
          (const cuDoubleComplex *) devPtrLx,
          ndrow2,         				/* A, LDA: L1, ndrow2 */
          &beta,          				/* BETA:   0 */
          (cuDoubleComplex *) devPtrC,
          ndrow2) ;       				/* C, LDC: C1 */
#endif


  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    ERROR (CHOLMOD_GPU_PROBLEM, "GPU cublasDsyrk error") ;
    return(0);
  }




  /*
   * Perform DSYRK on GPU for current descendant
   */
  if (ndrow3 > 0)
  {

#ifndef REAL
    cuDoubleComplex calpha  = {1.0,0.0} ;
    cuDoubleComplex cbeta   = {0.0,0.0} ;
#endif

#ifdef REAL
    alpha  = 1.0 ;
    beta   = 0.0 ;

    cublasStatus = cublasDgemm (
            Common->cublasHandle[gpuid / Common->numGPU_parallel],
            CUBLAS_OP_N, CUBLAS_OP_T,
            ndrow3, ndrow1, ndcol,          	/* M, N, K */
            &alpha,                         	/* ALPHA:  1 */
            devPtrLx + L_ENTRY*(ndrow1),    	/* A, LDA: L2*/
            ndrow2,                         	/* ndrow */
            devPtrLx,                       	/* B, LDB: L1 */
            ndrow2,                         	/* ndrow */
            &beta,                          	/* BETA:   0 */
            devPtrC + L_ENTRY*ndrow1,       	/* C, LDC: C2 */
            ndrow2);
#else
    cublasStatus = cublasZgemm (
            Common->cublasHandle[gpuid / Common->numGPU_parallel],
            CUBLAS_OP_N, CUBLAS_OP_C,
            ndrow3, ndrow1, ndcol,          	/* M, N, K */
            &calpha,                        	/* ALPHA:  1 */
            (const cuDoubleComplex*) devPtrLx + ndrow1,
            ndrow2,                         	/* ndrow */
            (const cuDoubleComplex *) devPtrLx,
            ndrow2,                         	/* ndrow */
            &cbeta,                         	/* BETA:   0 */
            (cuDoubleComplex *)devPtrC + ndrow1,
            ndrow2);
#endif



    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
      ERROR (CHOLMOD_GPU_PROBLEM, "GPU cublasDgemm error") ;
      return(0);
    }

  }

  cudaEventRecord (Common->updateCDevBuffersFree[gpuid][iDevBuff], Common->gpuStream[gpuid][iDevBuff]);



  /*
   * Assemble the update C on the devicet
   */
#if defined(CHOLMOD_DEVICE_C_BUFFERS) && (CHOLMOD_DEVICE_C_BUFFERS > 1)
    cudaErr = cudaStreamWaitEvent (Common->gpuStream[gpuid][iDevBuff], Common->cublasEventPotrf[gpuid][0], 0) ;
#endif
#ifdef REAL
  addUpdateOnDevice (
          gpu_p->d_A_root[gpuid][0],
          devPtrC,
          gpu_p->d_RelativeMap_root[gpuid][iDevCBuff],
          ndrow1,
          ndrow2,
          nsrow,
          &(Common->gpuStream[gpuid][iDevBuff]));
#else
  addComplexUpdateOnDevice (
          gpu_p->d_A_root[gpuid][0],
          devPtrC,
          gpu_p->d_RelativeMap_root[gpuid][iDevCBuff],
          ndrow1,
          ndrow2,
          nsrow,
          &(Common->gpuStream[gpuid][iDevBuff]));
#endif
#if defined(CHOLMOD_DEVICE_C_BUFFERS) && (CHOLMOD_DEVICE_C_BUFFERS > 1)
    cudaErr = cudaEventRecord (Common->cublasEventPotrf[gpuid][0], Common->gpuStream[gpuid][iDevBuff]) ;
#endif



  cudaErr = cudaGetLastError();
  if (cudaErr) {
    ERROR (CHOLMOD_GPU_PROBLEM,"\naddUpdateOnDevice error!\n");
    return (0) ;
  }



  /* record event indicating that kernels for descendant are complete */
  //cudaEventRecord (Common->updateCDevCBuffersFree[gpuid][iDevCBuff], Common->gpuStream[gpuid][iDevBuff]);
  cudaEventRecord (Common->updateCKernelsComplete[gpuid], Common->gpuStream[gpuid][iDevBuff]);



  return (1) ;
}










/*
 *  Function:
 *    gpu_final_assembly_root
 *
 *  Description:
 *    Sum all schur-comlement updates (computed on the GPU) to the supernode
 */
void TEMPLATE2 (CHOLMOD (gpu_final_assembly_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   double *Lx,
   Int psx,
   Int nscol,
   Int nsrow,
   int supernodeUsedGPU,
   int gpuid
   )
{
  /* local variables */
  Int iidx, i, j;
  cudaError_t cudaErr ;
  int numThreads;

  //cudaSetDevice(gpuid / Common->numGPU_parallel);


  numThreads = Common->ompNumThreads;



  /* only if descendant was assembled on GPU */
  if ( supernodeUsedGPU ) {

    /* only if descendant is large enough for GPU */
    if ( nscol * L_ENTRY >= CHOLMOD_POTRF_LIMIT ) {

      /* copy update assembled on CPU to a pinned buffer */
#pragma omp parallel for num_threads(numThreads) private(i, j, iidx) if (nscol>32)

      for ( j=0; j<nscol; j++ ) {
        for ( i=j; i<nsrow*L_ENTRY; i++ ) {
          iidx = j*nsrow*L_ENTRY+i;
          gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS][iidx] = Lx[psx*L_ENTRY+iidx];
        }
      }

      /* H2D transfer of update assembled on CPU */
      cudaMemcpyAsync (
              gpu_p->d_A_root[gpuid][1], gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS],
              nscol*nsrow*L_ENTRY*sizeof(double),
              cudaMemcpyHostToDevice,
              Common->gpuStream[gpuid][0] );

      cudaErr = cudaGetLastError();
      if (cudaErr) {
        ERROR (CHOLMOD_GPU_PROBLEM,"\nmemcopy H-D error!\n");
        return;
      }

    /* need both H2D and D2H copies to be complete */
    //cudaStreamSynchronize(Common->gpuStream[gpuid][0]);
    cudaStreamWaitEvent( Common->gpuStream[gpuid][0], Common->updateCKernelsComplete[gpuid], 0 );

      /*
       * sum updates from cpu and device on device
       */
#ifdef REAL
      sumAOnDevice ( gpu_p->d_A_root[gpuid][1], gpu_p->d_A_root[gpuid][0], -1.0, nsrow, nscol, Common->gpuStream[gpuid][0] );
#else
      sumComplexAOnDevice ( gpu_p->d_A_root[gpuid][1], gpu_p->d_A_root[gpuid][0], -1.0, nsrow, nscol, Common->gpuStream[gpuid][0] );
#endif


      cudaErr = cudaGetLastError();
      if (cudaErr) {
        ERROR (CHOLMOD_GPU_PROBLEM,"\nsumAonDevice error!\n");
        return;
      }

    } /* end if descendant large enough */
    /* if descendant too small assemble on CPU */
    else
    {

    /* copy assembled Schur-complement updates computed on GPU */
    cudaStreamWaitEvent( Common->gpuStream[gpuid][0], Common->updateCKernelsComplete[gpuid], 0 );
    cudaMemcpyAsync ( gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS],
		      gpu_p->d_A_root[gpuid][0],
                      nscol*nsrow*L_ENTRY*sizeof(double),
                      cudaMemcpyDeviceToHost,
                      Common->gpuStream[gpuid][0] );

    cudaErr = cudaGetLastError();
    if (cudaErr) {
      ERROR (CHOLMOD_GPU_PROBLEM,"\nmemcopy D-H error!\n");
      return ;
    }


    /* need both H2D and D2H copies to be complete */
    cudaStreamSynchronize(Common->gpuStream[gpuid][0]);

      /* assemble with CPU updates */
      #pragma omp parallel for num_threads(numThreads) private(i, j, iidx) if (nscol>32)
      for ( j=0; j<nscol; j++ ) {
        for ( i=j*L_ENTRY; i<nsrow*L_ENTRY; i++ ) {
          iidx = j*nsrow*L_ENTRY+i;
          Lx[psx*L_ENTRY+iidx] -= gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS][iidx];
        }
      }

    }


  } /* end if descendant assembled on GPU */


}










/*
 *  Function:
 *    gpu_lower_potrf_root
 *
 *  Description:
 *    computes cholesky factoriation for a supernode
 *    Performs one task:
 *      1. DPOTRF
 */
int TEMPLATE2 (CHOLMOD (gpu_lower_potrf_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   double *Lx,
   Int *info,
   Int nscol2,
   Int nsrow,
   Int psx,
   int gpuid
   )
{
  /* local variables */
  int ilda, ijb, iinfo = 0;
  Int j, n, jb, nb, nsrow2, gpu_lda, lda, gpu_ldb;
  double alpha, beta;
  double *devPtrA, *devPtrB, *A;
  cudaError_t cudaErr ;
  cublasStatus_t cublasStatus ;

  //cudaSetDevice(gpuid / Common->numGPU_parallel);



  /* early exit if descendant is too small for cuBlas */
  if (nscol2 * L_ENTRY < CHOLMOD_POTRF_LIMIT)
  {
    return (0) ;
  }


  /* set dimnsions & strides */
  nsrow2 = nsrow - nscol2 ;
  n  = nscol2 ;
  gpu_lda = ((nscol2+31)/32)*32 ;
  lda = nsrow ;
  gpu_ldb = 0 ;
  if (nsrow2 > 0) {
    gpu_ldb = ((nsrow2+31)/32)*32 ;
  }


  /* heuristic to get the block size depending of the problem size */
  nb = 128 ;
  if (nscol2 > 4096) nb = 256 ;
  if (nscol2 > 8192) nb = 384 ;


  /* set device pointers */
  A = gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS];
  devPtrA = gpu_p->d_Lx_root[gpuid][0];
  devPtrB = gpu_p->d_Lx_root[gpuid][1];


  /* copy B in advance, for gpu_triangular_solve */
  if (nsrow2 > 0) {
      cudaErr = cudaMemcpy2DAsync (
              devPtrB,
              gpu_ldb * L_ENTRY * sizeof (devPtrB [0]),
              gpu_p->d_A_root[gpuid][1] + L_ENTRY*nscol2,
              nsrow * L_ENTRY * sizeof (Lx [0]),
              nsrow2 * L_ENTRY * sizeof (devPtrB [0]),
              nscol2,
              cudaMemcpyDeviceToDevice,
              Common->gpuStream[gpuid][0]) ;
    if (cudaErr) {
      ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
    }
  }


  /* copy A from device to device */
  cudaErr = cudaMemcpy2DAsync (
          devPtrA,
          gpu_lda * L_ENTRY * sizeof (devPtrA[0]),
          gpu_p->d_A_root[gpuid][1],
          nsrow * L_ENTRY * sizeof (Lx[0]),
          nscol2 * L_ENTRY * sizeof (devPtrA[0]),
          nscol2,
          cudaMemcpyDeviceToDevice,
          Common->gpuStream[gpuid][0]);

  if ( cudaErr ){
    ERROR ( CHOLMOD_GPU_PROBLEM, "GPU memcopy device to device");
  }


    /* record end of cublasDsyrk */
    cudaErr = cudaEventRecord (Common->cublasEventPotrf[gpuid][0], Common->gpuStream[gpuid][0]) ;
    if (cudaErr) {
      ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
    }


    /* wait for cublasDsyrk to end */
    cudaErr = cudaStreamWaitEvent (Common->gpuStream[gpuid][1], Common->cublasEventPotrf[gpuid][0], 0) ;
    if (cudaErr) {
      ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
    }


  /*
   * block Cholesky factorization of S
   */
  /* loop over blocks */
  for (j = 0 ; j < n ; j += nb) {

  /* define the dpotrf stream */
  cublasStatus = cublasSetStream (Common->cublasHandle[gpuid / Common->numGPU_parallel], Common->gpuStream[gpuid][0]) ;
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS stream") ;
  }

    jb = nb < (n-j) ? nb : (n-j) ;


    /*
     * Perform DSYRK on GPU
     */
    alpha = -1.0;
    beta  = 1.0;

#ifdef REAL
    cublasStatus = cublasDsyrk (
            Common->cublasHandle[gpuid / Common->numGPU_parallel],
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            jb,
            j,
            &alpha, devPtrA + j,
            gpu_lda,
            &beta,  devPtrA + j + j*gpu_lda,
            gpu_lda);
#else
    cublasStatus = cublasZherk (
            Common->cublasHandle[gpuid / Common->numGPU_parallel],
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            jb,
            j,
            &alpha, (cuDoubleComplex*)devPtrA + j,
            gpu_lda,
            &beta,
            (cuDoubleComplex*)devPtrA + j + j*gpu_lda,
            gpu_lda);
#endif

    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
      ERROR (CHOLMOD_GPU_PROBLEM, "GPU cublasDsyrk error") ;
    }


    /* copy back the jb columns on two different streams */
    cudaErr = cudaMemcpy2DAsync (
            A + L_ENTRY*(j + j*lda),
            lda * L_ENTRY * sizeof (double),
            devPtrA + L_ENTRY*(j + j*gpu_lda),
            gpu_lda * L_ENTRY * sizeof (double),
            L_ENTRY * sizeof (double)*jb,
            jb,
            cudaMemcpyDeviceToHost,
            Common->gpuStream[gpuid][0]) ;


      /* wait fo end of DTRSM */
      cudaErr = cudaStreamWaitEvent (Common->gpuStream[gpuid][1], Common->cublasEventPotrf[gpuid][2], 0) ;
      if (cudaErr) {
        ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
      }



    /*
     * Perform DGEMM on GPU
     */
    if ((j+jb) < n) {

  cublasStatus = cublasSetStream (Common->cublasHandle[gpuid / Common->numGPU_parallel], Common->gpuStream[gpuid][1]) ;

#ifdef REAL
      alpha = -1.0 ;
      beta  = 1.0 ;
      cublasStatus = cublasDgemm (
              Common->cublasHandle[gpuid / Common->numGPU_parallel],
              CUBLAS_OP_N,
              CUBLAS_OP_T,
              (n-j-jb),
              jb,
              j,
              &alpha,
              devPtrA + (j+jb),
              gpu_lda,
              devPtrA + (j),
              gpu_lda,
              &beta,
              devPtrA + (j+jb + j*gpu_lda),
              gpu_lda);
#else
      cuDoubleComplex calpha = {-1.0,0.0} ;
      cuDoubleComplex cbeta  = { 1.0,0.0} ;

      cublasStatus = cublasZgemm (
              Common->cublasHandle[gpuid / Common->numGPU_parallel],
              CUBLAS_OP_N,
              CUBLAS_OP_C,
              (n-j-jb),
              jb,
              j,
              &calpha,
              (cuDoubleComplex*)devPtrA + (j+jb),
              gpu_lda,
              (cuDoubleComplex*)devPtrA + (j),
              gpu_lda,
              &cbeta,
              (cuDoubleComplex*)devPtrA + (j+jb + j*gpu_lda),
              gpu_lda);
#endif


      if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
      }

    }


      /* record end of DTRSM  */
      cudaErr = cudaEventRecord (Common->cublasEventPotrf[gpuid][1], Common->gpuStream[gpuid][1]) ;
      if (cudaErr) {
        ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
      }


    /* synchronize stream */
    cudaErr = cudaStreamSynchronize (Common->gpuStream[gpuid][0]) ;
    if (cudaErr) {
      ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
    }


    /* compute the Cholesky factorization of the jbxjb block on the CPU */
    ilda = (int) lda ;
    ijb  = jb ;

#ifdef REAL
    LAPACK_DPOTRF ("L", &ijb, A + L_ENTRY * (j + j*lda), &ilda, &iinfo) ;
#else
    LAPACK_ZPOTRF ("L", &ijb, A + L_ENTRY * (j + j*lda), &ilda, &iinfo) ;
#endif


    /* get parameter that determines if it is positive definite */
    *info = iinfo ;
    if (*info != 0) {
      *info = *info + j ;
      break ;
    }



    /* copy the result back to the GPU */
    cudaErr = cudaMemcpy2DAsync (
            devPtrA + L_ENTRY*(j + j*gpu_lda),
            gpu_lda * L_ENTRY * sizeof (double),
            A + L_ENTRY * (j + j*lda),
            lda * L_ENTRY * sizeof (double),
            L_ENTRY * sizeof (double) * jb,
            jb,
            cudaMemcpyHostToDevice,
            Common->gpuStream[gpuid][0]);

    if (cudaErr) {
      ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
    }



      /* wait fo end of DTRSM */
      cudaErr = cudaStreamWaitEvent (Common->gpuStream[gpuid][0], Common->cublasEventPotrf[gpuid][1], 0) ;
      if (cudaErr) {
        ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
      }


    /*
     * Perform DTRSM on GPU
     */
    if ((j+jb) < n) {

        cublasStatus = cublasSetStream (Common->cublasHandle[gpuid / Common->numGPU_parallel], Common->gpuStream[gpuid][0]) ;

#ifdef REAL
        alpha  = 1.0 ;
        cublasStatus = cublasDtrsm (
                Common->cublasHandle[gpuid / Common->numGPU_parallel],
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_T,
                CUBLAS_DIAG_NON_UNIT,
                (n-j-jb),
                jb,
                &alpha,
                devPtrA + (j + j*gpu_lda),
                gpu_lda,
                devPtrA + (j+jb + j*gpu_lda),
                gpu_lda);
#else
        cuDoubleComplex calpha  = {1.0,0.0};

        cublasStatus = cublasZtrsm (
                Common->cublasHandle[gpuid / Common->numGPU_parallel],
                CUBLAS_SIDE_RIGHT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_C,
                CUBLAS_DIAG_NON_UNIT,
                (n-j-jb),
                jb,
                &calpha,
                (cuDoubleComplex *)devPtrA + (j + j*gpu_lda),
                gpu_lda,
                (cuDoubleComplex *)devPtrA + (j+jb + j*gpu_lda),
                gpu_lda) ;
#endif


      if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
      }


      /* record end of DTRSM  */
      cudaErr = cudaEventRecord (Common->cublasEventPotrf[gpuid][2], Common->gpuStream[gpuid][0]) ;
      if (cudaErr) {
        ERROR (CHOLMOD_GPU_PROBLEM, "CUDA event failure") ;
      }


      /* Copy factored column back to host. */
      cudaErr = cudaMemcpy2DAsync (
              A + L_ENTRY*(j + jb + j * lda),
              lda * L_ENTRY * sizeof (double),
              devPtrA + L_ENTRY*
              (j + jb + j * gpu_lda),
              gpu_lda * L_ENTRY * sizeof (double),
              L_ENTRY * sizeof (double)*
              (n - j - jb), jb,
              cudaMemcpyDeviceToHost,
              Common->gpuStream[gpuid][0]) ;

      if (cudaErr) {
        ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy to device") ;
      }
    }

  } /* end loop over blocks */



  return (1) ;
}










/*
 *  Function:
 *    gpu_triangular_solve_root
 *
 *  Description:
 *    Computes triangular solve for a supernode.
 *    Performs one task:
 *      1. DTRSM
 *
 */
int TEMPLATE2 (CHOLMOD (gpu_triangular_solve_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   double *Lx,
   Int nsrow2,
   Int nscol2,
   Int nsrow,
   Int psx,
   int gpuid
   )
{
  /* local variables */
  int i, j, iwrap, ibuf = 0, iblock = 0, numThreads;
  Int iidx, gpu_lda, gpu_ldb, gpu_rowstep, gpu_row_max_chunk, gpu_row_chunk, gpu_row_start = 0;
  double *devPtrA, *devPtrB;
  cudaError_t cudaErr;
  cublasStatus_t cublasStatus;


  //cudaSetDevice(gpuid / Common->numGPU_parallel);


  numThreads = Common->ompNumThreads;



  /* early exit */
  if ( nsrow2 <= 0 )
  {
    return (0) ;
  }


  /* initialize parameters */
  gpu_lda = ((nscol2+31)/32)*32 ;
  gpu_ldb = ((nsrow2+31)/32)*32 ;
#ifdef REAL
    double alpha  = 1.0 ;
    gpu_row_max_chunk = 768;
#else
    cuDoubleComplex calpha  = {1.0,0.0} ;
    gpu_row_max_chunk = 256;
#endif


  /* initialize device pointers */
  devPtrA = gpu_p->d_Lx_root[gpuid][0];
  devPtrB = gpu_p->d_Lx_root[gpuid][1];


  /* make sure the copy of B has completed */
  cudaStreamSynchronize( Common->gpuStream[gpuid][0] );




  /*
   * Perform blocked TRSM:
   * 1. compute DTRSM
   * 2. copy Lx from pinned to host memory
   * Hide copies behind compute.
   */
  /* loop over blocks */
  while ( gpu_row_start < nsrow2 ) {

    /* set block dimensions */
    gpu_row_chunk = nsrow2 - gpu_row_start;
    if ( gpu_row_chunk  > gpu_row_max_chunk ) {
      gpu_row_chunk = gpu_row_max_chunk;
    }


    cublasStatus = cublasSetStream ( Common->cublasHandle[gpuid / Common->numGPU_parallel], Common->gpuStream[gpuid][ibuf] );

    if ( cublasStatus != CUBLAS_STATUS_SUCCESS ) {
      ERROR ( CHOLMOD_GPU_PROBLEM, "GPU CUBLAS stream");
    }


    /*
     * Perform DTRSM on GPU
     */
#ifdef REAL
    cublasStatus = cublasDtrsm (
            Common->cublasHandle[gpuid / Common->numGPU_parallel],
            CUBLAS_SIDE_RIGHT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T,
            CUBLAS_DIAG_NON_UNIT,
            gpu_row_chunk,
            nscol2,
            &alpha,
            devPtrA,
            gpu_lda,
            devPtrB + gpu_row_start,
            gpu_ldb) ;
#else
    cublasStatus = cublasZtrsm (
            Common->cublasHandle[gpuid / Common->numGPU_parallel],
            CUBLAS_SIDE_RIGHT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_C,
            CUBLAS_DIAG_NON_UNIT,
            gpu_row_chunk,
            nscol2,
            &calpha,
            (const cuDoubleComplex *) devPtrA,
            gpu_lda,
            (cuDoubleComplex *)devPtrB + gpu_row_start ,
            gpu_ldb) ;
#endif


    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
      ERROR (CHOLMOD_GPU_PROBLEM, "GPU CUBLAS routine failure") ;
    }


    /* copy result back to the CPU */
    cudaErr = cudaMemcpy2DAsync ( gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS] +
                                  L_ENTRY*(nscol2+gpu_row_start),
                                  nsrow * L_ENTRY * sizeof (Lx [0]),
                                  devPtrB + L_ENTRY*gpu_row_start,
                                  gpu_ldb * L_ENTRY * sizeof (devPtrB [0]),
                                  gpu_row_chunk * L_ENTRY *
                                  sizeof (devPtrB [0]),
                                  nscol2,
                                  cudaMemcpyDeviceToHost,
                                  Common->gpuStream[gpuid][ibuf]);

    if (cudaErr) {
      ERROR (CHOLMOD_GPU_PROBLEM, "GPU memcopy from device") ;
    }


    /* record end of copy */
    cudaEventRecord ( Common->updateCBuffersFree[gpuid][ibuf], Common->gpuStream[gpuid][ibuf] );


    /* update block dimensions */
    gpu_row_start += gpu_row_chunk;
    ibuf++;
    ibuf = ibuf % CHOLMOD_HOST_SUPERNODE_BUFFERS;
    iblock ++;


    /* only if enough available host buffers */
    if ( iblock >= CHOLMOD_HOST_SUPERNODE_BUFFERS ) {

      Int gpu_row_start2 ;
      Int gpu_row_end ;


      cudaErr = cudaEventSynchronize ( Common->updateCBuffersFree[gpuid][iblock%CHOLMOD_HOST_SUPERNODE_BUFFERS] );
      if ( cudaErr ) {
        ERROR (CHOLMOD_GPU_PROBLEM, "ERROR cudaEventSynchronize") ;
      }


      gpu_row_start2 = nscol2 + (iblock-CHOLMOD_HOST_SUPERNODE_BUFFERS)*gpu_row_max_chunk;
      gpu_row_end = gpu_row_start2+gpu_row_max_chunk;

      if ( gpu_row_end > nsrow ) gpu_row_end = nsrow;


      /* copy Lx from pinned to host memory */
      #pragma omp parallel for num_threads(numThreads) private(i, j, iidx) if ( nscol2 > 32 )
      for ( j=0; j<nscol2; j++ ) {
        for ( i=gpu_row_start2*L_ENTRY; i<gpu_row_end*L_ENTRY; i++ ) {
          iidx = j*nsrow*L_ENTRY+i;
          Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS][iidx];
        }
      }

    } /* end if enough buffers */

  } /* end while loop */




  /* Convenient to copy the L1 block here */
  #pragma omp parallel for num_threads(numThreads) private ( i, j, iidx ) if ( nscol2 > 32 )
  for ( j=0; j<nscol2; j++ ) {
    for ( i=j*L_ENTRY; i<nscol2*L_ENTRY; i++ ) {
      iidx = j*nsrow*L_ENTRY + i;
      Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS][iidx];
    }
  }




  /* now account for the last HSTREAMS buffers */
  for ( iwrap=0; iwrap<CHOLMOD_HOST_SUPERNODE_BUFFERS; iwrap++ ) {

    int i, j;
    Int gpu_row_start2 = nscol2 + (iblock-CHOLMOD_HOST_SUPERNODE_BUFFERS)*gpu_row_max_chunk;

    if (iblock-CHOLMOD_HOST_SUPERNODE_BUFFERS >= 0 && gpu_row_start2 < nsrow ) {

      Int iidx;
      Int gpu_row_end = gpu_row_start2+gpu_row_max_chunk;
      if ( gpu_row_end > nsrow ) gpu_row_end = nsrow;

      cudaEventSynchronize ( Common->updateCBuffersFree[gpuid][iblock%CHOLMOD_HOST_SUPERNODE_BUFFERS] );


      /* copy Lx from pinned to host memory */
      #pragma omp parallel for num_threads(numThreads) private(i, j, iidx) if ( nscol2 > 32 )
      for ( j=0; j<nscol2; j++ ) {
        for ( i=gpu_row_start2*L_ENTRY; i<gpu_row_end*L_ENTRY; i++ ) {
          iidx = j*nsrow*L_ENTRY+i;
          Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS][iidx];
        }
      }
    }
  iblock++;
  } /* end loop over wrap */



  return (1) ;
}











/*
 *  Function:
 *    gpu_copy_supernode_root
 *
 *  Description:
 *    Copies supernode from pinned to host memory.
 *    Case triangular_solve is not called..
 */
void TEMPLATE2 (CHOLMOD (gpu_copy_supernode_root))
  (
   cholmod_common *Common,
   cholmod_gpu_pointers *gpu_p,
   double *Lx,
   Int psx,
   Int nscol,
   Int nscol2,
   Int nsrow,
   int supernodeUsedGPU,
   int gpuid
   )
{
  /* local variables */
  Int iidx, i, j;
  int numThreads;

  //cudaSetDevice(gpuid / Common->numGPU_parallel);

  numThreads = Common->ompNumThreads;


  /* if supernode large enough for GPU */
  if ( supernodeUsedGPU && nscol2 * L_ENTRY >= CHOLMOD_POTRF_LIMIT ) {

    /* synchronize device */
    //cudaDeviceSynchronize();
    cudaStreamSynchronize(Common->gpuStream[gpuid][0]);

    /* copy Lx from pinned to host memory */
    #pragma omp parallel for num_threads(numThreads) private(i, j, iidx) if (nscol>32)
    for ( j=0; j<nscol; j++ ) {
      for ( i=j*L_ENTRY; i<nscol*L_ENTRY; i++ ) {
        iidx = j*nsrow*L_ENTRY+i;
        Lx[psx*L_ENTRY+iidx] = gpu_p->h_Lx_root[gpuid][CHOLMOD_HOST_SUPERNODE_BUFFERS][iidx];
      }
    }

  } /* end if statement */


}










#undef REAL
#undef COMPLEX
#undef ZOMPLEX












