/*
 * File:
 *   cholmod_gpu_kernels
 *
 * Description:
 *   Contains CUDA kernels and their wrappers
 *
 */


/* includes */
#include <stdio.h>
#include "SuiteSparse_config.h"
#include "cholmod_core.h"
#include "cholmod_subtree.h"

/* 64-bit version only */
#define Int SuiteSparse_long










extern "C" {
    /* *** CUDA kernels *** */




    /*
     *  Function:
     *    atomicAdd
     *
     *  Description:
     *    performs atomic add for type (double)
     *
     */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                        __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#endif





    /*
     *  Function:
     *    kernelCreateRelativeMap
     *
     *  Description:
     *    creates maps for a descendant
     *
     *  Parallelism:
     *    parallelism over rows (grid.x).
     *    1D Grid & 1D Blocks.
     */
    __global__ void kernelCreateRelativeMap ( Int *d_Map,
            Int *d_Ls,
            Int *d_RelativeMap,
            Int pdi1,
            Int ndrow )
    {
        /* set global thread index */
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        /* loop over rows */
        if ( tid < ndrow ) {
            d_RelativeMap[tid] = d_Map[d_Ls[pdi1+tid]];         /* set map */
        }

    }





    /*
     *  Function:
     *    kernelCreateMap
     *
     *  Description:
     *    creates maps for a supernode.
     *
     *  Parallelism:
     *    parallelism over rows (grid.x).
     *    1D Grid & 1D Blocks.
     */
    __global__ void kernelCreateMap ( Int *d_Map,
            Int *d_Ls,
            Int psi,
            Int nsrow )
    {
        /* set global thread index */
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        /* loop over rows */
        if ( tid < nsrow ) {
            d_Map[d_Ls[psi+tid]] = ((Int) (tid));               /* set map */
        }

    }





    /*
     *  Function:
     *    kernelCreateMap_batch
     *
     *  Description:
     *    creates maps for a batch of supernodes.
     *
     *  Parallelism:
     *    parallelism over rows (grid.x) and batch (grid.y).
     *    2D Grid & 2D Blocks.
     */
    __global__ void kernelCreateMap_batch ( Int *d_Map,		/* map on device */
            Int *d_Ls,    		/* Ls on device */
            Int *d_psi,		/* list of psi (for each supernode) */
            Int *d_nsrow,		/* list of nsrow */
            int n,
            int nbatch)		/* batch size */
    {
        /* set global thread indices */
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;;

        /* loop over batch (supernodes) */
        if(iy < nbatch) {
            /* loop over rows */
            if(ix < d_nsrow[iy]) {
                int mapid = iy*(n+1);                            	/* map to use (different for each supernode) */
                d_Map[mapid + d_Ls[d_psi[iy]+ix]] = ((Int) (ix));       	/* set map */
            }
        }
    }





    /*
     * Function:
     *   kernelAddUpdate
     *
     * Description:
     *   updates supernode with schur complement (dsyrk & dgemm) of its descendant (type double)
     *
     * Parallelism:
     *   parallelism over rows (block.x), and columns (block.y).
     *   1D Grid & 2D Blocks
     *
     */
    __global__ void kernelAddUpdate ( double *d_A,
            double *devPtrC,
            Int *d_RelativeMap,
            Int ndrow1,
            Int ndrow2,
            Int nsrow )
    {
        /* set global thread indices */
        int idrow = blockIdx.x * blockDim.x + threadIdx.x;
        int idcol = blockIdx.y * blockDim.y + threadIdx.y;

        /* loop over rows and columns */
        if ( idrow < ndrow2  && idcol < ndrow1 ) {
            Int idx = d_RelativeMap[idrow] + d_RelativeMap[idcol] * nsrow;
            d_A[idx] += devPtrC[idrow+ndrow2*idcol];                            /* add schur complement to supernode */
        }
    }





    /*
     * Function:
     *   kernelAddComplexUpdate
     *
     * Description:
     *   updates supernode with schur complement (dsyrk & dgemm) of its descendant (type complex)
     *
     * Parallelism:
     *   parallelism over rows (block.x), and columns (block.y).
     *   1D Grid & 2D Blocks
     *
     */
    __global__ void kernelAddComplexUpdate ( double *d_A,
            double *devPtrC,
            Int *d_RelativeMap,
            Int ndrow1,
            Int ndrow2,
            Int nsrow )
    {
        /* set global thread indices */
        int idrow = blockIdx.x * blockDim.x + threadIdx.x;
        int idcol = blockIdx.y * blockDim.y + threadIdx.y;

        /* loop over rows and columns */
        if ( idrow < ndrow2  && idcol < ndrow1 ) {
            Int idx = d_RelativeMap[idrow] + d_RelativeMap[idcol] * nsrow;
            d_A[idx*2] += devPtrC[(idrow+ndrow2*idcol)*2];                      /* add schur complement to supernode */
            d_A[idx*2+1] += devPtrC[(idrow+ndrow2*idcol)*2+1];
        }
    }





    /*
     * Function:
     *   kernelAddUpdate_batch
     *
     * Description:
     *   updates a batch of supernodes with their schur complement (dsyrk & dgemm) of all descendants
     *   in the batch.
     *
     * Parallelism:
     *   parallelism over batch (grid), rows (block.x), and columns (block.y).
     *
     *   2D Blocks: idx for rows, idy for columns. Each block loops over block tiles
     *              of rows (blockDim.x) and columns (blockDim.y).
     *
     *   1D Grid: for batch of descendants. Each block performs mapping of a descendant
     *            into its respective supernode.
     *
     */
    __global__ void kernelAddUpdate_batch ( double *d_A,
            double **devPtrC, /* list of schur complements */
            Int *d_Map, 	  /* map on device */
            Int *d_Ls,
            Int n,
            Int maxdrow,      /* maximum ndrow in batch  */
            Int maxdcol,      /* maximum ndcol in batch */
            Int *d_pdi1, 	  /* list of pdi1 (for each descendant) */
            Int *d_ndrow1,    /* list of ndrow1 */
            Int *d_ndrow2,    /* list of ndrow2 */
            Int *d_psx,       /* list of psx (for each supernode) */
            Int *d_nsrow,     /* list of nsrow */
            Int *dlist,       /* list of supernode id's (for each descendant) */
            int nbatch)       /* batch size (# descendants in batch pool) */
    {
        /* set thread & block indices */
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int batch = blockIdx.x;

        /* local variables */
        int idrow, idcol, irow, icol;

        /* loop over descendants */
        if(batch < nbatch) {

            Int ndrow1 = d_ndrow1[batch];	/* descendant dimensions */
            Int ndrow2 = d_ndrow2[batch];

            int super = dlist[batch];		/* supernode of the current descendant */

            Int psx = d_psx[super];             /* supernode dimensions */
            Int nsrow = d_nsrow[super];

            idrow = tx;

            /* loop over idrow tiles */
            for(irow = 0; irow < maxdrow ; irow += blockDim.x) {

                idcol = ty;

                /* loop over idcol tiles */
                for(icol = 0; icol < maxdcol ; icol += blockDim.y) {

                    /* check if decendant within bounds */
                    if(idrow < ndrow2 && idcol < ndrow1 ) {

                        int mapid = (n+1)*super;					/* map to use for supernode */
                        Int pdi1 = d_pdi1[batch];

                        Int isrow = d_Map[mapid + d_Ls[pdi1+idrow]];			/* supernode dimensions */
                        Int iscol = d_Map[mapid + d_Ls[pdi1+idcol]];

                        /* check for triangular part */
                        if(isrow >= iscol) {
                            Int idx = psx + isrow + iscol * nsrow;			/* mapping index */
                            atomicAdd(&d_A[idx],devPtrC[batch][idrow+ndrow2*idcol]);	/* add schur complement to supernode */
                        }
                    }

                    /* synchronize threads */
                    __syncthreads();

                    idcol += blockDim.y;
                } /* end icol tile loop */

                idrow += blockDim.x;
            } /* end irow tile loop */

        } /* end loop over descendants */
    }





    /*
     * Function:
     *   kernelAddUpdate
     *
     * Description:
     *   updates supernode with schur complement (dsyrk & dgemm) of current descendant
     *
     * Parallelism:
     *   parallelism over rows (grid.x) and columns (grid.y).
     *   2D Grid and 2D Block.
     *
     */
    __global__ void kernelAddUpdate_large ( double *d_A,
            double *devPtrC, /* list of schur complements */
            Int *d_Map,       /* map on device */
            Int *d_Ls,	    /* Ls on device */
            Int d_pdi1,       /* pdi1 (for current descendant) */
            Int d_ndrow1,     /* ndrow1 */
            Int d_ndrow2,     /* ndrow2 */
            Int d_psx,        /* psx (for supernode of current descendant) */
            Int d_nsrow,      /* nsrow */
            Int mapid)	    /* id for map */
    {
        /* set global thread indices */
        int idrow = blockIdx.x * blockDim.x + threadIdx.x;
        int idcol = blockIdx.y * blockDim.y + threadIdx.y;


        Int ndrow1 = d_ndrow1;                   	  /* descendant dimensions */
        Int ndrow2 = d_ndrow2;

        /* loop over rows & cols */
        if(idrow < ndrow2 && idcol < ndrow1 ) {

            Int psx = d_psx;           			 /* supernode dimensions */
            Int nsrow = d_nsrow;
            Int pdi1 = d_pdi1;
            Int isrow = d_Map[mapid + d_Ls[pdi1+idrow]];
            Int iscol = d_Map[mapid + d_Ls[pdi1+idcol]];

            /* check for triangular part */
            if(isrow >= iscol) {
                Int idx = psx + isrow + iscol * nsrow;  			 /* mapping index */
                atomicAdd(&d_A[idx],devPtrC[idrow+ndrow2*idcol]); 	 /* add schur complement to supernode */
            }
        }
    }





    /*
     * Function
     *   kernelSetLx
     *
     * Description:
     *  constructs Lx (factor) for a batch of supernodes
     *
     * Parallelism:
     * parallelism over..
     * 2D grid: ....
     */
    __global__ void kernelSetLx_batch ( double *Lx,     	/* Lx *factor) on device */
               double *Ax,     	/* Ax on device */
               Int *Ap,		/* Ap on device */
               Int *Ai,		/* Ai on device */
               Int *Map,       	/* map on device */
               Int *d_nsrow,   	/* list of nsrow (for current supernde) */
               Int *d_psx,     	/* list of psx */
               Int *d_k1,      	/* list of k1 */
               Int *d_k2,      	/* list of k2 */
               Int n,
               int nbatch)     	/* batch size (# supernodes) */
               {
                   /* set global thread index */
                   int idx = blockIdx.x * blockDim.x + threadIdx.x;
                   int idy = blockIdx.y * blockDim.y + threadIdx.y;

                   /* local variable */
                   int node=0;

                   /* loop over supernodes  */
                   for(node = 0; node < nbatch; node++) {

                       Int k1 = d_k1[node];        	 /* supernode dimensions */
                       Int k2 = d_k2[node];

                       /* loop over columns */
                       if(idx < (k2-k1)) {
                           Int k = idx + k1;
                           Int pstart = Ap[k];
                           Int pend = Ap[k+1];

                           /* loop over.. */
                           if(idy < pend-pstart) {
                               Int nsrow = d_nsrow[node];	/* supernode dimensions */
                               Int p = idy+pstart;
                               Int i = Ai[p];

                               /* check for triangular part */
                               if (i >= k) {
                                   Int imap = Map [node*(n+1) + i] ; 	/* map to use (different for each supernode) */

                                   /* only for map's for the current supernode */
                                   if (imap >= 0 && imap < nsrow) {
                                       Int id;
                                       Int psx = d_psx[node];
                                       id = imap+(psx+(k-k1)*nsrow);
                                       Lx[id] += Ax[p];
                                   }
                               }
                           } /* end loop over.. */
                       } /* end loop over columns */

                       /* synchronize threads */
                       __syncthreads();

                   } /* end loop over supernodes */
               }





    /*
     * Function:
     *   kernelSumA
     *
     * Description:
     *   sums A assembly (type double)
     *
     * Parallelism:
     *   parallelism over rows (block.x), and columns (block.y).
     *   1D Grid & 2D Blocks
     *
     */
    __global__ void kernelSumA ( double *a1,
            double *a2,
            const double alpha,
            Int nsrow,
            Int nscol )
    {
        /* set global thread indices */
        int isrow = blockIdx.x * blockDim.x + threadIdx.x;
        int iscol = blockIdx.y * blockDim.y + threadIdx.y;

        /* loop over rows and columns */
        if ( isrow < nsrow && iscol < nscol ) {
            Int idx = iscol*nsrow + isrow;
            a1[idx] += alpha * a2[idx];                         /* sum A components */
        }
    }





    /*
     * Function:
     *   kernelSumComplexA
     *
     * Description:
     *   sums A assembly (type complex)
     *
     * Parallelism:
     *   parallelism over rows (block.x), and columns (block.y).
     *   1D Grid & 2D Blocks
     *
     */
    __global__ void kernelSumComplexA ( double *a1,
            double *a2,
            const double alpha,
            Int nsrow,
            Int nscol )
    {
        /* set global thread indices */
        int isrow = blockIdx.x * blockDim.x + threadIdx.x;
        int iscol = blockIdx.y * blockDim.y + threadIdx.y;

        /* loop over rows and columns */
        if ( isrow < nsrow && iscol < nscol ) {
            Int idx = iscol*nsrow + isrow;
            a1[idx*2] += alpha * a2[idx*2];                   /* sum A components */
            a1[idx*2+1] += alpha * a2[idx*2+1];
        }
    }





    /*
     * Function:
     *   kernelCopyLx_small
     *
     * Description:
     *   copies batch of supernodes from device to pinned memory
     *
     * Parallelism:
     *   parallelism over rows*columns (grid.x) and supernodes (grid.y).
     *   2D Grid and 2D Block.
     *
     */
    __global__ void kernelCopyLx_small ( double *a1, 	/* pinned buffer (dst) */
            double *a2, 	/* device buffer (src) */
            Int *d_psx,	/* list of psx (for each supernode) */
            Int *d_nsrow, 	/* list of nsrow */
            Int *d_nscol, 	/* list of nscol */
            int nbatch ) 	/* batch size (# supernoeds) */
    {
        /* set global thread indices */
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchIdx = blockIdx.y * blockDim.y + threadIdx.y;

        /* loop over supernodes */
        if(batchIdx < nbatch) {
            /* loop over nsrow*nscol */
            if(idx < d_nsrow[batchIdx]*d_nscol[batchIdx]) {
                Int id = d_psx[batchIdx] + idx;
                a1[id] = a2[id];
            } /* end loop over nsrow*nscol */
        } /* end loop over supernodes */
    }










    /* *** CUDA wrappers *** */





    /*
     * Function:
     *   createRelativeMapOnDevice
     *
     * Description:
     *   wrapper for kernel: kernelCreateRelativeMap
     *
     */
    void createRelativeMapOnDevice ( Int *d_Map,
            Int *d_Ls,
            Int *d_RelativeMap,
            Int pdi1,
            Int ndrow,
            cudaStream_t* astream )
    {
        /* set blocks & grids */
        dim3 grids;
        dim3 blocks(256);

        grids.x = (ndrow + blocks.x - 1)/blocks.x;

        /* call kernel */
        kernelCreateRelativeMap <<<grids, blocks, 0, *astream>>> ( d_Map, d_Ls, d_RelativeMap, pdi1, ndrow);

    }





    /*
     * Function:
     *   createMapOnDevice
     *
     * Description:
     *   wrapper for kernel: kernelCreateMap
     *
     */
    void createMapOnDevice ( Int *d_Map,
            Int *d_Ls,
            Int psi,
            Int nsrow,
            cudaStream_t stream )
    {
        /* set blocks & grids */
        dim3 grids;
        dim3 blocks(32);

        grids.x = (nsrow + blocks.x - 1)/blocks.x;

        /* call kernel */
        kernelCreateMap <<<grids, blocks, 0, stream>>> ( d_Map, d_Ls, psi, nsrow );

    }





    /*
     * Function:
     *   createMapOnDevice_batch
     *
     * Description:
     *   wrapper for kernel: kernelCreateMap_batch
     *
     */
    void createMapOnDevice_batch ( Int *d_Map,    	   	/* map on device */
            Int *d_Ls,	   	/* Ls on device */
            Int *d_psi,     	    	/* list of psi (for each supernode */
         Int *d_nsrow,     	/* list of nsrow */
         Int maxsnsrow,      	/* maximum nsrow in batch of supernodes */
         int n,
         int nbatch,             	/* batch size (# supernodes) */
         cudaStream_t* astream ) 	/* cuda stream */
            {
            /* set blocks & grids */
            dim3 grids;
            dim3 blocks(32,32);

            grids.x = (maxsnsrow + blocks.x - 1)/blocks.x;
            grids.y = (nbatch + blocks.y - 1)/blocks.y;

            /* call kernel */
            kernelCreateMap_batch <<<grids, blocks, 0, *astream>>> ( d_Map, d_Ls, d_psi, d_nsrow, n, nbatch );

            }





    /*
     * Function:
     *   addUpdateOnDevice
     *
     * Description:
     *   wrapper for kernel: kernelAddUpdate
     *
     */
    void addUpdateOnDevice ( double *d_A,
            double *devPtrC,
            Int *d_RelativeMap,
            Int ndrow1,
            Int ndrow2,
            Int nsrow,
            cudaStream_t* astream )
    {
        /* set blocks & grids */
        dim3 grids;
        dim3 blocks(16,16);

        grids.x = (ndrow2 + blocks.x - 1)/blocks.x;
        grids.y = (ndrow1 + blocks.y - 1)/blocks.y;

        /* call kernel */
        kernelAddUpdate <<<grids, blocks, 0, *astream>>> ( d_A, devPtrC, d_RelativeMap, ndrow1, ndrow2, nsrow );

    }





    /*
     * Function:
     *   addComplexUpdateOnDevice
     *
     * Description:
     *   wrapper for kernel: kernelAddComplexUpdate
     *
     */
    void addComplexUpdateOnDevice ( double *d_A,
            double *devPtrC,
            Int *d_RelativeMap,
            Int ndrow1,
            Int ndrow2,
            Int nsrow,
            cudaStream_t* astream )
    {
        /* set blocks & grids */
        dim3 grids;
        dim3 blocks(16,16);

        grids.x = (ndrow2 + blocks.x - 1)/blocks.x;
        grids.y = (ndrow1 + blocks.y - 1)/blocks.y;

        /* call kernel */
        kernelAddComplexUpdate <<<grids, blocks, 0, *astream>>> ( d_A, devPtrC, d_RelativeMap, ndrow1, ndrow2, nsrow );

    }





    /*
     * Function:
     *   addUpdateOnDevice_batch
     *
     * Description:
     *   wrapper for kernel: kernelAddUpdate_batch
     *
     */
    void addUpdateOnDevice_batch ( double *d_A,
            double **devPtrC,
            Int *d_Map, 		/* map on device */
            Int *d_Ls,		/* Ls on device */
            Int n,
            Int *pdi1, 		/* list of pdi1 (for each descendant) */
            Int *ndrow1, 		/* list of ndrow1 */
            Int *ndrow2, 		/* list of ndrow2 */
            Int *psx, 		/* list of psx (for each supernode) */
            Int *nsrow,		/* list of nsrow */
            Int *dlist,
            Int *max_dim,		/* maximum ndrow1 & ndrow2 in batch of descendants */
            int nbatch, 		/* batch size (# descendants) */
            cudaStream_t* astream )	/* cuda stream */
    {
        /* set grids & blocks */
        dim3 grids(nbatch);
        dim3 blocks(16,16);

        /* call kernel */
        kernelAddUpdate_batch <<<grids, blocks, 0, *astream>>> ( d_A, devPtrC, d_Map, d_Ls, n, max_dim[2], max_dim[1],
                pdi1, ndrow1, ndrow2, psx, nsrow, dlist, nbatch);
    }





    /*
     * Function:
     *   addUpdateOnDevice
     *
     * Description:
     *   wrapper for kernel: kernelAddUpdate
     *
     */
    void addUpdateOnDevice_large ( double *d_A,
            double *devPtrC,
            Int *d_Map,           	/* map on device */
            Int *d_Ls,		/* Ls on device */
            Int pdi1,
            Int ndrow1,
            Int ndrow2,
            Int psx,
            Int nsrow,
            Int mapid,		/* map id of supernode */
            cudaStream_t* astream )/* cuda stream */
    {
        /* set grids & blocks */
        dim3 grids;
        dim3 blocks(16,16);

        grids.x = (ndrow2 + blocks.x - 1)/blocks.x;
        grids.y = (ndrow1 + blocks.y - 1)/blocks.y;

        /* call kernel */
        kernelAddUpdate_large <<<grids, blocks, 0, *astream>>> ( d_A, devPtrC, d_Map, d_Ls, pdi1, ndrow1, ndrow2, psx, nsrow, mapid);

    }





    /*
     * Function:
     *   initLxOnDevice_batch
     *
     * Description:
     *   wrapper for kernel: kernelSetLx_batch
     *
     */
    void initLxOnDevice_batch ( double *d_Lx,		/* Lx (factor) on device */
            double *d_Ax,       	/* Ax on device */
            Int *d_Ap,          	/* Ap on device */
            Int *d_Ai,          	/* Ai on device */
            Int *d_Map,         	/* map on device */
            Int *d_nsrow,       	/* list of snrow (for each supernode) */
            Int *d_psx,      		/* list of psx */
            Int *d_k1,			/* list of k1 */
            Int *d_k2,			/* list of k2 */
            Int nzmax,
            Int maxkdif,
            Int n,
            int nbatch,			/* batch size (# supernodes) */
            cudaStream_t* astream)  	/* cuda stream */
    {
        /* set grids & blocks */
        dim3 grids;
        dim3 blocks(16,16);

        grids.x = (maxkdif + blocks.x - 1)/blocks.x;
        grids.y = (nzmax + blocks.y - 1)/blocks.y;

        /* call kernel */
        kernelSetLx_batch <<<grids, blocks, 0, *astream>>> ( d_Lx, d_Ax, d_Ap, d_Ai, d_Map, d_nsrow, d_psx, d_k1, d_k2, n, nbatch);

    }





    /*
     * Function:
     *   sumAOnDevice
     *
     * Description:
     *   wrapper for kernel: kernelSumA
     *
     */
    void sumAOnDevice ( double *a1,
            double *a2,
            const double alpha,
            Int nsrow,
            Int nscol,
            cudaStream_t stream )
    {
        /* set blocks & grids */
        dim3 grids;
        dim3 blocks(16,16);

        grids.x = (nsrow + blocks.x - 1)/blocks.x;
        grids.y = (nscol + blocks.y - 1)/blocks.y;

        /* call kernel */
        kernelSumA <<<grids, blocks, 0, stream>>> ( a1, a2, alpha, nsrow, nscol );

    }





    /*
     * Function:
     *   sumComplexAOnDevice
     *
     * Description:
     *   wrapper for kernel: sumComplexAOnDevice
     *
     */
    void sumComplexAOnDevice ( double *a1,
            double *a2,
            const double alpha,
            Int nsrow,
            Int nscol,
            cudaStream_t stream )
    {
        /* set blocks & grids */
        dim3 grids;
        dim3 blocks(16,16);

        grids.x = (nsrow + blocks.x - 1)/blocks.x;
        grids.y = (nscol + blocks.y - 1)/blocks.y;

        /* call kernel */
        kernelSumComplexA <<<grids, blocks, 0, stream>>> ( a1, a2, alpha, nsrow, nscol );

    }





    /*
     * Function:
     *   copyLx_small
     *
     * Description:
     *   wrapper for kernel: kernelCopyLx_small
     *
     */
    void copyLx_small ( double *d_A,		/* pinned buffer (dst) */
            double *d_B, 		/* device buffer (src) */
            Int *psx, 			/* list of psx (for each supernode) */
            Int *nsrow, 		/* list of nsrow */
            Int *nscol, 		/* list of nscol */
            int batch,			/* batch size (# supernodes) */
            Int maxnsrownscol, 		/* max nsrow*nscol in batch of supernodes */
            cudaStream_t* astream )	/* cuda stream */
    {
        /* set grids & blocks */
        dim3 grids;
        dim3 blocks;

        /* select block size to prevent idle threads */
        if(batch <= 4) 	blocks.y = 4;
        else if(batch <=8)	blocks.y = 8;
        else if(batch <= 16) 	blocks.y = 16;
        else if (batch <= 32) blocks.y = 32;
        blocks.x = 32;

        grids.x = (maxnsrownscol + blocks.x - 1)/blocks.x;
        grids.y = (batch + blocks.y - 1)/blocks.y;

        /* call kernel */
        kernelCopyLx_small <<<grids, blocks, 0, *astream>>> ( d_A, d_B, psx, nsrow, nscol, batch);

    }





    __global__ void kernelCreateMap_factorized
        ( Int *Map,		/* map on device */
          Int *Ls,    		/* Ls on device */
          Int pdi,
          Int ndrow )
        {
            /* set global thread indices */
            int ix = blockIdx.x * blockDim.x + threadIdx.x;

            /* loop over batch (supernodes) */
            /* loop over rows */
            if(ix < ndrow) {
                Map[Ls[pdi+ix]] = (Int) ix;       	/* set map */
            }

            /* synchronize threads */
            __syncthreads();
        }

    void createMapOnDevice_factorized
        ( Int *d_Map,    	   	/* map on device */
          Int *d_Ls,	   	/* Ls on device */
          Int pdi,
          Int ndrow,
          cudaStream_t stream ) 	/* cuda stream */
        {
            /* set blocks & grids */
            dim3 grids;
            dim3 blocks(32);

            grids.x = (ndrow + blocks.x - 1)/blocks.x;

            /* call kernel */
            kernelCreateMap_factorized <<<grids, blocks, 0, stream>>> ( d_Map, d_Ls, pdi, ndrow );

        }

    __global__ void kernelSetLx_factorized
        ( double *Lx,     	/* Lx *factor) on device */
        double *Ax,     	/* Ax on device */
        Int *Ap,		/* Ap on device */
        Int *Ai,		/* Ai on device */
        Int *Map,       	/* map on device */
        Int kd1,
        Int kd2,
        Int ndrow,
        Int ndrow0 )
        {
            /* set global thread index */
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;

            /* loop over columns */
            if(idx < (kd2-kd1)) {
                Int k = idx + kd1;
                Int pstart = Ap[k];
                Int pend = Ap[k+1];

                /* loop over.. */
                if(idy < pend-pstart) {
                    Int p = idy+pstart;
                    Int i = Ai[p];

                    /* check for triangular part */
                    if (i >= k) {
                        Int imap = Map [i] - (ndrow - ndrow0) ; 	/* map to use (different for each supernode) */

                        /* only for map's for the current supernode */
                        if (imap >= 0 && imap < ndrow0) {
                            Int id;
                            id = imap+((k-kd1)*ndrow0);
                            Lx[id] = Ax[p];
                        }
                    }
                } /* end loop over.. */
            } /* end loop over columns */

            /* synchronize threads */
            __syncthreads();
        }

    void initLxOnDevice_factorized
        ( double *d_Lx,		/* Lx (factor) on device */
          double *d_Ax,       	/* Ax on device */
          Int *d_Ap,          	/* Ap on device */
          Int *d_Ai,          	/* Ai on device */
          Int *d_Map,         	/* map on device */
          Int kd1,
          Int kd2,
          Int ndcol,
          Int ndrow,
          Int ndrow0,
          Int nzmax,
          cudaStream_t stream )  	/* cuda stream */
        {
            /* set grids & blocks */
            dim3 grids;
            dim3 blocks(16,16);

            grids.x = (ndcol + blocks.x - 1)/blocks.x;
            grids.y = (nzmax + blocks.y - 1)/blocks.y;

            /* call kernel */
            kernelSetLx_factorized <<<grids, blocks, 0, stream>>> ( d_Lx, d_Ax, d_Ap, d_Ai, d_Map, kd1, kd2, ndrow, ndrow0 );

        }

    __global__ void kernelAddUpdate_factorized
        ( double *d_A,
          double *d_C, /* list of schur complements */
          Int *d_Ls,	    /* Ls on device */
          Int *d_Map,
          Int d_k1,
          Int d_k2,
          Int d_psi,        /* psi */
          Int d_psend,      /* psend */
          Int d_nsrow,      /* nsrow */
          Int d_pdi1,       /* pdi1 (for current descendant) */
          Int d_ndrow0,     /* ndrow0 */
          Int d_ndrow1,     /* ndrow1 */
          Int d_ndrow2 )    /* ndrow2 */
        {
            /* set global thread indices */
            int idrow = blockIdx.x * blockDim.x + threadIdx.x;
            int idcol = blockIdx.y * blockDim.y + threadIdx.y;


            Int k1 = d_k1;
            Int k2 = d_k2;
            Int psi = d_psi;
            Int psend = d_psend;
            Int nsrow = d_nsrow;
            Int pdi1 = d_pdi1;
            Int ndrow0 = d_ndrow0;
            Int ndrow1 = d_ndrow1;
            Int ndrow2 = d_ndrow2;

            /* loop over rows & cols */
            if( idcol < ndrow1 && idrow < ndrow2 ) {

                Int iscol = d_Map[d_Ls[pdi1+idcol]];
                Int isrow = d_Map[d_Ls[pdi1+idrow]];

                /* check for triangular part */
                if(isrow >= iscol) {
                    Int idx = isrow + iscol * nsrow;  			 /* mapping index */
                    d_A[idx] += d_C[idrow+idcol*ndrow0]; 	 /* add schur complement to supernode */
                    //atomicAdd(&d_A[idx], d_C[idrow+idcol*ndrow0]);	/* add schur complement to supernode */
                }
            }

            /* synchronize threads */
            __syncthreads();
        }

    void addUpdateOnDevice_factorized
        ( double *d_A,
          double *d_C,
          Int *d_Ls,		/* Ls on device */
          Int *d_Map,
          Int k1,
          Int k2,
          Int psi,
          Int psend,
          Int nsrow,
          Int pdi1,
          Int ndrow0,
          Int ndrow1,
          Int ndrow2,
          cudaStream_t stream )/* cuda stream */
        {
            /* set grids & blocks */
            dim3 grids;
            dim3 blocks(16,16);

            grids.x = (ndrow2 + blocks.x - 1)/blocks.x;
            grids.y = (ndrow1 + blocks.y - 1)/blocks.y;

            /* call kernel */
            kernelAddUpdate_factorized <<<grids, blocks, 0, stream>>> ( d_A, d_C, d_Ls, d_Map, k1, k2, psi, psend, nsrow, pdi1, ndrow0, ndrow1, ndrow2 );

        }

} /* end extern C */
