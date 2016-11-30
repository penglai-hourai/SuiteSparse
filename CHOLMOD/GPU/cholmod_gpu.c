/* ========================================================================== */
/* === GPU/cholmod_gpu -===================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/GPU Module.  Copyright (C) 2014, Timothy A. Davis.
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Primary routines:
 * -----------------
 * cholmod_gpu_memorysize       determine free memory on current GPU
 * cholmod_gpu_probe            ensure a GPU is available
 * cholmod_gpu_allocate         allocate GPU resources
 * cholmod_gpu_deallocate       free GPU resources
 */

#include "cholmod_internal.h"
#include "cholmod_core.h"
#include "cholmod_gpu.h"
#include "stdio.h"
#ifdef GPU_BLAS
#include <cuda_runtime.h>
#endif

#define MINSIZE (64 * 1024 * 1024)

/* ========================================================================== */
/* === cholmod_gpu_memorysize =============================================== */
/* ========================================================================== */

/* Determine the amount of free memory on the current GPU.  To use another
 * GPU, use cudaSetDevice (k) prior to calling this routine, where k is an
 * integer in the range 0 to the number of devices-1.   If the free size is
 * less than 64 MB, then a size of 1 is returned.  Normal usage:
 *
 *  Common->useGPU = 1 ;
 *  err = cholmod_gpu_memorysize (&totmem, &availmem, Common);
 *  Returns 1 if GPU requested but not available, 0 otherwise
 */

#ifdef GPU_BLAS

static int poll_gpu (size_t s)          /* TRUE if OK, FALSE otherwise */
{
    /* Returns TRUE if the GPU has a block of memory of size s,
       FALSE otherwise.  The block of memory is immediately freed. */
    void *p = NULL ;
    /* double t = SuiteSparse_time ( ) ; */
    if (s == 0)
    {
        return (FALSE) ;
    }
    if (cudaMalloc (&p, s) != cudaSuccess)
    {
        /* t = SuiteSparse_time ( ) - t ; */
        /* printf ("s %lu failed, time %g\n", s, t) ; */
        return (FALSE) ;
    }
    cudaFree (p) ;
    /* t = SuiteSparse_time ( ) - t ; */
    /* printf ("s %lu OK time %g\n", s, t) ; */
    return (TRUE) ;
}

#endif

int CHOLMOD(gpu_memorysize)      /* returns 1 on error, 0 otherwise */
(
    size_t         *total_mem,
    size_t         *available_mem,
    cholmod_common *Common
)
{
    size_t good, bad, s, total_free, total_memory ;
    int k ;
    double t ;

    *total_mem = 0;
    *available_mem = 0;
#ifndef DLONG
    return 0;
#endif

    if (Common->useGPU != 1)
    {
        return (0) ;                    /* not using the GPU at all */
    }

#ifdef GPU_BLAS

    /* find the total amount of free memory */
    t = SuiteSparse_time ( ) ;
    cudaMemGetInfo (&total_free, &total_memory) ;
    t = SuiteSparse_time ( ) - t ;
    /* printf ("free %lu tot %lu time %g\n", total_free, total_memory, t) ; */

    *total_mem = total_memory;

    if (total_free < MINSIZE)
    {
        return (1) ;                    /* not even 64MB; return failure code */
    }

    /* try a bit less than the total free memory */
    s = MAX (MINSIZE, total_free*0.98) ;
    if (poll_gpu (s))
    {
        /* printf ("quick %lu\n", s) ; */
        *available_mem = s;
        return (0) ;  /* no error */
    }

    /* ensure s = 64 MB is OK */
    if (!poll_gpu (MINSIZE))
    {
        return (1) ;                    /* not even 64MB; return failure code */
    }

    /* 8 iterations of binary search */
    good = MINSIZE ;                    /* already known to be OK */
    bad  = total_free ;                 /* already known to be bad */
    for (k = 0 ; k < 8 ; k++)
    {
        s = (good + bad) / 2 ;
        if (poll_gpu (s))
        {
            good = s ;                  /* s is OK, increase good */
        }
        else
        {
            bad = s ;                   /* s failed, decrease bad */
        }
    }

    /* printf ("final %lu\n", good) ; */
    *available_mem = good;

#endif

    return (0) ; /* no error */
}


/* ========================================================================== */
/* === cholmod_gpu_probe ==================================================== */
/* ========================================================================== */
/*
 * Used to ensure that a suitable GPU is available.  As this version of
 * CHOLMOD can only utilize a single GPU, only the default (i.e. selected as
 * 'best' by the NVIDIA driver) is verified as suitable.  If this selection
 * is incorrect, the user can select the proper GPU with the
 * CUDA_VISIBLE_DEVICES environment variable.
 *
 * To be considered suitable, the GPU must have a compute capability > 1 and
 * more than 1 GB of device memory.
 */

int CHOLMOD(gpu_probe) ( cholmod_common *Common )
{

#ifdef GPU_BLAS
    int ngpus, idevice;
    double tstart, tend;
    struct cudaDeviceProp gpuProp;

    if (Common->useGPU != 1)
    {
        return (0) ;
    }

    cudaGetDeviceCount(&ngpus);

    if ( ngpus ) {
        cudaGetDevice ( &idevice );
        cudaGetDeviceProperties ( &gpuProp, idevice );
        if ( gpuProp.major > 1 && 1.0e-9*gpuProp.totalGlobalMem > 1.0 ) {
            return 1;  /* useGPU = 1 */
        }
    }
    CHOLMOD_GPU_PRINTF (("GPU WARNING: useGPUs was selected, "
        "but no applicable GPUs were found. useGPU reset to FALSE.\n")) ;
#endif

    /* no GPU is available */
    return 0;  /* useGPU = 0 */
}

/* ========================================================================== */
/* === cholmod_gpu_deallocate =============================================== */
/* ========================================================================== */

/*
 * Deallocate all GPU related buffers.
 */

int CHOLMOD(gpu_deallocate) ( cholmod_common *Common, int device )
{

#ifdef GPU_BLAS
    cudaError_t cudaErr;

    cudaSetDevice(device);

    if ( Common->dev_mempool[device] )
    {
        /* fprintf (stderr, "free dev_mempool[%d]\n", device) ; */
        cudaErr = cudaFree (Common->dev_mempool[device]);
        /* fprintf (stderr, "free dev_mempool[%d] done\n", device) ; */
        if ( cudaErr )
        {
            ERROR ( CHOLMOD_GPU_PROBLEM,
                    "GPU error when freeing device memory.");
        }
    }
    Common->dev_mempool[device] = NULL;
    Common->dev_mempool_size[device] = 0;

    if ( Common->host_pinned_mempool[device] )
    {
        /* fprintf (stderr, "free host_pinned_mempool[%d]\n", device) ; */
        cudaErr = cudaFreeHost ( Common->host_pinned_mempool[device] );
        /* fprintf (stderr, "free host_pinned_mempool[%d] done\n", device) ; */
        if ( cudaErr )
        {
            ERROR ( CHOLMOD_GPU_PROBLEM,
                    "GPU error when freeing host pinned memory.");
        }
    }
    Common->host_pinned_mempool[device] = NULL;
    Common->host_pinned_mempool_size[device] = 0;

    CHOLMOD (gpu_end) (Common, device) ;
#endif

    return (0);
}

/* ========================================================================== */
/* === cholmod_gpu_end ====================================================== */
/* ========================================================================== */

void CHOLMOD(gpu_end) (cholmod_common *Common, int device)
{
#ifdef GPU_BLAS
    int k, vdevice;

#ifdef MAGMA
    for (k = 0 ; k < CHOLMOD_HOST_SUPERNODE_BUFFERS ; k++)
    {
        for (vdevice = device * Common->cuda_gpu_parallel; vdevice < (device + 1) * Common->cuda_gpu_parallel; vdevice++)
        if (Common->magmaQueue[vdevice] [k])
        {
            magma_queue_destroy(Common->magmaQueue[vdevice][k]) ;
            Common->magmaQueue[vdevice][k] = NULL ;
        }
    }
#endif

    /* ------------------------------------------------------------------ */
    /* destroy Cublas Handle */
    /* ------------------------------------------------------------------ */

    for (vdevice = device * Common->cuda_gpu_parallel; vdevice < (device + 1) * Common->cuda_gpu_parallel; vdevice++)
    if (Common->cublasHandle[vdevice])
    {
        /* fprintf (stderr, "destroy cublas[%d] %p\n", device, Common->cublasHandle[vdevice]) ; */
        cublasDestroy (Common->cublasHandle[vdevice]) ;
        /* fprintf (stderr, "destroy cublas[%d] done\n", device) ; */
        Common->cublasHandle[vdevice] = NULL ;
    }

    {
    /* ------------------------------------------------------------------ */
    /* destroy each CUDA stream */
    /* ------------------------------------------------------------------ */

    for (k = 0 ; k < CHOLMOD_HOST_SUPERNODE_BUFFERS ; k++)
    {
        for (vdevice = device * Common->cuda_gpu_parallel; vdevice < (device + 1) * Common->cuda_gpu_parallel; vdevice++)
        if (Common->gpuStream[vdevice] [k])
        {
            /* fprintf (stderr, "destroy gpuStream[%d] [%d] %p\n", vdevice, k,
                Common->gpuStream[vdevice] [k]) ; */
            cudaStreamDestroy (Common->gpuStream[vdevice] [k]) ;
            /* fprintf (stderr, "destroy gpuStream[%d] [%d] done\n", vdevice, k) ; */
            Common->gpuStream[vdevice] [k] = NULL ;
        }
    }

    /* ------------------------------------------------------------------ */
    /* destroy each CUDA event */
    /* ------------------------------------------------------------------ */

    for (k = 0 ; k < 3 ; k++)
    {
        for (vdevice = device * Common->cuda_gpu_parallel; vdevice < (device + 1) * Common->cuda_gpu_parallel; vdevice++)
        if (Common->cublasEventPotrf[vdevice] [k])
        {
            /* fprintf (stderr, "destroy cublasEnventPotrf[%d] [%d] %p\n", vdevice, k,
                Common->cublasEventPotrf[vdevice] [k]) ; */
            cudaEventDestroy (Common->cublasEventPotrf[vdevice] [k]) ;
            /* fprintf (stderr, "destroy cublasEnventPotrf[%d] [%d] done\n", vdevice, k) ; */
            Common->cublasEventPotrf[vdevice] [k] = NULL ;
        }
    }
    }

    for (k = 0 ; k < CHOLMOD_HOST_SUPERNODE_BUFFERS ; k++)
    {
        for (vdevice = device * Common->cuda_gpu_parallel; vdevice < (device + 1) * Common->cuda_gpu_parallel; vdevice++)
        if (Common->updateCBuffersFree[vdevice] [k])
        {
            /* fprintf (stderr, "destroy updateCBuffersFree[%d] [%d] %p\n", vdevice, k,
                Common->updateCBuffersFree[vdevice] [k]) ; */
            cudaEventDestroy (Common->updateCBuffersFree[vdevice] [k]) ;
            /* fprintf (stderr, "destroy updateCBuffersFree[%d] [%d] done\n", vdevice, k) ;*/
            Common->updateCBuffersFree[vdevice] [k] = NULL ;
        }
    }

    if (Common->updateCKernelsComplete[device])
    {
        for (vdevice = device * Common->cuda_gpu_parallel; vdevice < (device + 1) * Common->cuda_gpu_parallel; vdevice++)
        {
            /* fprintf (stderr, "destroy updateCKernelsComplete[%d] %p\n", vdevice,
               Common->updateCKernelsComplete[vdevice]) ; */
            cudaEventDestroy (Common->updateCKernelsComplete[vdevice]) ;
            /* fprintf (stderr, "destroy updateCKernelsComplete[%d] done\n", vdevice) ; */
            Common->updateCKernelsComplete[vdevice] = NULL;
        }
    }
#endif
}


/* ========================================================================== */
/* === cholmod_gpu_allocate ================================================= */
/* ========================================================================== */
/*
 * Allocate both host and device memory needed for GPU computation.
 *
 * Memory allocation is expensive and should be done once and reused for
 * multiple factorizations.
 *
 * When gpu_allocate is called, the requested amount of device (and by
 * association host) memory is computed.  If that amount or more memory has
 * already been allocated, then nothing is done.  (i.e. memory allocation is
 * not reduced.)  If the requested amount is more than is currently allcoated
 * then both device and pinned host memory is freed and the new amount
 * allocated.
 *
 * This routine will allocate the minimum of either:
 *
 * maxGpuMemBytes - size of requested device allocation in bytes
 *
 * maxGpuMemFraction - size of requested device allocation as a fraction of
 *                     total GPU memory
 *
 * If both maxGpuMemBytes and maxGpuMemFraction are zero, this will allocate
 * the maximum amount of GPU memory possible.
 *
 * Note that since the GPU driver requires some memory, it is not advisable
 * to request maxGpuMemFraction of 1.0 (which will request all GPU memory and
 * will fail).  If maximum memory is requested then call this routine wtih
 * both maxGpuMemBytes and maxGpuMemFraction of 0.0.
 *
 */

int CHOLMOD(gpu_allocate) ( cholmod_common *Common, int device )
{

#ifdef GPU_BLAS

    int k;
    size_t fdm, tdm;
    size_t requestedDeviceMemory, requestedHostMemory;
    double tstart, tend;
    cudaError_t cudaErr;
    cublasStatus_t cublasErr;
    size_t maxGpuMemBytes;
    double maxGpuMemFraction;

    size_t devBuffSize;

    int vdevice;

    cudaSetDevice(device);

    /* fprintf (stderr, "gpu_allocate useGPU %d\n", Common->useGPU) ; */
    if (Common->useGPU != 1) return (0) ;

    maxGpuMemBytes = Common->maxGpuMemBytes;
    maxGpuMemFraction = Common->maxGpuMemFraction;

    /* ensure valid input */
    if ( maxGpuMemBytes < 0 ) maxGpuMemBytes = 0;
    if ( maxGpuMemFraction < 0 ) maxGpuMemFraction = 0;
    if ( maxGpuMemFraction > 1 ) maxGpuMemFraction = 1;

    int err = CHOLMOD(gpu_memorysize) (&tdm,&fdm,Common) ;
    if (err)
    {
        printf ("GPU failure in cholmod_gpu: gpu_memorysize %g %g MB\n",
            ((double) tdm) / (1024*1024),
            ((double) fdm) / (1024*1024)) ;
        ERROR (CHOLMOD_GPU_PROBLEM, "gpu memorysize failure\n") ;
    }

    /* compute the amount of device memory requested */
    if ( maxGpuMemBytes == 0 && maxGpuMemFraction == 0 ) {
        /* no specific request - take all available GPU memory
         *  (other programs could already have allocated some GPU memory,
         *  possibly even previous calls to gpu_allocate).  Always leave
         *  50 MB free for driver use. */
        requestedDeviceMemory = fdm+Common->dev_mempool_size[device]-
            1024ll*1024ll*50ll;
    }
    else if ( maxGpuMemBytes > 0 && maxGpuMemFraction > 0 ) {
        /* both byte and fraction limits - take the lowest of the two */
        requestedDeviceMemory = maxGpuMemBytes;
        if ( requestedDeviceMemory > tdm*maxGpuMemFraction ) {
            requestedDeviceMemory = tdm*maxGpuMemFraction;
        }
    }
    else if ( maxGpuMemFraction > 0 ) {
        /* just a fraction requested */
        requestedDeviceMemory = maxGpuMemFraction * tdm;
    }
    else {
        /* specific number of bytes requested */
        requestedDeviceMemory = maxGpuMemBytes;
        if ( maxGpuMemBytes > fdm ) {
            CHOLMOD_GPU_PRINTF ((
                "GPU WARNING: Requested amount of device memory not available\n"
                )) ;
            requestedDeviceMemory = fdm;
        }
    }

    /* do nothing if sufficient memory has already been allocated */
    if ( requestedDeviceMemory <= Common->dev_mempool_size[device] ) {

        CHOLMOD_GPU_PRINTF (("requested = %d, mempool = %d \n",
            requestedDeviceMemory, Common->dev_mempool_size[device]));
        CHOLMOD_GPU_PRINTF (("GPU NOTE:  gpu_allocate did nothing \n"));
        return 0;
    }

    CHOLMOD(gpu_deallocate) (Common, device);

    /* create cuBlas handle */
    for (vdevice = device * Common->cuda_gpu_parallel; vdevice < (device + 1) * Common->cuda_gpu_parallel; vdevice++)
    if ( ! Common->cublasHandle[vdevice] ) {
        cublasErr = cublasCreate (&(Common->cublasHandle[vdevice])) ;
        if (cublasErr != CUBLAS_STATUS_SUCCESS) {
            ERROR (CHOLMOD_GPU_PROBLEM, "CUBLAS initialization") ;
            return 1;
        }
    }

    /* allocated corresponding pinned host memory */
    requestedHostMemory = requestedDeviceMemory
        * (Common->cuda_gpu_parallel * CHOLMOD_HOST_SUPERNODE_BUFFERS)
        / (CHOLMOD_DEVICE_LS_BUFFERS + Common->cuda_gpu_parallel * CHOLMOD_DEVICE_SUPERNODE_BUFFERS);

    cudaErr = cudaMallocHost ( (void**)&(Common->host_pinned_mempool[device]),
                               requestedHostMemory );
    while ( cudaErr ) {
        /* insufficient host memory, try again with less */
        requestedHostMemory *= .5;
        cudaErr = cudaMallocHost ( (void**)&(Common->host_pinned_mempool[device]),
                                   requestedHostMemory );
    }
    Common->host_pinned_mempool_size[device] = requestedHostMemory;

    requestedDeviceMemory = requestedHostMemory
        * (CHOLMOD_DEVICE_LS_BUFFERS + Common->cuda_gpu_parallel * CHOLMOD_DEVICE_SUPERNODE_BUFFERS)
        / (Common->cuda_gpu_parallel * CHOLMOD_HOST_SUPERNODE_BUFFERS);

    /* Split up the memory allocations into required device buffers. */
    devBuffSize = requestedDeviceMemory
        / (CHOLMOD_DEVICE_LS_BUFFERS + Common->cuda_gpu_parallel * CHOLMOD_DEVICE_SUPERNODE_BUFFERS);
    devBuffSize -= devBuffSize%0x20000;
    if (Common->devBuffSize <= 0 || Common->devBuffSize > devBuffSize)
        Common->devBuffSize = devBuffSize;

    cudaErr = cudaMalloc ( &(Common->dev_mempool[device]), requestedDeviceMemory );
    /*
    CHOLMOD_HANDLE_CUDA_ERROR (cudaErr,"device memory allocation failure\n");
    */
    if (cudaErr)
    {
        printf ("GPU failure in cholmod_gpu: requested %g MB\n",
            ((double) requestedDeviceMemory) / (1024*1024)) ;
        ERROR (CHOLMOD_GPU_PROBLEM, "device memory allocation failure\n") ;
    }

    Common->dev_mempool_size[device] = requestedDeviceMemory;

#endif

    return (0);
}
