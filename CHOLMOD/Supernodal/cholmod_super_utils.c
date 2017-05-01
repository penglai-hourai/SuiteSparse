#ifndef NGPL
#ifndef NSUPERNODAL

#include <cuda_runtime.h>
#include "cholmod_internal.h"
#include "cholmod_supernodal.h"
#include "cholmod_gpu.h"
#include "cholmod_super_utils.h"

void CHOLMOD (qSort) (Int *key, Int *value, Int low, Int high)
{
    Int l, m, h, tmp;

    if (low >= high)
        return;

    l = low;
    for (h = low + 1; h <= high && key[h] <= key[l]; h++);
    m = h - 1;

    while (h <= high)
    {
        if (key[h] <= key[l])
        {
            m++;
            tmp = key[m];
            key[m] = key[h];
            key[h] = tmp;
            tmp = value[m];
            value[m] = value[h];
            value[h] = tmp;
        }
        h++;
    }

    if (low < m)
    {
        tmp = key[low];
        key[low] = key[m];
        key[m] = tmp;
        tmp = value[low];
        value[low] = value[m];
        value[m] = tmp;
        CHOLMOD (qSort) (key, value, low, m - 1);
    }
    if (m < high)
        CHOLMOD (qSort) (key, value, m + 1, high);

    return;
}

void CHOLMOD (qRevSort) (Int *key, Int *value, Int low, Int high)
{
    Int l, m, h, tmp;

    if (low >= high)
        return;

    l = low;
    for (h = low + 1; h <= high && key[h] >= key[l]; h++);
    m = h - 1;

    while (h <= high)
    {
        if (key[h] >= key[l])
        {
            m++;
            tmp = key[m];
            key[m] = key[h];
            key[h] = tmp;
            tmp = value[m];
            value[m] = value[h];
            value[h] = tmp;
        }
        h++;
    }

    if (low < m)
    {
        tmp = key[low];
        key[low] = key[m];
        key[m] = tmp;
        tmp = value[low];
        value[low] = value[m];
        value[m] = tmp;
        CHOLMOD (qRevSort) (key, value, low, m - 1);
    }
    if (m < high)
        CHOLMOD (qRevSort) (key, value, m + 1, high);

    return;
}

void CHOLMOD (init_gpus) (int for_whom, cholmod_common *Common)
{
    const int pdev = Common->pdev;

    const char *env_use_gpu, *env_hybrid_gpu, *env_num_gpu, *env_max_bytes, *env_max_fraction;
    size_t max_bytes;
    double max_fraction;
#ifdef SUITESPARSE_CUDA
    int device, dev_l, dev_h;
#endif

    double timestamp;

    /* ---------------------------------------------------------------------- */
    /* allocate GPU workspace */
    /* ---------------------------------------------------------------------- */

#ifndef SUITESPARSE_CUDA
    /* GPU module is not installed */
    Common->useGPU = 0;
    Common->cuda_gpu_num = 0;
    Common->useHybrid = 0;
#endif

#ifndef DLONG
    /* GPU module supported only for long int */
    Common->useGPU = 0;
    Common->cuda_gpu_num = 0;
    Common->useHybrid = 0;
#endif

#ifdef SUITESPARSE_CUDA
    /* GPU module is installed */
    if ( for_whom == CHOLMOD_ANALYZE_FOR_CHOLESKY )
    {
        /* only allocate GPU workspace for supernodal Cholesky, and only when
           the GPU is requested and available. */

        max_bytes = 0;
        max_fraction = 0;

#ifdef DLONG
        if ( Common->useGPU == EMPTY )
        {
            /* useGPU not explicity requested by the user, but not explicitly
             * prohibited either.  Query OS environment variables for request.*/
            env_use_gpu  = getenv("CHOLMOD_USE_GPU");

            /* CHOLMOD_USE_GPU environment variable is set */
            if ( env_use_gpu )
            {
                if ( atoi ( env_use_gpu ) == 0 )
                {
                    Common->useGPU = 0; /* don't use the gpu */
                }
                else
                {
                    Common->useGPU = 1; /* use the gpu */
                    /* Query OS environment variables for request.*/
                    env_hybrid_gpu  = getenv("CHOLMOD_GPU_HYBRID");
                    /* CHOLMOD_GPU_HYBRID environment variable is set */
                    if ( env_hybrid_gpu )
                    {
                        if ( atoi ( env_hybrid_gpu ) == 0 )
                        {
                            Common->useHybrid = 0;                           	/* don't use hybrid (GPU only) */
                        }
                        else
                        {
                            Common->useHybrid = 1; 				/* use hybrid (GPU + CPU) */
                        }
                    }
                    /* CHOLMOD_GPU_HYBRID environment variable not set */
                    else
                    {
                        Common->useHybrid = 0;                              	/* default, hybrid is not defined */
                    }
                    env_num_gpu  = getenv("CHOLMOD_NUM_GPUS");
                    if ( env_num_gpu )
                    {	    
                        Common->cuda_gpu_num = atoi ( env_num_gpu );              	/* set # GPUs */	
                    }
                    /* CHOLMOD_USE_GPU environment variable not set */
                    else
                    {
                        Common->cuda_gpu_num = 0;				    	/* default, #GPUs is not defined */
                    }	        	
                    env_max_bytes = getenv("CHOLMOD_GPU_MEM_BYTES");
                    if ( env_max_bytes )
                    {
                        max_bytes = atol(env_max_bytes);
                        Common->maxGpuMemBytes = max_bytes;
                    }
                    env_max_fraction = getenv("CHOLMOD_GPU_MEM_FRACTION");
                    if ( env_max_fraction )
                    {
                        max_fraction = atof (env_max_fraction);
                        if ( max_fraction < 0 ) max_fraction = 0;
                        if ( max_fraction > 1 ) max_fraction = 1;
                        Common->maxGpuMemFraction = max_fraction;
                    }	
                }
            }
            else
            {
                /* CHOLMOD_USE_GPU environment variable not set, so no GPU
                 * acceleration will be used */
                Common->useGPU = 0;
            }
            /* fprintf (stderr, "useGPU queried: %d\n", Common->useGPU) ; */
        }

        if ( !Common->useGPU || Common->partialFactorization )
        {
            Common->useGPU = 0;
            Common->useHybrid = 0;
        }

        /* Ensure that a GPU is present */
        if ( Common->useGPU == 1 )
        {
            CHOLMOD_HANDLE_CUDA_ERROR (cudaGetDeviceCount(&(Common->cuda_gpu_num)), "cudaGetDeviceCount error");
            if (Common->cuda_gpu_num > CUDA_GPU_NUM)
                Common->cuda_gpu_num = CUDA_GPU_NUM;
        }
        else
        {
            Common->useGPU = 0;
            Common->cuda_gpu_num = 0;
        }
#else
        /* GPU acceleration is only supported for long int version */
        Common->useGPU = 0;
        Common->cuda_gpu_num = 0;
#endif

#ifdef DLONG
        timestamp = SuiteSparse_time();
        if ( Common->useGPU == 1 )
        {
            /* fprintf (stderr, "\nprobe GPU:\n") ; */
#if 0
            Common->useGPU = CHOLMOD(gpu_probe) (Common);
#endif
            /* fprintf (stderr, "\nprobe GPU: result %d\n", Common->useGPU) ; */

            if (pdev < 0)
            {
                dev_l = 0;
                dev_h = Common->cuda_gpu_num;
            }
            else
            {
                dev_l = pdev;
                dev_h = dev_l + 1;
            }
            /* Cholesky + GPU, so allocate space */
            for (device = dev_l; device < dev_h; device++)
            {
                /* fprintf (stderr, "allocate GPU:\n") ; */
                CHOLMOD(gpu_allocate) ( Common, device );
                /* fprintf (stderr, "allocate GPU done\n") ; */
            }
        }
        printf ("GPU memory allocation time = %lf\n", SuiteSparse_time() - timestamp);
    }
#endif

    Common->cuda_gpu_parallel = CUDA_GPU_PARALLEL;

    /*
    if (Common->cuda_gpu_num > 0)
    {
        Common->cuda_gpu_parallel = (L->nleaves - 1) / Common->cuda_gpu_num + 1;
        if (Common->cuda_gpu_parallel > CUDA_GPU_PARALLEL)
            Common->cuda_gpu_parallel = CUDA_GPU_PARALLEL;
        else if (Common->cuda_gpu_parallel <= 0)
            Common->cuda_gpu_parallel = 1;
    }
    else
    */
        Common->cuda_vgpu_num = Common->cuda_gpu_num * Common->cuda_gpu_parallel;

    if (Common->cuda_vgpu_num > 0)
        Common->cholmod_parallel_num_threads = Common->cuda_vgpu_num;
    else
        Common->cholmod_parallel_num_threads = CPU_THREAD_NUM;
#else
    /* GPU module is not installed */
    Common->useGPU = 0 ;
#endif
}

#endif
#endif
