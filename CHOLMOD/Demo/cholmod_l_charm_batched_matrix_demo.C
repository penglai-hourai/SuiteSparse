/* ========================================================================== */
/* === Demo/cholmod_l_demo ================================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Demo Module.  Copyright (C) 2005-2013, Timothy A. Davis
 * -------------------------------------------------------------------------- */

/* Read in a matrix from a file, and use CHOLMOD to solve Ax=b if A is
 * symmetric, or (AA'+beta*I)x=b otherwise.  The file format is a simple
 * triplet format, compatible with most files in the Matrix Market format.
 * See cholmod_read.c for more details.  The readhb.f program reads a
 * Harwell/Boeing matrix (excluding element-types) and converts it into the
 * form needed by this program.  reade.f reads a matrix in Harwell/Boeing
 * finite-element form.
 *
 * Usage:
 *	cholmod_l_demo matrixfile
 *	cholmod_l_demo < matrixfile
 *
 * The matrix is assumed to be positive definite (a supernodal LL' or simplicial
 * LDL' factorization is used).
 *
 * Requires the Core, Cholesky, MatrixOps, and Check Modules.
 * Optionally uses the Partition and Supernodal Modules.
 * Does not use the Modify Module.
 *
 * See cholmod_simple.c for a simpler demo program.
 *
 * SuiteSparse_long is normally defined as long, except for WIN64.
 */

#include <cuda_runtime.h>
#include <omp.h>

#include "cholmod_demo.h"
#include "cholmod_l_charm_batched_matrix_demo.decl.h"

#define NTRIALS 100

CProxy_main mainProxy;
size_t cm_queue [CUDA_GPU_NUM];

int prefer_zomplex = 0;

class main : public CBase_main
{
    private:
        int argc;
        char **argv;
        int nfiles;
        int nGPUs;
        int device_mark [CUDA_GPU_NUM];
        omp_lock_t gpu_lock [CUDA_GPU_NUM];
        int file_mark [NMATRICES];
        double begin_time, end_time;
        CProxy_file_struct file_structs;

        cholmod_common Common_queue [CUDA_GPU_NUM];

    public:
        main (CkArgMsg *msg)
        {
            int ver [3] ;

            int k;

            argc = msg->argc;
            argv = msg->argv;

            mainProxy = thisProxy;

            nfiles = argc - 1;
            if (nfiles == 0)
                nfiles = 1;

            file_structs = CProxy_file_struct::ckNew(nfiles);

            for (k = 0; k < nfiles; k++)
            {
                file_mark[k] = FALSE;
            }

            /* ---------------------------------------------------------------------- */
            /* read in a matrix */
            /* ---------------------------------------------------------------------- */

            CkPrintf ("\n---------------------------------- cholmod_l_demo:\n") ;
            cholmod_l_version (ver) ;
            CkPrintf ("cholmod version %d.%d.%d\n", ver [0], ver [1], ver [2]) ;
            SuiteSparse_version (ver) ;
            CkPrintf ("SuiteSparse version %d.%d.%d\n", ver [0], ver [1], ver [2]) ;

            cudaGetDeviceCount(&nGPUs);
            if (nGPUs > CUDA_GPU_NUM)
                nGPUs = CUDA_GPU_NUM;

            for (k = 0; k < nGPUs; k++)
            {
                omp_init_lock (&gpu_lock[k]);
            }

            for (k = 0; k < nGPUs; k++)
            {
                device_mark[k] = FALSE;
            }

            initialize();

            begin_time = CPUTIME;
            CkPrintf ("---------------------------------- cholesky begin timestamp = %12.4lf:\n", begin_time);

            file_structs.cholesky(nGPUs, nfiles);
        }

        void initialize ()
        {
            int device;
            cholmod_common *cm;

            for (device = 0; device < nGPUs; device++)
            {
                ((cholmod_common**)cm_queue)[device] = &Common_queue[device];
                cm = ((cholmod_common**)cm_queue)[device];
                cm->pdev = device;

                CkPrintf ("================ device %d initialize begin\n", device);

                /* ---------------------------------------------------------------------- */
                /* start CHOLMOD and set parameters */
                /* ---------------------------------------------------------------------- */

                cholmod_l_start (cm) ;
                CHOLMOD_FUNCTION_DEFAULTS ;     /* just for testing (not required) */

                cm->pdev = device;

                /* cm->useGPU = 1; */
                cm->prefer_zomplex = prefer_zomplex ;

                /* use default parameter settings, except for the error handler.  This
                 * demo program terminates if an error occurs (out of memory, not positive
                 * definite, ...).  It makes the demo program simpler (no need to check
                 * CHOLMOD error conditions).  This non-default parameter setting has no
                 * effect on performance. */
                cm->error_handler = NULL;

                /* Note that CHOLMOD will do a supernodal LL' or a simplicial LDL' by
                 * default, automatically selecting the latter if flop/nnz(L) < 40. */

                cholmod_l_init_gpus (CHOLMOD_ANALYZE_FOR_CHOLESKY, cm);

                mark_device(device);

                CkPrintf ("================ device %d initialize end\n", device);
            }
        }

        void destroy ()
        {
            int device;
            cholmod_common *cm;

            for (device = 0; device < nGPUs; device++)
            {
                if (lock_gpu(device))
                {
                    cm = ((cholmod_common**)cm_queue)[device];

                    CkPrintf ("================ device %d free begin\n", device);

                    cholmod_l_finish (cm) ;

                    CkPrintf ("================ device %d free end\n", device);
                }
            }

            destroy_gpu_lock (device);

            exit_main();
        }

        void mark_device (int GPUindex)
        {
            device_mark[GPUindex] = TRUE;
        }

        int get_device_mark (int GPUindex)
        {
            return device_mark[GPUindex];
        }

        void mark_file (int findex)
        {
            file_mark[findex] = TRUE;
        }

        int get_file_mark (int findex)
        {
            return file_mark[findex];
        }

        int lock_gpu (int GPUindex)
        {
            return omp_test_lock(&gpu_lock[GPUindex]);
        }

        void unlock_gpu (int GPUindex)
        {
            omp_unset_lock(&gpu_lock[GPUindex]);
        }

        void destroy_gpu_lock (int GPUindex)
        {
            omp_destroy_lock(&gpu_lock[GPUindex]);
        }

        std::string get_filename (int findex)
        {
            std::string filename;

            if (argc == 1)
                filename = "";
            else
                filename = argv[findex+1];

            return filename;
        }

        void exit_main ()
        {
            end_time = CPUTIME;
            CkPrintf ("---------------------------------- cholesky end timestamp = %12.4lf:\n", end_time);
            CkPrintf ("total time = %12.4lf:\n", end_time - begin_time);
            CkExit();
        }
};

class file_struct : public CBase_file_struct
{
    private:
        int nGPUs;
        int findex;
        std::string filename;
        FILE *file;

        cholmod_factor *L ;

        CProxy_front_struct front_structs;

    public:
        file_struct ()
        {
        }

        file_struct (CkMigrateMessage *msg)
        {
        }

        void initialize (int nGPUs)
        {
            int k;

            this->nGPUs = nGPUs;

            for (k = 0; k < nGPUs; k++)
            {
                while (mainProxy.get_device_mark(k) == FALSE);
            }

            findex = thisIndex;
            filename = mainProxy.get_filename(findex);
            if (filename.empty())
                file = stdin;
            else
                file = fopen (filename.c_str(), "r");
        }

        void analyze ()
        {
        }

        void factorize ()
        {
            int GPUindex, selected;
            cholmod_common *cm;

            int device;

            int k;
            FILE *log;
            char logname[16];

            double resid [4], t, ta, tf, ts [3], tot, bnorm, xnorm, anorm, rnorm, fl,
                   anz, axbnorm, rnorm2, resid2, rcond ;
            cholmod_sparse *A ;
            cholmod_dense *X = NULL, *B, *W, *R = NULL ;
            double one [2], zero [2], minusone [2], beta [2], xlnz ;
            double *Bx, *Rx, *Xx, *Bz, *Xz, *Rz ;
            SuiteSparse_long i, n, isize, xsize, ordering, xtype, s, ss, lnz ;
            int trial, method, L_is_super ;
            int nmethods ;

            GPUindex = -1;
            selected = 0;
            while (!selected)
            {
                GPUindex = (GPUindex + 1) % nGPUs;
                selected = mainProxy.lock_gpu(GPUindex);
            }
            cm = ((cholmod_common**)cm_queue)[GPUindex];
            device = cm->pdev;

            memset (logname, 0, sizeof(char) * 16);

            CkPrintf ("================ device %d factorize begin\n", device);

            sprintf (logname, "log%04d.log", findex);
            log = fopen (logname, "w");

            CkPrintf ("================ device %d factorizes file %d: %s\n", device, findex, filename.c_str());

            if (file != NULL)
            {
                ts[0] = 0.;
                ts[1] = 0.;
                ts[2] = 0.;

                /* ---------------------------------------------------------------------- */
                /* create basic scalars */
                /* ---------------------------------------------------------------------- */

                zero [0] = 0 ;
                zero [1] = 0 ;
                one [0] = 1 ;
                one [1] = 0 ;
                minusone [0] = -1 ;
                minusone [1] = 0 ;
                beta [0] = 1e-6 ;
                beta [1] = 0 ;

                /* ---------------------------------------------------------------------- */
                /* get the file containing the input matrix */
                /* ---------------------------------------------------------------------- */

                A = cholmod_l_read_sparse (file, cm) ;
                anorm = 1 ;
#ifndef NMATRIXOPS
                anorm = cholmod_l_norm_sparse (A, 0, cm) ;
                fprintf (log, "norm (A,inf) = %g\n", anorm) ;
                fprintf (log, "norm (A,1)   = %g\n", cholmod_l_norm_sparse (A, 1, cm)) ;
#endif

                if (prefer_zomplex && A->xtype == CHOLMOD_COMPLEX)
                {
                    /* Convert to zomplex, just for testing.  In a zomplex matrix,
                       the real and imaginary parts are in separate arrays.  MATLAB
                       uses zomplex matrix exclusively. */
                    double *Ax = (double *) (A->x) ;
                    SuiteSparse_long nz = cholmod_l_nnz (A, cm) ;
                    fprintf (log, "nz: %ld\n", nz) ;
                    double *Ax2 = (double *) cholmod_l_malloc (nz, sizeof (double), cm) ;
                    double *Az2 = (double *) cholmod_l_malloc (nz, sizeof (double), cm) ;
                    for (i = 0 ; i < nz ; i++)
                    {
                        Ax2 [i] = Ax [2*i  ] ;
                        Az2 [i] = Ax [2*i+1] ;
                    }
                    cholmod_l_free (A->nzmax, 2*sizeof(double), Ax, cm) ;
                    A->x = Ax2 ;
                    A->z = Az2 ;
                    A->xtype = CHOLMOD_ZOMPLEX ;
                    /* cm->print = 5 ; */
                }

                xtype = A->xtype ;
                cholmod_l_print_sparse (A, "A", cm) ;

                if (A->nrow > A->ncol)
                {
                    /* Transpose A so that A'A+beta*I will be factorized instead */
                    cholmod_sparse *C = cholmod_l_transpose (A, 2, cm) ;
                    cholmod_l_free_sparse (&A, cm) ;
                    A = C ;
                    fprintf (log, "transposing input matrix\n") ;
                }

                /* ---------------------------------------------------------------------- */
                /* create an arbitrary right-hand-side */
                /* ---------------------------------------------------------------------- */

                n = A->nrow ;
                B = cholmod_l_zeros (n, 1, xtype, cm) ;
                Bx = (double *) (B->x) ;
                Bz = (double *) (B->z) ;

#if GHS
                {
                    /* b = A*ones(n,1), used by Gould, Hu, and Scott in their experiments */
                    cholmod_dense *X0 ;
                    X0 = cholmod_l_ones (A->ncol, 1, xtype, cm) ;
                    cholmod_l_sdmult (A, 0, one, zero, X0, B, cm) ;
                    cholmod_l_free_dense (&X0, cm) ;
                }
#else
                if (xtype == CHOLMOD_REAL)
                {
                    /* real case */
                    for (i = 0 ; i < n ; i++)
                    {
                        double x = n ;
                        Bx [i] = 1 + i / x ;
                    }
                }
                else if (xtype == CHOLMOD_COMPLEX)
                {
                    /* complex case */
                    for (i = 0 ; i < n ; i++)
                    {
                        double x = n ;
                        Bx [2*i  ] = 1 + i / x ;		/* real part of B(i) */
                        Bx [2*i+1] = (x/2 - i) / (3*x) ;	/* imag part of B(i) */
                    }
                }
                else /* (xtype == CHOLMOD_ZOMPLEX) */
                {
                    /* zomplex case */
                    for (i = 0 ; i < n ; i++)
                    {
                        double x = n ;
                        Bx [i] = 1 + i / x ;		/* real part of B(i) */
                        Bz [i] = (x/2 - i) / (3*x) ;	/* imag part of B(i) */
                    }
                }

#endif

                cholmod_l_print_dense (B, "B", cm) ;
                bnorm = 1 ;
#ifndef NMATRIXOPS
                bnorm = cholmod_l_norm_dense (B, 0, cm) ;	/* max norm */
                fprintf (log, "bnorm %g\n", bnorm) ;
#endif

                /* ---------------------------------------------------------------------- */
                /* analyze and factorize */
                /* ---------------------------------------------------------------------- */

                t = CPUTIME ;
                L = cholmod_l_analyze (A, cm) ;
                ta = CPUTIME - t ;
                ta = MAX (ta, 0) ;

                fprintf (log, "Analyze: flop %g lnz %g\n", cm->fl, cm->lnz) ;

                front_structs = CProxy_front_struct::ckNew(L->nsuper);

                front_structs.front_cholesky (nGPUs, (size_t) L);

                if (A->stype == 0)
                {
                    fprintf (log, "Factorizing A*A'+beta*I\n") ;
                    t = CPUTIME ;
                    cholmod_l_factorize_p (A, beta, NULL, 0, L, cm) ;
                    tf = CPUTIME - t ;
                    tf = MAX (tf, 0) ;
                }
                else
                {
                    fprintf (log, "Factorizing A\n") ;
                    t = CPUTIME ;
                    cholmod_l_factorize (A, L, cm) ;
                    tf = CPUTIME - t ;
                    tf = MAX (tf, 0) ;
                }

                cholmod_l_print_factor (L, "L", cm) ;

                /* determine the # of integers's and reals's in L.  See cholmod_free */
                if (L->is_super)
                {
                    s = L->nsuper + 1 ;
                    xsize = L->xsize ;
                    ss = L->ssize ;
                    isize =
                        n	/* L->Perm */
                        + n	/* L->ColCount, nz in each column of 'pure' L */
                        + s	/* L->pi, column pointers for L->s */
                        + s	/* L->px, column pointers for L->x */
                        + s	/* L->super, starting column index of each supernode */
                        + ss ;	/* L->s, the pattern of the supernodes */
                }
                else
                {
                    /* this space can increase if you change parameters to their non-
                     * default values (cm->final_pack, for example). */
                    lnz = L->nzmax ;
                    xsize = lnz ;
                    isize =
                        n	/* L->Perm */
                        + n	/* L->ColCount, nz in each column of 'pure' L */
                        + n+1	/* L->p, column pointers */
                        + lnz	/* L->i, integer row indices */
                        + n	/* L->nz, nz in each column of L */
                        + n+2	/* L->next, link list */
                        + n+2 ;	/* L->prev, link list */
                }

                /* solve with Bset will change L from simplicial to supernodal */
                rcond = cholmod_l_rcond (L, cm) ;
                L_is_super = L->is_super ;

                /* ---------------------------------------------------------------------- */
                /* solve */
                /* ---------------------------------------------------------------------- */

                if (n >= 1000)
                {
                    nmethods = 1 ;
                }
                else if (xtype == CHOLMOD_ZOMPLEX)
                {
                    nmethods = 2 ;
                }
                else
                {
                    nmethods = 3 ;
                }
                fprintf (log, "nmethods: %d\n", nmethods) ;

                for (method = 0 ; method <= nmethods ; method++)
                {
                    double x = n ;
                    resid [method] = -1 ;       /* not yet computed */

                    if (method == 0)
                    {
                        /* basic solve, just once */
                        t = CPUTIME ;
                        X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;
                        ts [0] = CPUTIME - t ;
                        ts [0] = MAX (ts [0], 0) ;
                    }
                    else if (method == 1)
                    {
                        /* basic solve, many times, but keep the last one */
                        t = CPUTIME ;
                        for (trial = 0 ; trial < NTRIALS ; trial++)
                        {
                            cholmod_l_free_dense (&X, cm) ;
                            Bx [0] = 1 + trial / x ;        /* tweak B each iteration */
                            X = cholmod_l_solve (CHOLMOD_A, L, B, cm) ;
                        }
                        ts [1] = CPUTIME - t ;
                        ts [1] = MAX (ts [1], 0) / NTRIALS ;
                    }
                    else if (method == 2)
                    {
                        /* solve with reused workspace */
                        cholmod_dense *Ywork = NULL, *Ework = NULL ;
                        cholmod_l_free_dense (&X, cm) ;

                        t = CPUTIME ;
                        for (trial = 0 ; trial < NTRIALS ; trial++)
                        {
                            Bx [0] = 1 + trial / x ;        /* tweak B each iteration */
                            cholmod_l_solve2 (CHOLMOD_A, L, B, NULL, &X, NULL,
                                    &Ywork, &Ework, cm) ;
                        }
                        cholmod_l_free_dense (&Ywork, cm) ;
                        cholmod_l_free_dense (&Ework, cm) ;
                        ts [2] = CPUTIME - t ;
                        ts [2] = MAX (ts [2], 0) / NTRIALS ;

                    }
                    else
                    {
                        /* solve with reused workspace and sparse Bset */
                        cholmod_dense *Ywork = NULL, *Ework = NULL ;
                        cholmod_dense *X2 = NULL, *B2 = NULL ;
                        cholmod_sparse *Bset, *Xset = NULL ;
                        SuiteSparse_long *Bsetp, *Bseti, *Xsetp, *Xseti, xlen, j, k, *Lnz ;
                        double *X1x, *X2x, *B2x, err ;
                        FILE *timelog = fopen ("timelog.m", "w") ;
                        if (timelog) fprintf (timelog, "results = [\n") ;

                        B2 = cholmod_l_zeros (n, 1, xtype, cm) ;
                        B2x = (double *) (B2->x) ;

                        Bset = cholmod_l_allocate_sparse (n, 1, 1, FALSE, TRUE, 0,
                                CHOLMOD_PATTERN, cm) ;
                        Bsetp = (SuiteSparse_long *) (Bset->p) ;
                        Bseti = (SuiteSparse_long *) (Bset->i) ;
                        Bsetp [0] = 0 ;     /* nnz(B) is 1 (it can be anything) */
                        Bsetp [1] = 1 ;
                        resid [3] = 0 ;

                        for (i = 0 ; i < MIN (100,n) ; i++)
                        {
                            /* B (i) is nonzero, all other entries are ignored
                               (implied to be zero) */
                            Bseti [0] = i ;
                            if (xtype == CHOLMOD_REAL)
                            {
                                B2x [i] = 3.1 * i + 0.9 ;
                            }
                            else /* (xtype == CHOLMOD_COMPLEX) */
                            {
                                B2x [2*i  ] = i + 0.042 ;
                                B2x [2*i+1] = i - 92.7 ;
                            }

                            /* first get the entire solution, to compare against */
                            cholmod_l_solve2 (CHOLMOD_A, L, B2, NULL, &X, NULL,
                                    &Ywork, &Ework, cm) ;

                            /* now get the sparse solutions; this will change L from
                               supernodal to simplicial */

                            if (i == 0)
                            {
                                /* first solve can be slower because it has to allocate
                                   space for X2, Xset, etc, and change L.
                                   So don't time it */
                                cholmod_l_solve2 (CHOLMOD_A, L, B2, Bset, &X2, &Xset,
                                        &Ywork, &Ework, cm) ;
                            }

                            t = CPUTIME ;
                            for (trial = 0 ; trial < NTRIALS ; trial++)
                            {
                                /* solve Ax=b but only to get x(i).
                                   b is all zero except for b(i).
                                   This takes O(xlen) time */
                                cholmod_l_solve2 (CHOLMOD_A, L, B2, Bset, &X2, &Xset,
                                        &Ywork, &Ework, cm) ;
                            }
                            t = CPUTIME - t ;
                            t = MAX (t, 0) / NTRIALS ;

                            /* check the solution and log the time */
                            Xsetp = (SuiteSparse_long *) (Xset->p) ;
                            Xseti = (SuiteSparse_long *) (Xset->i) ;
                            xlen = Xsetp [1] ;
                            X1x = (double *) (X->x) ;
                            X2x = (double *) (X2->x) ;
                            Lnz = (SuiteSparse_long *) (L->nz) ;

                            if (xtype == CHOLMOD_REAL)
                            {
                                fl = 2 * xlen ;
                                for (k = 0 ; k < xlen ; k++)
                                {
                                    j = Xseti [k] ;
                                    fl += 4 * Lnz [j] ;
                                    err = X1x [j] - X2x [j] ;
                                    err = ABS (err) ;
                                    resid [3] = MAX (resid [3], err) ;
                                }
                            }
                            else /* (xtype == CHOLMOD_COMPLEX) */
                            {
                                fl = 16 * xlen ;
                                for (k = 0 ; k < xlen ; k++)
                                {
                                    j = Xseti [k] ;
                                    fl += 16 * Lnz [j] ;
                                    err = X1x [2*j  ] - X2x [2*j  ] ;
                                    err = ABS (err) ;
                                    resid [3] = MAX (resid [3], err) ;
                                    err = X1x [2*j+1] - X2x [2*j+1] ;
                                    err = ABS (err) ;
                                    resid [3] = MAX (resid [3], err) ;
                                }
                            }

                            if (timelog) fprintf (timelog, "%g %g %g %g\n",
                                    (double) i, (double) xlen, fl, t);

                            /* clear B for the next test */
                            if (xtype == CHOLMOD_REAL)
                            {
                                B2x [i] = 0 ;
                            }
                            else /* (xtype == CHOLMOD_COMPLEX) */
                            {
                                B2x [2*i  ] = 0 ;
                                B2x [2*i+1] = 0 ;
                            }
                        }

                        if (timelog)
                        {
                            fprintf (timelog, "] ; resid = %g ;\n", resid [3]) ;
                            fprintf (timelog, "lnz = %g ;\n", cm->lnz) ;
                            fprintf (timelog, "t = %g ;   %% dense solve time\n", ts [2]) ;
                            fclose (timelog) ;
                        }

#ifndef NMATRIXOPS
                        resid [3] = resid [3] / cholmod_l_norm_dense (X, 1, cm) ;
#endif

                        cholmod_l_free_dense (&Ywork, cm) ;
                        cholmod_l_free_dense (&Ework, cm) ;
                        cholmod_l_free_dense (&X2, cm) ;
                        cholmod_l_free_dense (&B2, cm) ;
                        cholmod_l_free_sparse (&Xset, cm) ;
                        cholmod_l_free_sparse (&Bset, cm) ;
                    }

                    /* ------------------------------------------------------------------ */
                    /* compute the residual */
                    /* ------------------------------------------------------------------ */

                    if (method < 3)
                    {
#ifndef NMATRIXOPS
                        if (A->stype == 0)
                        {
                            /* (AA'+beta*I)x=b is the linear system that was solved */
                            /* W = A'*X */
                            W = cholmod_l_allocate_dense (A->ncol, 1, A->ncol, xtype, cm) ;
                            cholmod_l_sdmult (A, 2, one, zero, X, W, cm) ;
                            /* R = B - beta*X */
                            cholmod_l_free_dense (&R, cm) ;
                            R = cholmod_l_zeros (n, 1, xtype, cm) ;
                            Rx = (double *) (R->x) ;
                            Rz = (double *) (R->z) ;
                            Xx = (double *) (X->x) ;
                            Xz = (double *) (X->z) ;
                            if (xtype == CHOLMOD_REAL)
                            {
                                for (i = 0 ; i < n ; i++)
                                {
                                    Rx [i] = Bx [i] - beta [0] * Xx [i] ;
                                }
                            }
                            else if (xtype == CHOLMOD_COMPLEX)
                            {
                                /* complex case */
                                for (i = 0 ; i < n ; i++)
                                {
                                    Rx [2*i  ] = Bx [2*i  ] - beta [0] * Xx [2*i  ] ;
                                    Rx [2*i+1] = Bx [2*i+1] - beta [1] * Xx [2*i+1] ;
                                }
                            }
                            else /* (xtype == CHOLMOD_ZOMPLEX) */
                            {
                                /* zomplex case */
                                for (i = 0 ; i < n ; i++)
                                {
                                    Rx [i] = Bx [i] - beta [0] * Xx [i] ;
                                    Rz [i] = Bz [i] - beta [1] * Xz [i] ;
                                }
                            }

                            /* R = A*W - R */
                            cholmod_l_sdmult (A, 0, one, minusone, W, R, cm) ;
                            cholmod_l_free_dense (&W, cm) ;
                        }
                        else
                        {
                            /* Ax=b was factorized and solved, R = B-A*X */
                            cholmod_l_free_dense (&R, cm) ;
                            R = cholmod_l_copy_dense (B, cm) ;
                            cholmod_l_sdmult (A, 0, minusone, one, X, R, cm) ;
                        }
                        rnorm = cholmod_l_norm_dense (R, 0, cm) ;	    /* max abs. entry */
                        xnorm = cholmod_l_norm_dense (X, 0, cm) ;	    /* max abs. entry */

                        axbnorm = (anorm * xnorm + bnorm + ((n == 0) ? 1 : 0)) ;
                        resid [method] = rnorm / axbnorm ;
#else
                        fprintf (log, "residual not computed (requires CHOLMOD/MatrixOps)\n") ;
#endif
                    }
                }

                tot = ta + tf + ts [0] ;

                /* ---------------------------------------------------------------------- */
                /* iterative refinement (real symmetric case only) */
                /* ---------------------------------------------------------------------- */

                resid2 = -1 ;
#ifndef NMATRIXOPS
                if (A->stype != 0 && A->xtype == CHOLMOD_REAL)
                {
                    cholmod_dense *R2 ;

                    /* R2 = A\(B-A*X) */
                    R2 = cholmod_l_solve (CHOLMOD_A, L, R, cm) ;
                    /* compute X = X + A\(B-A*X) */
                    Xx = (double *) (X->x) ;
                    Rx = (double *) (R2->x) ;
                    for (i = 0 ; i < n ; i++)
                    {
                        Xx [i] = Xx [i] + Rx [i] ;
                    }
                    cholmod_l_free_dense (&R2, cm) ;
                    cholmod_l_free_dense (&R, cm) ;

                    /* compute the new residual, R = B-A*X */
                    cholmod_l_free_dense (&R, cm) ;
                    R = cholmod_l_copy_dense (B, cm) ;
                    cholmod_l_sdmult (A, 0, minusone, one, X, R, cm) ;
                    rnorm2 = cholmod_l_norm_dense (R, 0, cm) ;
                    resid2 = rnorm2 / axbnorm ;
                }
#endif

                cholmod_l_free_dense (&R, cm) ;

                /* ---------------------------------------------------------------------- */
                /* print results */
                /* ---------------------------------------------------------------------- */

                anz = cm->anz ;
                for (i = 0 ; i < CHOLMOD_MAXMETHODS ; i++)
                    /* for (i = 4 ; i < 3 ; i++) */
                {
                    fl = cm->method [i].fl ;
                    xlnz = cm->method [i].lnz ;
                    cm->method [i].fl = -1 ;
                    cm->method [i].lnz = -1 ;
                    ordering = cm->method [i].ordering ;
                    if (fl >= 0)
                    {
                        fprintf (log, "Ordering: ") ;
                        if (ordering == CHOLMOD_POSTORDERED) fprintf (log, "postordered ") ;
                        if (ordering == CHOLMOD_NATURAL)     fprintf (log, "natural ") ;
                        if (ordering == CHOLMOD_GIVEN)	 fprintf (log, "user    ") ;
                        if (ordering == CHOLMOD_AMD)	 fprintf (log, "AMD     ") ;
                        if (ordering == CHOLMOD_METIS)	 fprintf (log, "METIS   ") ;
                        if (ordering == CHOLMOD_NESDIS)      fprintf (log, "NESDIS  ") ;
                        if (xlnz > 0)
                        {
                            fprintf (log, "fl/lnz %10.1f", fl / xlnz) ;
                        }
                        if (anz > 0)
                        {
                            fprintf (log, "  lnz/anz %10.1f", xlnz / anz) ;
                        }
                        fprintf (log, "\n") ;
                    }
                }

                fprintf (log, "ints in L: %15.0f, doubles in L: %15.0f\n",
                        (double) isize, (double) xsize) ;
                fprintf (log, "factor flops %g nnz(L) %15.0f (w/no amalgamation)\n",
                        cm->fl, cm->lnz) ;
                if (A->stype == 0)
                {
                    fprintf (log, "nnz(A):    %15.0f\n", cm->anz) ;
                }
                else
                {
                    fprintf (log, "nnz(A*A'): %15.0f\n", cm->anz) ;
                }
                if (cm->lnz > 0)
                {
                    fprintf (log, "flops / nnz(L):  %8.1f\n", cm->fl / cm->lnz) ;
                }
                if (anz > 0)
                {
                    fprintf (log, "nnz(L) / nnz(A): %8.1f\n", cm->lnz / cm->anz) ;
                }
                fprintf (log, "analyze cputime:  %12.4f\n", ta) ;
                fprintf (log, "factor  cputime:   %12.4f mflop: %8.1f\n", tf,
                        (tf == 0) ? 0 : (1e-6*cm->fl / tf)) ;
                fprintf (log, "solve   cputime:   %12.4f mflop: %8.1f\n", ts [0],
                        (ts [0] == 0) ? 0 : (1e-6*4*cm->lnz / ts [0])) ;
                fprintf (log, "overall cputime:   %12.4f mflop: %8.1f\n", 
                        tot, (tot == 0) ? 0 : (1e-6 * (cm->fl + 4 * cm->lnz) / tot)) ;
                fprintf (log, "solve   cputime:   %12.4f mflop: %8.1f (%d trials)\n", ts [1],
                        (ts [1] == 0) ? 0 : (1e-6*4*cm->lnz / ts [1]), NTRIALS) ;
                fprintf (log, "solve2  cputime:   %12.4f mflop: %8.1f (%d trials)\n", ts [2],
                        (ts [2] == 0) ? 0 : (1e-6*4*cm->lnz / ts [2]), NTRIALS) ;
                fprintf (log, "peak memory usage: %12.0f (MB)\n",
                        (double) (cm->memory_usage) / 1048576.) ;
                fprintf (log, "residual (|Ax-b|/(|A||x|+|b|)): ") ;
                for (method = 0 ; method <= nmethods ; method++)
                {
                    fprintf (log, "%8.2e ", resid [method]) ;
                }
                fprintf (log, "\n") ;
                if (resid2 >= 0)
                {
                    fprintf (log, "residual %8.1e (|Ax-b|/(|A||x|+|b|))"
                            " after iterative refinement\n", resid2) ;
                }
                fprintf (log, "rcond    %8.1e\n\n", rcond) ;

                if (L_is_super)
                {
                    cholmod_l_gpu_stats (cm) ;
                }

                cholmod_l_free_factor (&L, cm) ;
                cholmod_l_free_dense (&X, cm) ;

                /* ---------------------------------------------------------------------- */
                /* free matrices and finish CHOLMOD */
                /* ---------------------------------------------------------------------- */

                cholmod_l_free_sparse (&A, cm) ;
                cholmod_l_free_dense (&B, cm) ;
            }

            if (log) fclose (log);

            mainProxy.mark_file(findex);

            mainProxy.unlock_gpu(device);

            CkPrintf ("================ device %d factorize end\n", device);
        }

        void destroy (int nfiles)
        {
            int findex;

            if (!filename.empty())
                fclose (file);

            for (findex = 0; findex < nfiles; findex++)
            {
                while (mainProxy.get_file_mark(findex) == FALSE);
            }

            mainProxy.destroy();
        }

        void cholesky (int nGPUs, int nfiles)
        {
            initialize(nGPUs);
            analyze();
            factorize();
            destroy(nfiles);
        }
};

class front_struct : public CBase_front_struct
{
    private:
        int nGPUs;
        cholmod_factor *L;

    public:
        front_struct ()
        {
        }

        void initialize (int nGPUs, cholmod_factor *L)
        {
            this->nGPUs = nGPUs;
            this->L = (cholmod_factor *) L;
        }

        void front_cholesky (int nGPUs, size_t L)
        {
        }
};

#include "cholmod_l_charm_batched_matrix_demo.def.h"
