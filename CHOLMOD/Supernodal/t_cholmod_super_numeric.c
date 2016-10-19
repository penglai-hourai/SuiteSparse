/* ========================================================================== */
/* === Supernodal/t_cholmod_super_numeric =================================== */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Supernodal Module.  Copyright (C) 2005-2012, Timothy A. Davis
 * http://www.suitesparse.com
 * -------------------------------------------------------------------------- */

/* Template routine for cholmod_super_numeric.  All xtypes supported, except
 * that a zomplex A and F result in a complex L (there is no supernodal
 * zomplex L).
 */

/* ========================================================================== */
/* === complex arithmetic =================================================== */
/* ========================================================================== */

#include <semaphore.h>
#include <pthread.h>
#include "cholmod_template.h"

#undef L_ENTRY
#undef L_CLEAR
#undef L_ASSIGN
#undef L_MULTADD
#undef L_ASSEMBLE
#undef L_ASSEMBLESUB

#ifdef REAL

/* -------------------------------------------------------------------------- */
/* A, F, and L are all real */
/* -------------------------------------------------------------------------- */

#define L_ENTRY 1
#define L_CLEAR(Lx,p)               Lx [p] = 0
#define L_ASSIGN(Lx,q, Ax,Az,p)     Lx [q] = Ax [p]
#define L_MULTADD(Lx,q, Ax,Az,p, f) Lx [q] += Ax [p] * f [0]
#define L_ASSEMBLE(Lx,q,b)          Lx [q] += b [0]
#define L_ASSEMBLESUB(Lx,q,C,p)     Lx [q] -= C [p]

#else

/* -------------------------------------------------------------------------- */
/* A and F are complex or zomplex, L and C are complex */
/* -------------------------------------------------------------------------- */

#define L_ENTRY 2
#define L_CLEAR(Lx,p)               Lx [2*(p)] = 0 ; Lx [2*(p)+1] = 0
#define L_ASSEMBLE(Lx,q,b)          Lx [2*(q)] += b [0] ;
#define L_ASSEMBLESUB(Lx,q,C,p)                 \
    Lx [2*(q)  ] -= C [2*(p)  ] ;               \
    Lx [2*(q)+1] -= C [2*(p)+1] ;

#ifdef COMPLEX

/* -------------------------------------------------------------------------- */
/* A, F, L, and C are all complex */
/* -------------------------------------------------------------------------- */

#define L_ASSIGN(Lx,q, Ax,Az,p)                 \
    Lx [2*(q)  ] = Ax [2*(p)  ] ;               \
    Lx [2*(q)+1] = Ax [2*(p)+1]

#define L_MULTADD(Lx,q, Ax,Az,p, f)                                     \
    Lx [2*(q)  ] += Ax [2*(p)  ] * f [0] - Ax [2*(p)+1] * f [1] ;       \
    Lx [2*(q)+1] += Ax [2*(p)+1] * f [0] + Ax [2*(p)  ] * f [1]

#else

/* -------------------------------------------------------------------------- */
/* A and F are zomplex, L and C is complex */
/* -------------------------------------------------------------------------- */

#define L_ASSIGN(Lx,q, Ax,Az,p)                 \
    Lx [2*(q)  ] = Ax [p] ;                     \
    Lx [2*(q)+1] = Az [p] ;

#define L_MULTADD(Lx,q, Ax,Az,p, f)                     \
    Lx [2*(q)  ] += Ax [p] * f [0] - Az [p] * f [1] ;   \
    Lx [2*(q)+1] += Az [p] * f [0] + Ax [p] * f [1]

#endif
#endif

typedef struct
{
    cholmod_sparse *A;
    cholmod_sparse *F;
    double *zero;
    double *one;
    double *beta;
    cholmod_factor *L;
    cholmod_dense *Cwork;
    cholmod_common *Common;
    Int *Map;
    Int *RelativeMap;
    double *C;
#ifdef GPU_BLAS
    int *GPUslot_p;
    int useGPU;
    cholmod_gpu_pointers *gpu_p;
#endif
    sem_t *thread_semaphore_p;
    pthread_mutex_t *thread_mutex_p;
    Int s;
    Int nscol_new;
    Int repeat_supernode;
    int *to_return_p;
} TEMPLATE (CHOLMOD (thread_args));

static void * TEMPLATE (cholmod_super_numeric_pthread) (void *void_args)
{
    TEMPLATE (CHOLMOD (thread_args)) *thread_args;

    sem_t *thread_semaphore_p;
    pthread_mutex_t *thread_mutex_p;

    cholmod_sparse *A, *F;
    double *zero, *one, *beta;
    cholmod_factor *L;
    cholmod_dense *Cwork;
    cholmod_common *Common;

    Int repeat_supernode;

    Int *Ap, *Ai, *Anz, *Az, Apacked, stype;
    Int *Fp, *Fi, *Fnz, *Fz, Fpacked;
    Int *Ls, n;
    Int *Super, *Lpi, *Lpx, nsuper;
    double *Ax, *Fx, *Lx;
    double *C;

    Int *Iwork, *SuperMap, *Next, *Lpos, *Next_save, *Lpos_save, *Previous, *pending, *front_col;
    Int *Head, *Map, *RelativeMap;

    Int s, ss, sparent;
    Int k1, k2, nscol, nscol2, nscol_new, psi, psend, nsrow, nsrow2, ndrow3, psx, pend, px, p, pk, q;
    Int pf, pfend;
    Int d, dnext, dancestor;
    Int kd1, kd2, pdi, pdend, ndcol, pdi1, pdi2, ndrow, ndrow1, ndrow2, pdx, pdx1;
    Int i, j, k, imap, tail;

    Int info = 0;

    double tstart;

    /* ---------------------------------------------------------------------- */
    /* declarations for the GPU */
    /* ---------------------------------------------------------------------- */

    /* these variables are not used if the GPU module is not installed */

#ifdef GPU_BLAS
    Int ndescendants, mapCreatedOnGpu, supernodeUsedGPU,
        idescendant, dlarge, dsmall, skips ;
    int iHostBuff, iDevBuff, useGPU, GPUavailable ;
    int *GPUslot_p;
    cholmod_gpu_pointers *gpu_p ;
#endif

    thread_args = void_args;

    thread_semaphore_p = thread_args->thread_semaphore_p;
    thread_mutex_p = thread_args->thread_mutex_p;

    A = thread_args->A;
    F = thread_args->F;
    zero = thread_args->zero;
    one = thread_args->one;
    beta = thread_args->beta;
    L = thread_args->L;
    Cwork = thread_args->Cwork;
    Common = thread_args->Common;
    Map = thread_args->Map;
    RelativeMap = thread_args->RelativeMap;
    C = thread_args->C;
#ifdef GPU_BLAS
    GPUslot_p = thread_args->GPUslot_p;
    useGPU = thread_args->useGPU;
    gpu_p = thread_args->gpu_p;
#endif
    s = thread_args->s;
    nscol_new = thread_args->nscol_new;
    repeat_supernode = thread_args->repeat_supernode;

    n = L->n;
    nsuper = L->nsuper;

    Ls = L->s ;
    Lpi = L->pi ;
    Lpx = L->px ;

    Super = L->super ;

    Lx = L->x ;

    stype = A->stype ;

    Ap = A->p ;
    Ai = A->i ;
    Ax = A->x ;
    Az = A->z ;
    Anz = A->nz ;
    Apacked = A->packed ;

    if (stype != 0)
    {
        /* F not accessed */
        Fp = NULL ;
        Fi = NULL ;
        Fx = NULL ;
        Fz = NULL ;
        Fnz = NULL ;
        Fpacked = TRUE ;
    }
    else
    {
        Fp = F->p ;
        Fi = F->i ;
        Fx = F->x ;
        Fz = F->z ;
        Fnz = F->nz ;
        Fpacked = F->packed ;
    }

    //C = Cwork->x ;      /* workspace of size L->maxcsize */

    /* allocate integer workspace */
    Iwork = Common->Iwork ;
    SuperMap    = Iwork ;                                   /* size n (i/i/l) */
    //RelativeMap = Iwork + n ;                               /* size n (i/i/l) */
    Next        = Iwork + 2*((size_t) n) ;                  /* size nsuper*/
    Lpos        = Iwork + 2*((size_t) n) + nsuper ;         /* size nsuper*/
    Next_save   = Iwork + 2*((size_t) n) + 2*((size_t) nsuper) ;/* size nsuper*/
    Lpos_save   = Iwork + 2*((size_t) n) + 3*((size_t) nsuper) ;/* size nsuper*/
    Previous    = Iwork + 2*((size_t) n) + 4*((size_t) nsuper) ;/* size nsuper*/
    pending     = Iwork + 2*((size_t) n) + 5*((size_t) nsuper) ;/* size nsuper*/
    front_col   = Iwork + 2*((size_t) n) + 6*((size_t) nsuper) ;/* size nsuper*/

    //Map  = Common->Flag ;   /* size n, use Flag as workspace for Map array */
    Head = Common->Head ;   /* size n+1, only Head [0..nsuper-1] used */

    /* clear the Map so that changes in the pattern of A can be detected */

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS) \
    if ( n > 128 ) schedule (static)

    for (i = 0 ; i < n ; i++)
    {
        Map [i] = EMPTY ;
    }

        /* ------------------------------------------------------------------ */
        /* get the size of supernode s */
        /* ------------------------------------------------------------------ */

        k1 = Super [s] ;            /* s contains columns k1 to k2-1 of L */
        k2 = Super [s+1] ;
        nscol = k2 - k1 ;           /* # of columns in all of s */
        psi = Lpi [s] ;             /* pointer to first row of s in Ls */
        psx = Lpx [s] ;             /* pointer to first row of s in Lx */
        psend = Lpi [s+1] ;         /* pointer just past last row of s in Ls */
        nsrow = psend - psi ;       /* # of rows in all of s */

        PRINT1 (("====================================================\n"
                 "S "ID" k1 "ID" k2 "ID" nsrow "ID" nscol "ID" psi "ID" psend "
                 ""ID" psx "ID"\n", s, k1, k2, nsrow, nscol, psi, psend, psx)) ;
        /* ------------------------------------------------------------------ */
        /* zero the supernode s */
        /* ------------------------------------------------------------------ */

        ASSERT ((size_t) (psx + nsrow*nscol) <= L->xsize) ;

        pend = psx + nsrow * nscol ;        /* s is nsrow-by-nscol */

#ifdef GPU_BLAS
        if ( !useGPU )
#endif
        {
            /* Case of no GPU, zero individual supernodes */

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    schedule (static) if ( pend - psx > 1024 )

            for (p = psx ; p < pend ; p++) {
                L_CLEAR (Lx,p);
            }
        }

        /* ------------------------------------------------------------------ */
        /* construct the scattered Map for supernode s */
        /* ------------------------------------------------------------------ */

        /* If row i is the kth row in s, then Map [i] = k.  Similarly, if
         * column j is the kth column in s, then  Map [j] = k. */

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    if ( nsrow > 128 )

        for (k = 0 ; k < nsrow ; k++)
        {
            PRINT1 (("  "ID" map "ID"\n", Ls [psi+k], k)) ;
            Map [Ls [psi + k]] = k ;
        }

        /* ------------------------------------------------------------------ */
        /* when using GPU, reorder supernodes by levels.*/
        /* (all supernodes in a level are independent) */
        /* ------------------------------------------------------------------ */

                    pthread_mutex_lock(thread_mutex_p);
#ifdef GPU_BLAS
        if ( useGPU )
        {
            TEMPLATE2 (CHOLMOD (gpu_reorder_descendants))
                ( Common, Super, &s, Lpi, Lpos, Head, Next, Previous,
                  &ndescendants, &tail, &mapCreatedOnGpu, gpu_p ) ;
        }
#endif
                    pthread_mutex_unlock(thread_mutex_p);

        /* ------------------------------------------------------------------ */
        /* copy matrix into supernode s (lower triangular part only) */
        /* ------------------------------------------------------------------ */

        pk = psx ;

#pragma omp parallel for private ( p, pend, pfend, pf, i, j, imap, q )  \
    num_threads(CHOLMOD_OMP_NUM_THREADS) if ( k2-k1 > 64 )

        for (k = k1 ; k < k2 ; k++)
        {
            if (stype != 0)
            {
                /* copy the kth column of A into the supernode */
                p = Ap [k] ;
                pend = (Apacked) ? (Ap [k+1]) : (p + Anz [k]) ;
                for ( ; p < pend ; p++)
                {
                    /* row i of L is located in row Map [i] of s */
                    i = Ai [p] ;
                    if (i >= k)
                    {
                        /* This test is here simply to avoid a segfault.  If
                         * the test is false, the numeric factorization of A
                         * is undefined.  It does not detect all invalid
                         * entries, only some of them (when debugging is
                         * enabled, and Map is cleared after each step, then
                         * all entries not in the pattern of L are detected). */
                        imap = Map [i] ;
                        if (imap >= 0 && imap < nsrow)
                        {
                            /* Lx [Map [i] + pk] = Ax [p] ; */
                            L_ASSIGN (Lx,(imap+(psx+(k-k1)*nsrow)), Ax,Az,p) ;
                        }
                    }
                }
            }
            else
            {
                double fjk[2];
                /* copy the kth column of A*F into the supernode */
                pf = Fp [k] ;
                pfend = (Fpacked) ? (Fp [k+1]) : (p + Fnz [k]) ;
                for ( ; pf < pfend ; pf++)
                {
                    j = Fi [pf] ;

                    /* fjk = Fx [pf] ; */
                    L_ASSIGN (fjk,0, Fx,Fz,pf) ;

                    p = Ap [j] ;
                    pend = (Apacked) ? (Ap [j+1]) : (p + Anz [j]) ;
                    for ( ; p < pend ; p++)
                    {
                        i = Ai [p] ;
                        if (i >= k)
                        {
                            /* See the discussion of imap above. */
                            imap = Map [i] ;
                            if (imap >= 0 && imap < nsrow)
                            {
                                /* Lx [Map [i] + pk] += Ax [p] * fjk ; */
                                L_MULTADD (Lx,(imap+(psx+(k-k1)*nsrow)),
                                           Ax,Az,p, fjk) ;
                            }
                        }
                    }
                }
            }
        }

        /* add beta to the diagonal of the supernode, if nonzero */
        if (beta [0] != 0.0)
        {
            /* note that only the real part of beta is used */
            pk = psx ;
            for (k = k1 ; k < k2 ; k++)
            {
                /* Lx [pk] += beta [0] ; */
                L_ASSEMBLE (Lx,pk, beta) ;
                pk += nsrow + 1 ;       /* advance to the next diagonal entry */
            }
        }

        PRINT1 (("Supernode with just A: repeat: "ID"\n", repeat_supernode)) ;
        DEBUG (CHOLMOD(dump_super) (s, Super, Lpi, Ls, Lpx, Lx, L_ENTRY,
                                    Common)) ;
        PRINT1 (("\n\n")) ;

        /* ------------------------------------------------------------------ */
        /* save/restore the list of supernodes */
        /* ------------------------------------------------------------------ */

        if (!repeat_supernode)
        {
            /* Save the list of pending descendants in case s is not positive
             * definite.  Also save Lpos for each descendant d, so that we can
             * find which part of d is used to update s. */
            for (d = Head [s] ; d != EMPTY ; d = Next [d])
            {
                Lpos_save [d] = Lpos [d] ;
                Next_save [d] = Next [d] ;
            }
        }
        else
        {
            for (d = Head [s] ; d != EMPTY ; d = Next [d])
            {
                Lpos [d] = Lpos_save [d] ;
                Next [d] = Next_save [d] ;
            }
        }

        /* ------------------------------------------------------------------ */
        /* update supernode s with each pending descendant d */
        /* ------------------------------------------------------------------ */

#ifndef NDEBUG
        for (d = Head [s] ; d != EMPTY ; d = Next [d])
        {
            PRINT1 (("\nWill update "ID" with Child: "ID"\n", s, d)) ;
            DEBUG (CHOLMOD(dump_super) (d, Super, Lpi, Ls, Lpx, Lx, L_ENTRY,
                                        Common)) ;
        }
        PRINT1 (("\nNow factorizing supernode "ID":\n", s)) ;
#endif

#ifdef GPU_BLAS
        /* initialize the buffer counter */
        if ( useGPU ) {
            Common->ibuffer = 0;
            supernodeUsedGPU = 0;
            idescendant = 0;
            d = Head[s];
            dnext = d;
            dlarge = Next[d];
            dsmall = tail;
            GPUavailable = 1;
            skips = 0;
        }
        else
        {
            dnext = Head[s];
        }
#else
        /* GPU module not installed */
        dnext = Head[s];
#endif

        while

#ifdef GPU_BLAS
            ( (!useGPU && (dnext != EMPTY))
               || (useGPU && (idescendant < ndescendants)))
#else

            ( dnext != EMPTY )
#endif
        {

#ifdef GPU_BLAS

            if ( useGPU ) {

                /* Conditionally select the next descendant supernode to
                 *  assemble.
                 *   + first, select the largest descendant
                 *   + subsequently, if gpu host buffers are available, select
                 *     the largest remaining descendant for assembly on the GPU
                 *   + otherwise select the smallest remaining descendant for
                 *     assembly on the CPU
                 *
                 * The objective is to keep the GPU busy assembling the largest
                 * descendants, and simultaneously keep the CPU busy assembling
                 * the smallest descendants.
                 *
                 * As this is called for every descendent supernode, moving
                 * this code to t_cholmod_gpu incurs substantial overhead -
                 * ~20 GF/s on audikw_1 - so it is being left here.
                 */

                iHostBuff =
                    (Common->ibuffer) % CHOLMOD_HOST_SUPERNODE_BUFFERS;
                cudaError_t cuErr;

                if ( idescendant > 0 )  {
                    if ( GPUavailable == -1 || skips > 0) {
                        d = dsmall;
                        dsmall = Previous[dsmall];
                        skips--;
                    }
                    else {
                        cuErr = cudaEventQuery
                            ( Common->updateCBuffersFree[iHostBuff] );
                        if ( cuErr == cudaSuccess ) {
                            /* buffers are available, so assemble a large
                             * descendant (anticipating that this will be
                             * assembled on the GPU) */
                            d = dlarge;
                            dlarge = Next[dlarge];
                            GPUavailable = 1;
                            skips = 0;
                        }
                        else {
                            /* buffers are not available, so the GPU is busy,
                             * so assemble a small descendant (anticipating
                             * that it will be assembled on the host) */
                            d = dsmall;
                            dsmall = Previous[dsmall];
                            GPUavailable = 0;

                            /* if the GPUs are busy, then do this many
                             * supernodes on the CPU before querying GPUs
                             * again. */
                            skips = CHOLMOD_GPU_SKIP;
                        }
                    }
                }

                idescendant++;

            }
            else
            {
                d = dnext;
            }
#else
            /* GPU module not installed at compile time */
            d = dnext ;
#endif
            /* -------------------------------------------------------------- */
            /* get the size of supernode d */
            /* -------------------------------------------------------------- */

            kd1 = Super [d] ;       /* d contains cols kd1 to kd2-1 of L */
            kd2 = Super [d+1] ;
            ndcol = kd2 - kd1 ;     /* # of columns in all of d */
            pdi = Lpi [d] ;         /* pointer to first row of d in Ls */
            pdx = Lpx [d] ;         /* pointer to first row of d in Lx */
            pdend = Lpi [d+1] ;     /* pointer just past last row of d in Ls */
            ndrow = pdend - pdi ;   /* # rows in all of d */

            PRINT1 (("Child: ")) ;
            DEBUG (CHOLMOD(dump_super) (d, Super, Lpi, Ls, Lpx, Lx, L_ENTRY,
                                        Common)) ;

            /* -------------------------------------------------------------- */
            /* find the range of rows of d that affect rows k1 to k2-1 of s */
            /* -------------------------------------------------------------- */

            p = Lpos [d] ;          /* offset of 1st row of d affecting s */
            pdi1 = pdi + p ;        /* ptr to 1st row of d affecting s in Ls */
            pdx1 = pdx + p ;        /* ptr to 1st row of d affecting s in Lx */

            /* there must be at least one row remaining in d to update s */
            ASSERT (pdi1 < pdend) ;
            PRINT1 (("Lpos[d] "ID" pdi1 "ID" Ls[pdi1] "ID"\n",
                     Lpos[d], pdi1, Ls [pdi1])) ;
            ASSERT (Ls [pdi1] >= k1 && Ls [pdi1] < k2) ;

            for (pdi2 = pdi1 ; pdi2 < pdend && Ls [pdi2] < k2 ; pdi2++) ;
            ndrow1 = pdi2 - pdi1 ;      /* # rows in first part of d */
            ndrow2 = pdend - pdi1 ;     /* # rows in remaining d */

            /* rows Ls [pdi1 ... pdi2-1] are in the range k1 to k2-1.  Since d
             * affects s, this set cannot be empty. */
            ASSERT (pdi1 < pdi2 && pdi2 <= pdend) ;
            PRINT1 (("ndrow1 "ID" ndrow2 "ID"\n", ndrow1, ndrow2)) ;
            DEBUG (for (p = pdi1 ; p < pdi2 ; p++)
                       PRINT1 (("Ls["ID"] "ID"\n", p, Ls[p]))) ;

            /* -------------------------------------------------------------- */
            /* construct the update matrix C for this supernode d */
            /* -------------------------------------------------------------- */

            /* C = L (k1:n-1, kd1:kd2-1) * L (k1:k2-1, kd1:kd2-1)', except
             * that k1:n-1 refers to all of the rows in L, but many of the
             * rows are all zero.  Supernode d holds columns kd1 to kd2-1 of L.
             * Nonzero rows in the range k1:k2-1 are in the list
             * Ls [pdi1 ... pdi2-1], of size ndrow1.  Nonzero rows in the range
             * k2:n-1 are in the list Ls [pdi2 ... pdend], of size ndrow2.  Let
             * L1 = L (Ls [pdi1 ... pdi2-1], kd1:kd2-1), and let
             * L2 = L (Ls [pdi2 ... pdend],  kd1:kd2-1).  C is ndrow2-by-ndrow1.
             * Let C1 be the first ndrow1 rows of C and let C2 be the last
             * ndrow2-ndrow1 rows of C.  Only the lower triangular part of C1
             * needs to be computed since C1 is symmetric.
             */

            /* maxcsize is the largest size of C for all pairs (d,s) */
            ASSERT (ndrow2 * ndrow1 <= ((Int) L->maxcsize)) ;

            /* compute leading ndrow1-by-ndrow1 lower triangular block of C,
             * C1 = L1*L1' */

            ndrow3 = ndrow2 - ndrow1 ;  /* number of rows of C2 */
            ASSERT (ndrow3 >= 0) ;


                    pthread_mutex_lock(thread_mutex_p);
#ifdef GPU_BLAS
            if ( useGPU ) {
                /* set up GPU to assemble new supernode */
                if ( GPUavailable == 1) {
                    if ( ndrow2 * L_ENTRY >= CHOLMOD_ND_ROW_LIMIT &&
                         ndcol * L_ENTRY >= CHOLMOD_ND_COL_LIMIT ) {
                        if ( ! mapCreatedOnGpu ) {
                            TEMPLATE2 ( CHOLMOD (gpu_initialize_supernode))
                                ( Common, nscol, nsrow, psi, gpu_p );
                            mapCreatedOnGpu = 1;
                        }
                    }
                    else {
                        /* we've reached the limit of GPU-eligible descendants
                         * flag to stop stop performing cudaEventQueries */
                        GPUavailable = -1;
                    }
                }
            }
#endif
                    pthread_mutex_unlock(thread_mutex_p);

                    pthread_mutex_lock(thread_mutex_p);
#ifdef GPU_BLAS
            if ( !useGPU
                || GPUavailable!=1
                || !TEMPLATE2 (CHOLMOD (gpu_updateC)) (ndrow1, ndrow2, ndrow,
                        ndcol, nsrow, pdx1, pdi1, Lx, C, Common, gpu_p))
#endif
            {
                /* GPU not installed, or not used */
#ifndef NTIMER

                Common->CHOLMOD_CPU_SYRK_CALLS++ ;
                tstart = SuiteSparse_time () ;
#endif
#ifdef REAL
                BLAS_dsyrk ("L", "N",
                    ndrow1, ndcol,              /* N, K: L1 is ndrow1-by-ndcol*/
                    one,                        /* ALPHA:  1 */
                    Lx + L_ENTRY*pdx1, ndrow,   /* A, LDA: L1, ndrow */
                    zero,                       /* BETA:   0 */
                    C, ndrow2) ;                /* C, LDC: C1 */
#else
                BLAS_zherk ("L", "N",
                    ndrow1, ndcol,              /* N, K: L1 is ndrow1-by-ndcol*/
                    one,                        /* ALPHA:  1 */
                    Lx + L_ENTRY*pdx1, ndrow,   /* A, LDA: L1, ndrow */
                    zero,                       /* BETA:   0 */
                    C, ndrow2) ;                /* C, LDC: C1 */
#endif
#ifndef NTIMER
                Common->CHOLMOD_CPU_SYRK_TIME += SuiteSparse_time () - tstart ;
#endif
                /* compute remaining (ndrow2-ndrow1)-by-ndrow1 block of C,
                 * C2 = L2*L1' */
                if (ndrow3 > 0)
                {
#ifndef NTIMER
                    Common->CHOLMOD_CPU_GEMM_CALLS++ ;
                    tstart = SuiteSparse_time () ;
#endif
#ifdef REAL
                    BLAS_dgemm ("N", "C",
                        ndrow3, ndrow1, ndcol,          /* M, N, K */
                        one,                            /* ALPHA:  1 */
                        Lx + L_ENTRY*(pdx1 + ndrow1),   /* A, LDA: L2 */
                        ndrow,                          /* ndrow */
                        Lx + L_ENTRY*pdx1,              /* B, LDB: L1 */
                        ndrow,                          /* ndrow */
                        zero,                           /* BETA:   0 */
                        C + L_ENTRY*ndrow1,             /* C, LDC: C2 */
                        ndrow2) ;
#else
                    BLAS_zgemm ("N", "C",
                        ndrow3, ndrow1, ndcol,          /* M, N, K */
                        one,                            /* ALPHA:  1 */
                        Lx + L_ENTRY*(pdx1 + ndrow1),   /* A, LDA: L2 */
                        ndrow,                          /* ndrow */
                        Lx + L_ENTRY*pdx1,              /* B, LDB: L1, ndrow */
                        ndrow,
                        zero,                           /* BETA:   0 */
                        C + L_ENTRY*ndrow1,             /* C, LDC: C2 */
                        ndrow2) ;
#endif
#ifndef NTIMER
                    Common->CHOLMOD_CPU_GEMM_TIME +=
                        SuiteSparse_time () - tstart ;
#endif
                }

                /* ---------------------------------------------------------- */
                /* construct relative map to assemble d into s */
                /* ---------------------------------------------------------- */

                DEBUG (CHOLMOD(dump_real) ("C", C, ndrow2, ndrow1, TRUE,
                                           L_ENTRY, Common)) ;

#pragma omp parallel for num_threads(CHOLMOD_OMP_NUM_THREADS)   \
    if ( ndrow2 > 64 )

                for (i = 0 ; i < ndrow2 ; i++)
                {
                    RelativeMap [i] = Map [Ls [pdi1 + i]] ;
                    ASSERT (RelativeMap [i] >= 0 && RelativeMap [i] < nsrow) ;
                }

                /* ---------------------------------------------------------- */
                /* assemble C into supernode s using the relative map */
                /* ---------------------------------------------------------- */

#pragma omp parallel for private ( j, i, px, q )                \
    num_threads(CHOLMOD_OMP_NUM_THREADS) if (ndrow1 > 64 )

                for (j = 0 ; j < ndrow1 ; j++)              /* cols k1:k2-1 */
                {
                    ASSERT (RelativeMap [j] == Map [Ls [pdi1 + j]]) ;
                    ASSERT (RelativeMap [j] >= 0 && RelativeMap [j] < nscol) ;
                    px = psx + RelativeMap [j] * nsrow ;
                    for (i = j ; i < ndrow2 ; i++)          /* rows k1:n-1 */
                    {
                        ASSERT (RelativeMap [i] == Map [Ls [pdi1 + i]]) ;
                        ASSERT (RelativeMap [i] >= j && RelativeMap[i] < nsrow);
                        q = px + RelativeMap [i] ;
                        L_ASSEMBLESUB (Lx,q, C, i+ndrow2*j) ;
                    }
                }

            }
#ifdef GPU_BLAS
            else
            {
                supernodeUsedGPU = 1;   /* GPU was used for this supernode*/
                Common->ibuffer++;      /* gpu_updateC is asynchronous, so use
                                         * the next host buffer for the next
                                         * supernode */
                Common->ibuffer = Common->ibuffer%
                    (CHOLMOD_HOST_SUPERNODE_BUFFERS*CHOLMOD_DEVICE_STREAMS);
            }
#endif
                    pthread_mutex_unlock(thread_mutex_p);

            /* -------------------------------------------------------------- */
            /* prepare this supernode d for its next ancestor */
            /* -------------------------------------------------------------- */

            dnext = Next [d] ;

            if (!repeat_supernode)
            {
                /* If node s is being repeated, Head [dancestor] has already
                 * been cleared (set to EMPTY).  It must remain EMPTY.  The
                 * dancestor will not be factorized since the factorization
                 * terminates at node s. */
                Lpos [d] = pdi2 - pdi ;
                if (Lpos [d] < ndrow)
                {
                    dancestor = SuperMap [Ls [pdi2]] ;
                    ASSERT (dancestor > s && dancestor < nsuper) ;
                    pthread_mutex_lock(thread_mutex_p);
                    /* place d in the link list of its next ancestor */
                    Next [d] = Head [dancestor] ;
                    Head [dancestor] = d ;
                    pthread_mutex_unlock(thread_mutex_p);
                }
            }

        }  /* end of descendant supernode loop */

                    pthread_mutex_lock(thread_mutex_p);
#ifdef GPU_BLAS
        if ( useGPU ) {
            iHostBuff = (Common->ibuffer)%CHOLMOD_HOST_SUPERNODE_BUFFERS;
            iDevBuff = (Common->ibuffer)%CHOLMOD_DEVICE_STREAMS;

            /* combine updates assembled on the GPU with updates
             * assembled on the CPU */
            TEMPLATE2 ( CHOLMOD (gpu_final_assembly ))
                ( Common, Lx, psx, nscol, nsrow, supernodeUsedGPU,
                  &iHostBuff, &iDevBuff, gpu_p );
        }
#endif
                    pthread_mutex_unlock(thread_mutex_p);

        PRINT1 (("\nSupernode with contributions A: repeat: "ID"\n",
                 repeat_supernode)) ;
        DEBUG (CHOLMOD(dump_super) (s, Super, Lpi, Ls, Lpx, Lx, L_ENTRY,
                                    Common)) ;
        PRINT1 (("\n\n")) ;

        /* ------------------------------------------------------------------ */
        /* factorize diagonal block of supernode s in LL' */
        /* ------------------------------------------------------------------ */

        /* The current supernode s is ready to factorize.  It has been updated
         * by all descendant supernodes.  Let S = the current supernode, which
         * holds rows k1:n-1 and columns k1:k2-1 of the updated matrix.   It
         * splits into two parts:  the square diagonal block S1, and the
         * rectangular part S2.  Here, S1 is factorized into L1*L1' and
         * overwritten by L1.
         *
         * If supernode s is being repeated, only factorize it up to but not
         * including the column containing the problematic entry.
         */

        nscol2 = (repeat_supernode) ? (nscol_new) : (nscol) ;

                    pthread_mutex_lock(thread_mutex_p);
#ifdef GPU_BLAS
        if ( !useGPU
            || !supernodeUsedGPU
            || !TEMPLATE2 (CHOLMOD (gpu_lower_potrf))(nscol2, nsrow, psx, Lx,
                                                   &info, Common, gpu_p))
#endif
        {
            /* Note that the GPU will not be used for the triangular solve */
#ifdef GPU_BLAS
            supernodeUsedGPU = 0;
#endif
#ifndef NTIMER
            Common->CHOLMOD_CPU_POTRF_CALLS++ ;
            tstart = SuiteSparse_time () ;
#endif
#ifdef REAL
            LAPACK_dpotrf ("L",
                nscol2,                     /* N: nscol2 */
                Lx + L_ENTRY*psx, nsrow,    /* A, LDA: S1, nsrow */
                info) ;                     /* INFO */
#else
            LAPACK_zpotrf ("L",
                nscol2,                     /* N: nscol2 */
                Lx + L_ENTRY*psx, nsrow,    /* A, LDA: S1, nsrow */
                info) ;                     /* INFO */
#endif
#ifndef NTIMER
            Common->CHOLMOD_CPU_POTRF_TIME += SuiteSparse_time ()- tstart ;
#endif
        }
                    pthread_mutex_unlock(thread_mutex_p);

        /* ------------------------------------------------------------------ */
        /* check if the matrix is not positive definite */
        /* ------------------------------------------------------------------ */

        if (repeat_supernode)
        {
            /* the leading part has been refactorized; it must have succeeded */
            info = 0 ;

            /* zero out the rest of this supernode */
            p = psx + nsrow * nscol_new ;
            pend = psx + nsrow * nscol ;            /* s is nsrow-by-nscol */
            for ( ; p < pend ; p++)
            {
                /* Lx [p] = 0 ; */
                L_CLEAR (Lx,p) ;
            }
        }

        /* info is set to one in LAPACK_*potrf if blas_ok is FALSE.  It is
         * set to zero in dpotrf/zpotrf if the factorization was successful. */
        if (CHECK_BLAS_INT && !Common->blas_ok)
        {
            ERROR (CHOLMOD_TOO_LARGE, "problem too large for the BLAS") ;
        }

        if (info != 0)
        {
            /* Matrix is not positive definite.  dpotrf/zpotrf do NOT report an
             * error if the diagonal of L has NaN's, only if it has a zero. */
            if (Common->status == CHOLMOD_OK)
            {
                ERROR (CHOLMOD_NOT_POSDEF, "matrix not positive definite") ;
            }

            /* L->minor is the column of L that contains a zero or negative
             * diagonal term. */
            L->minor = k1 + info - 1 ;

            /* clear the link lists of all subsequent supernodes */
            for (ss = s+1 ; ss < nsuper ; ss++)
            {
                Head [ss] = EMPTY ;
            }

            /* zero this supernode, and all remaining supernodes */
            pend = L->xsize ;
            for (p = psx ; p < pend ; p++)
            {
                /* Lx [p] = 0. ; */
                L_CLEAR (Lx,p) ;
            }

            /* If L is indefinite, it still contains useful information.
             * Supernodes 0 to s-1 are valid, similar to MATLAB [R,p]=chol(A),
             * where the 1-based p is identical to the 0-based L->minor.  Since
             * L->minor is in the current supernode s, it and any columns to the
             * left of it in supernode s are also all zero.  This differs from
             * [R,p]=chol(A), which contains nonzero rows 1 to p-1.  Fix this
             * by setting repeat_supernode to TRUE, and repeating supernode s.
             *
             * If Common->quick_return_if_not_posdef is true, then the entire
             * supernode s is not factorized; it is left as all zero.
             */

            if (info == 1 || Common->quick_return_if_not_posdef)
            {
                /* If the first column of supernode s contains a zero or
                 * negative diagonal entry, then it is already properly set to
                 * zero.  Also, info will be 1 if integer overflow occured in
                 * the BLAS. */
                Head [s] = EMPTY ;

                *(thread_args->to_return_p) = TRUE;
                return void_args;
            }
            else
            {
                /* Repeat supernode s, but only factorize it up to but not
                 * including the column containing the problematic diagonal
                 * entry. */
                repeat_supernode = TRUE ;
                //s-- ;
                nscol_new = info - 1 ;

                thread_args->s = s;
                thread_args->nscol_new = nscol_new;
                thread_args->repeat_supernode = repeat_supernode;
                TEMPLATE (cholmod_super_numeric_pthread) (void_args);

                return void_args;
            }
        }

        /* ------------------------------------------------------------------ */
        /* compute the subdiagonal block and prepare supernode for its parent */
        /* ------------------------------------------------------------------ */

        nsrow2 = nsrow - nscol2 ;
        if (nsrow2 > 0)
        {
            /* The current supernode is columns k1 to k2-1 of L.  Let L1 be the
             * diagonal block (factorized by dpotrf/zpotrf above; rows/cols
             * k1:k2-1), and L2 be rows k2:n-1 and columns k1:k2-1 of L.  The
             * triangular system to solve is L2*L1' = S2, where S2 is
             * overwritten with L2.  More precisely, L2 = S2 / L1' in MATLAB
             * notation.
             */

                    pthread_mutex_lock(thread_mutex_p);
#ifdef GPU_BLAS
            if ( !useGPU
                || !supernodeUsedGPU
                || !TEMPLATE2 (CHOLMOD(gpu_triangular_solve))
                        (nsrow2, nscol2, nsrow, psx, Lx, Common, gpu_p))
#endif
            {
#ifndef NTIMER
                Common->CHOLMOD_CPU_TRSM_CALLS++ ;
                tstart = SuiteSparse_time () ;
#endif
#ifdef REAL
                BLAS_dtrsm ("R", "L", "C", "N",
                    nsrow2, nscol2,                 /* M, N */
                    one,                            /* ALPHA: 1 */
                    Lx + L_ENTRY*psx, nsrow,        /* A, LDA: L1, nsrow */
                    Lx + L_ENTRY*(psx + nscol2),    /* B, LDB, L2, nsrow */
                    nsrow) ;
#else
                BLAS_ztrsm ("R", "L", "C", "N",
                    nsrow2, nscol2,                 /* M, N */
                    one,                            /* ALPHA: 1 */
                    Lx + L_ENTRY*psx, nsrow,        /* A, LDA: L1, nsrow */
                    Lx + L_ENTRY*(psx + nscol2),    /* B, LDB, L2, nsrow */
                    nsrow) ;
#endif
#ifndef NTIMER
                Common->CHOLMOD_CPU_TRSM_TIME += SuiteSparse_time () - tstart ;
#endif
            }
                    pthread_mutex_unlock(thread_mutex_p);

            if (CHECK_BLAS_INT && !Common->blas_ok)
            {
                ERROR (CHOLMOD_TOO_LARGE, "problem too large for the BLAS") ;
            }

            if (!repeat_supernode)
            {
                /* Lpos [s] is offset of first row of s affecting its parent */
                Lpos [s] = nscol ;
                sparent = SuperMap [Ls [psi + nscol]] ;
                ASSERT (sparent != EMPTY) ;
                ASSERT (Ls [psi + nscol] >= Super [sparent]) ;
                ASSERT (Ls [psi + nscol] <  Super [sparent+1]) ;
                ASSERT (SuperMap [Ls [psi + nscol]] == sparent) ;
                ASSERT (sparent > s && sparent < nsuper) ;
                pthread_mutex_lock(thread_mutex_p);
                /* place s in link list of its parent */
                Next [s] = Head [sparent] ;
                Head [sparent] = s ;
                pthread_mutex_unlock(thread_mutex_p);
            }
        }
        else
        {
#ifdef GPU_BLAS
                    pthread_mutex_lock(thread_mutex_p);
            TEMPLATE2 ( CHOLMOD (gpu_copy_supernode) )
                ( Common, Lx, psx, nscol, nscol2, nsrow,
                  supernodeUsedGPU, iHostBuff, gpu_p);
                    pthread_mutex_unlock(thread_mutex_p);
#endif
        }

        Head [s] = EMPTY ;  /* link list for supernode s no longer needed */

        /* clear the Map (debugging only, to detect changes in pattern of A) */
        DEBUG (for (k = 0 ; k < nsrow ; k++) Map [Ls [psi + k]] = EMPTY) ;
        DEBUG (CHOLMOD(dump_super) (s, Super, Lpi, Ls, Lpx, Lx, L_ENTRY,
                                    Common)) ;

        if (repeat_supernode)
        {
            /* matrix is not positive definite; finished clean-up for supernode
             * containing negative diagonal */

                *(thread_args->to_return_p) = TRUE;
        }

        front_col[s] = Lpi[s+1];
        sparent = SuperMap [Ls [psi + nscol]] ;
        if (sparent != EMPTY)
            pending[sparent]--;

        Map = CHOLMOD(free) (n, sizeof(Int), Map, Common);
        RelativeMap = CHOLMOD(free) (n, sizeof(Int), RelativeMap, Common);
        C = CHOLMOD(free) (n, sizeof(double), C, Common);

        thread_args->Map = Map;
        thread_args->RelativeMap = RelativeMap;
        thread_args->C = C;

#ifdef GPU_BLAS
    pthread_mutex_lock(thread_mutex_p);
    if (useGPU && GPUslot_p != NULL)
        *GPUslot_p = TRUE;
    pthread_mutex_unlock(thread_mutex_p);
#endif

    sem_post(thread_semaphore_p);

    return void_args;
}

/* ========================================================================== */
/* === t_cholmod_super_numeric ============================================== */
/* ========================================================================== */

/* This function returns FALSE only if integer overflow occurs in the BLAS.
 * It returns TRUE otherwise whether or not the matrix is positive definite. */

static int TEMPLATE (cholmod_super_numeric)
(
    /* ---- input ---- */
    cholmod_sparse *A,  /* matrix to factorize */
    cholmod_sparse *F,  /* F = A' or A(:,f)' */
    double beta [2],    /* beta*I is added to diagonal of matrix to factorize */
    /* ---- in/out --- */
    cholmod_factor *L,  /* factorization */
    /* -- workspace -- */
    cholmod_dense *Cwork,       /* size (L->maxcsize)-by-1 */
    /* --------------- */
    cholmod_common *Common
    )
{
    int to_return = FALSE, end_of_factorization = FALSE, front_ready;
    sem_t thread_semaphore;
    pthread_mutex_t main_mutex, thread_mutex;
    pthread_t *threads;
    TEMPLATE (CHOLMOD (thread_args)) *thread_args;

    double one [2], zero [2];
    double *Lx;
    Int *Super, *Ls, *Lpi, *Lpx, *SuperMap, *Next,
        *Lpos, *Iwork, *Next_save, *Lpos_save,
        *Previous, *pending, *front_col;
    Int i, nsuper, n, s, sparent, t;

    sem_init(&thread_semaphore, 0, CHOLMOD_PTHREADS_NUM_THREADS);
    pthread_mutex_init(&main_mutex, NULL);
    pthread_mutex_init(&thread_mutex, NULL);

    /* ---------------------------------------------------------------------- */
    /* declarations for the GPU */
    /* ---------------------------------------------------------------------- */

    /* these variables are not used if the GPU module is not installed */

#ifdef GPU_BLAS
    int useGPU = FALSE, useGPU_p[GPU_NUM];
    cholmod_gpu_pointers gpu_p[GPU_NUM];
#endif

    /* ---------------------------------------------------------------------- */
    /* guard against integer overflow in the BLAS */
    /* ---------------------------------------------------------------------- */

    /* If integer overflow occurs in the BLAS, Common->status is set to
     * CHOLMOD_TOO_LARGE, and the contents of Lx are undefined. */
    Common->blas_ok = TRUE ;

    /* ---------------------------------------------------------------------- */
    /* get inputs */
    /* ---------------------------------------------------------------------- */

    nsuper = L->nsuper ;
    n = L->n ;

    one [0] =  1.0 ;    /* ALPHA for *syrk, *herk, *gemm, and *trsm */
    one [1] =  0. ;
    zero [0] = 0. ;     /* BETA for *syrk, *herk, and *gemm */
    zero [1] = 0. ;

    /* Iwork must be of size 2n + 5*nsuper, allocated in the caller,
     * cholmod_super_numeric.  The memory cannot be allocated here because the
     * cholmod_super_numeric initializes SuperMap, and cholmod_allocate_work
     * does not preserve existing workspace if the space needs to be increase
     * in size. */

    /* allocate integer workspace */
    Iwork = Common->Iwork ;
    SuperMap    = Iwork ;                                   /* size n (i/i/l) */
    //RelativeMap = Iwork + n ;                               /* size n (i/i/l) */
    Next        = Iwork + 2*((size_t) n) ;                  /* size nsuper*/
    Lpos        = Iwork + 2*((size_t) n) + nsuper ;         /* size nsuper*/
    Next_save   = Iwork + 2*((size_t) n) + 2*((size_t) nsuper) ;/* size nsuper*/
    Lpos_save   = Iwork + 2*((size_t) n) + 3*((size_t) nsuper) ;/* size nsuper*/
    Previous    = Iwork + 2*((size_t) n) + 4*((size_t) nsuper) ;/* size nsuper*/
    pending     = Iwork + 2*((size_t) n) + 5*((size_t) nsuper) ;/* size nsuper*/
    front_col   = Iwork + 2*((size_t) n) + 6*((size_t) nsuper) ;/* size nsuper*/

    Ls = L->s ;
    Lpi = L->pi ;
    Lpx = L->px ;

    Super = L->super ;

    Lx = L->x ;

    for (s = 0; s < nsuper; s++)
        pending[s] = 0;
    for (s = 0; s < nsuper; s++)
    {
        sparent = SuperMap[Ls[Lpi[s]+(Super[s+1]-Super[s])]];
        if (sparent > s && sparent < nsuper)
            pending[sparent]++;
    }
    for (s = 0; s < nsuper; s++)
        front_col[s] = -1;

#ifdef GPU_BLAS
    for (i = 0; i < GPU_NUM; i++)
    {
        /* local copy of useGPU */
        if ( (Common->useGPU == 1) && L->useGPU)
        {
            /* Initialize the GPU.  If not found, don't use it. */
            useGPU_p[i] = TEMPLATE2 (CHOLMOD (gpu_init))
                (/*C, */L, Common, nsuper, n, Lpi[nsuper]-Lpi[0], i, &gpu_p[i]) ;
        }
        else
        {
            useGPU_p[i] = 0;
        }
        /* fprintf (stderr, "local useGPU %d\n", useGPU) ; */
        useGPU |= useGPU_p[i];
    }
#endif

#ifndef NTIMER
    /* clear GPU / CPU statistics */
    Common->CHOLMOD_CPU_GEMM_CALLS  = 0 ;
    Common->CHOLMOD_CPU_SYRK_CALLS  = 0 ;
    Common->CHOLMOD_CPU_TRSM_CALLS  = 0 ;
    Common->CHOLMOD_CPU_POTRF_CALLS = 0 ;
    Common->CHOLMOD_GPU_GEMM_CALLS  = 0 ;
    Common->CHOLMOD_GPU_SYRK_CALLS  = 0 ;
    Common->CHOLMOD_GPU_TRSM_CALLS  = 0 ;
    Common->CHOLMOD_GPU_POTRF_CALLS = 0 ;
    Common->CHOLMOD_CPU_GEMM_TIME   = 0 ;
    Common->CHOLMOD_CPU_SYRK_TIME   = 0 ;
    Common->CHOLMOD_CPU_TRSM_TIME   = 0 ;
    Common->CHOLMOD_CPU_POTRF_TIME  = 0 ;
    Common->CHOLMOD_GPU_GEMM_TIME   = 0 ;
    Common->CHOLMOD_GPU_SYRK_TIME   = 0 ;
    Common->CHOLMOD_GPU_TRSM_TIME   = 0 ;
    Common->CHOLMOD_GPU_POTRF_TIME  = 0 ;
    Common->CHOLMOD_ASSEMBLE_TIME   = 0 ;
    Common->CHOLMOD_ASSEMBLE_TIME2  = 0 ;
#endif

    /* If the matrix is not positive definite, the supernode s containing the
     * first zero or negative diagonal entry of L is repeated (but factorized
     * only up to just before the problematic diagonal entry). The purpose is
     * to provide MATLAB with [R,p]=chol(A); columns 1 to p-1 of L=R' are
     * required, where L(p,p) is the problematic diagonal entry.  The
     * repeat_supernode flag tells us whether this is the repeated supernode.
     * Once supernode s is repeated, the factorization is terminated. */

#ifdef GPU_BLAS
    if ( useGPU )
    {
        /* Case of GPU, zero all supernodes at one time for better performance*/
        TEMPLATE2 (CHOLMOD (gpu_clear_memory))(Lx, L->xsize,
            CHOLMOD_OMP_NUM_THREADS);
    }
#endif

    threads = CHOLMOD(malloc) (nsuper, sizeof(pthread_t), Common);
    thread_args = CHOLMOD(malloc) (nsuper, sizeof(TEMPLATE (CHOLMOD (thread_args))), Common);

    /* ---------------------------------------------------------------------- */
    /* supernodal numerical factorization */
    /* ---------------------------------------------------------------------- */

    while (!end_of_factorization)
    {
        end_of_factorization = TRUE;
        for (s = nsuper - 1 ; s >= 0 ; s--)
        //for (s = 0 ; s < nsuper ; s++)
        {
            if (front_col[s] < 0)
            {
                end_of_factorization = FALSE;
                if (pending[s] <= 0)
                {
                    sem_wait(&thread_semaphore);
                    front_col[s] = Lpi[s];
                    thread_args[s].A = A;
                    thread_args[s].F = F;
                    thread_args[s].zero = zero;
                    thread_args[s].one = one;
                    thread_args[s].beta = beta;
                    thread_args[s].L = L;
                    thread_args[s].Cwork = Cwork;
                    thread_args[s].Common = Common;
                    thread_args[s].Map = CHOLMOD(malloc) (n, sizeof(Int), Common);
                    thread_args[s].RelativeMap = CHOLMOD(malloc) (n, sizeof(Int), Common);
                    thread_args[s].C = CHOLMOD(malloc) (L->maxcsize, sizeof(double), Common);
#ifdef GPU_BLAS
                    thread_args[s].GPUslot_p = NULL;
                    thread_args[s].useGPU = FALSE;
                    thread_args[s].gpu_p = GPU_NUM;
                    for (i = 0; i < GPU_NUM; i++)
                        if (useGPU_p[i])
                        {
                            useGPU_p[i] = FALSE;
                            thread_args[s].GPUslot_p = &useGPU_p[i];
                            thread_args[s].useGPU = TRUE;
                            thread_args[s].gpu_p = gpu_p;
                        }
#endif
                    thread_args[s].thread_semaphore_p = &thread_semaphore;
                    thread_args[s].thread_mutex_p = &thread_mutex;
                    thread_args[s].s = s;
                    thread_args[s].nscol_new = 0;
                    thread_args[s].repeat_supernode = FALSE;
                    thread_args[s].to_return_p = &to_return;;

                    pthread_create(&threads[s], NULL, TEMPLATE (cholmod_super_numeric_pthread), &thread_args[s]);
                    //pthread_join(threads[s], NULL);
                }
            }
        }
        if (to_return)
        {
#ifdef GPU_BLAS
            if ( useGPU ) {
                CHOLMOD (gpu_end) (Common) ;
            }
#endif
            return (Common->status >= CHOLMOD_OK) ;
        }
    }

    for (s = 0 ; s < nsuper ; s++)
        pthread_join(threads[s], NULL);

    sem_destroy(&thread_semaphore);
    pthread_mutex_destroy(&main_mutex);
    pthread_mutex_destroy(&thread_mutex);

    threads = CHOLMOD(free) (nsuper, sizeof(pthread_t), threads, Common);
    thread_args = CHOLMOD(free) (nsuper, sizeof(TEMPLATE (CHOLMOD (thread_args))), thread_args, Common);

    /* success; matrix is positive definite */
    L->minor = n ;

#ifdef GPU_BLAS
    if ( useGPU )
    {
        CHOLMOD (gpu_end) (Common) ;
    }
#endif

    return (Common->status >= CHOLMOD_OK) ;

}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX
