/* ========================================================================== */
/* === GPU/t_factorize_root_parallel.c ====================================== */
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
 *   t_factorize_root
 *
 * Description:
 *   Contains functions for factorization of the root algorithm.
 *   Returns 1 if matrix not positive-definite, 0 otherwise.
 *
 */


/* includes */
#include <string.h>
#include <time.h>
#ifdef MKLROOT
#include "mkl.h"
#endif
#include "nvToolsExt.h"
#include <pthread.h>


/* undef macros */
#undef L_ENTRY
#undef L_CLEAR
#undef L_ASSIGN
#undef L_MULTADD
#undef L_ASSEMBLE
#undef L_ASSEMBLESUB


/* macros */
#ifdef REAL

/* A, F, and L are all real */
#define L_ENTRY 1
#define L_CLEAR(Lx,p)                   Lx [p] = 0
#define L_ASSIGN(Lx,q, Ax,Az,p)         Lx [q] = Ax [p]
#define L_MULTADD(Lx,q, Ax,Az,p, f)     Lx [q] += Ax [p] * f [0]
#define L_ASSEMBLE(Lx,q,b)              Lx [q] += b [0]
#define L_ASSEMBLESUB(Lx,q,C,p)         Lx [q] -= C [p]

#else

/* A and F are complex or zomplex, L and C are complex */
#define L_ENTRY 2
#define L_CLEAR(Lx,p)                   Lx [2*(p)] = 0 ; Lx [2*(p)+1] = 0
#define L_ASSEMBLE(Lx,q,b)              Lx [2*(q)] += b [0] ;
#define L_ASSEMBLESUB(Lx,q,C,p)         Lx [2*(q)  ] -= C [2*(p)  ] ; \
                                                        Lx [2*(q)+1] -= C [2*(p)+1] ;

#ifdef COMPLEX
/* A, F, L, and C are all complex */
#define L_ASSIGN(Lx,q, Ax,Az,p)         Lx [2*(q)  ] = Ax [2*(p)  ] ; \
                                                       Lx [2*(q)+1] = Ax [2*(p)+1]
#define L_MULTADD(Lx,q, Ax,Az,p, f)     Lx [2*(q)  ] += Ax [2*(p)  ] * f [0] - Ax [2*(p)+1] * f [1] ;           \
                                                        Lx [2*(q)+1] += Ax [2*(p)+1] * f [0] + Ax [2*(p)  ] * f [1]

#else
/* A and F are zomplex, L and C is complex */
#define L_ASSIGN(Lx,q, Ax,Az,p)         Lx [2*(q)  ] = Ax [p] ;                 \
                                                       Lx [2*(q)+1] = Az [p] ;
#define L_MULTADD(Lx,q, Ax,Az,p, f)     Lx [2*(q)  ] += Ax [p] * f [0] - Az [p] * f [1] ;   \
                                                        Lx [2*(q)+1] += Az [p] * f [0] + Ax [p] * f [1]
#endif
#endif


/*
 * Function:
 *   gpu_factorize_root_parallel
 *
 * Description:
 *   Factorizes top-of-tree of elimination tree, where
 *   the subtree does not fit the GPU. Utilizes a hybrid algorithm
 *   presented at GTC14.
 *   Returns 0 if matrix not positive definite, 1 otherwise.
 *
 */
    int TEMPLATE2 (CHOLMOD (gpu_factorize_root_parallel))
(
 cholmod_common *Common,
 cholmod_factor *L,
 cholmod_gpu_pointers *gpu_p,
 cholmod_cpu_pointers *cpu_p,
 cholmod_tree_pointers *tree_p,
 Int subtree
 )
{
#ifdef SUITESPARSE_CUDA
    /* local variables */
    size_t devBuffSize;
    int gpuid, numThreads, numThreads1;
    Int start_global, end_global, node, Apacked, Fpacked, stype, n;
    Int *Ls, *Lpi, *Lpx, *Lpos, *Fp, *Fi, *Fnz, *Ap, *Ai, *Anz, *Super, *h_Map, *SuperMap, *Head, *Next, *Next_save, *Previous, *Lpos_save,
        *supernode_levels, *supernode_levels_ptrs, *supernode_levels_subtree_ptrs, *supernode_num_levels;
    double *Lx, *Ax, *Az, *Fx, *Fz, *h_C, *beta;
    double one[2] = {1.0, 0.0}, zero[2] = {0.0, 0.0};

    omp_set_nested(1);

    /*
     * Set variables & pointers
     */
    /* set host variables */
    n	 		= L->n;
    numThreads		= Common->ompNumThreads;
    numThreads1		= (Common->ompNumThreads + Common->numGPU - 1)/Common->numGPU;
    gpu_p->gpuid 		= 0;
    devBuffSize		= ((size_t)(Common->devBuffSize))/sizeof(double);
    Apacked               = cpu_p->Apacked;
    Fpacked               = cpu_p->Fpacked;
    stype                 = cpu_p->stype;
    beta                  = cpu_p->beta;

    /* set host pointers */
    Ls            = cpu_p->Ls;
    Lpi           = cpu_p->Lpi;
    Lpx           = L->px;
    Lpos          = cpu_p->Lpos;
    Fp            = cpu_p->Fp;
    Fi            = cpu_p->Fi;
    Fnz           = cpu_p->Fnz;
    Ap            = cpu_p->Ap;
    Ai            = cpu_p->Ai;
    Anz           = cpu_p->Anz;
    Super         = cpu_p->Super;
    h_Map         = cpu_p->Map;
    SuperMap      = cpu_p->SuperMap;
    Head          = cpu_p->Head;
    Next          = cpu_p->Next;
    Next_save     = cpu_p->Next_save;
    Lpos_save     = cpu_p->Lpos_save;
    Previous	= cpu_p->Previous;
    Lx            = cpu_p->Lx;
    Ax            = cpu_p->Ax;
    Az            = cpu_p->Az;
    Fx            = cpu_p->Fx;
    Fz            = cpu_p->Fz;
    h_C           = cpu_p->C;

    /* set tree pointers */
    supernode_levels              = tree_p->supernode_levels;
    supernode_levels_ptrs         = tree_p->supernode_levels_ptrs;
    supernode_levels_subtree_ptrs  = tree_p->supernode_levels_subtree_ptrs;
    supernode_num_levels          = tree_p->supernode_num_levels;



    /* initialize GPU */
    for(gpuid = 0; gpuid < Common->numGPU; gpuid++) {
        TEMPLATE2 (CHOLMOD (gpu_init_root))(Common, gpu_p, L, Lpi, L->nsuper, n, gpuid);
    }


    /*
     *  loop over levels in subtree
     *  Previously this looped over levels, synchronizing between levels.
     *  Now this loops over all supernodes, ordered by levels, but with no synchronization
     *  between levels. Instead, supernodes from different levels can proceed in parallel,
     *  with appropriate flags and spin-waits to ensure descendant supernodes are complete.
     *  This provided a major performance increase when using multiple GPUs.
     */
    {

        start_global = supernode_levels_ptrs[supernode_levels_subtree_ptrs[subtree]];
        end_global = supernode_levels_ptrs[supernode_levels_subtree_ptrs[subtree]+supernode_num_levels[subtree]];

        Int *Next_local = (Int*) malloc ( (end_global+1)*sizeof(Int) );
        Int *Previous_local = (Int*) malloc ( (end_global+1)*sizeof(Int) );
        Int *Lpos_local = (Int*) malloc ( (end_global+1)*sizeof(Int) );

        /* create two vectors - one with the supernode id and one with a counter to synchronize supernodes */
        Int event_len = end_global - start_global;
        Int *pending = (Int *) malloc (event_len*sizeof(Int));
        Int *leaves = (Int *) malloc (event_len*sizeof(Int));
        Int nleaves;
        omp_lock_t *node_locks = (omp_lock_t *) malloc (event_len*sizeof(omp_lock_t));

        for ( node = start_global; node < end_global; node++ ) {
            pending[node-start_global] = 0;
        }

        for ( node = start_global; node < end_global; node++ ) {
            Int s, psi, nscol, sparent;
            Int inode;
            s = supernode_levels[node];
            psi = Lpi[s];
            nscol = Super[s+1] - Super[s];
            sparent = SuperMap [Ls [psi + nscol]];
            if (sparent > s && sparent < L->nsuper)
            {
                for (inode = node; inode < end_global; inode++)
                {
                    if (sparent == supernode_levels[inode])
                    {
                        pending[inode-start_global]++;
                        break;
                    }
                }
            }
        }
        
        nleaves = 0;

        for ( node = start_global; node < end_global; node++ ) {
            Int s;
            s = supernode_levels[node];
            if (pending[node-start_global] == 0)
            leaves[nleaves++] = s;
        }

        for ( node=0; node<event_len; node++ ) {
            omp_init_lock(&node_locks[node]);
        }

        /* loop over supernodes */
        {
            Int leaf_idx;
            printf ("num_threads = %ld, numGPU = %d, threads = %d\n", Common->numGPU, Common->numGPU_physical, Common->numGPU_parallel);
#pragma omp parallel for schedule(static) private ( leaf_idx, node, gpuid ) num_threads(Common->numGPU)
            for(leaf_idx = 0; leaf_idx < nleaves; leaf_idx++)
            {


                /* local variables */
                int i, j, k;
                Int px, pk, pf, p, q, d, s, ss, ndrow, ndrow1, ndrow2, ndrow3, ndcol, nsrow, nsrow2, nscol, nscol2, nscol3,
                    kd1, kd2, k1, k2, psx, psi, pdx, pdx1, pdi, pdi1, pdi2, pdend, psend, pfend, pend, dancestor, sparent, imap,
                    idescendant, ndescendants, dlarge, iHostBuff, iDevBuff, iDevCBuff, dsmall, tail, info = 0,
                    GPUavailable, mapCreatedOnGpu, supernodeUsedGPU;
                Int repeat_supernode;
                cudaError_t cuErrHost;
#ifdef QUERY_LX_EVENTS
                cudaError_t cuErrDev;
#endif

                int desc_count;
                int syrk_count;
                int gemm_count;
                Int counter;

                struct cholmod_desc_t desc[Common->ompNumThreads];
                struct cholmod_syrk_t syrk[Common->ompNumThreads];
                struct cholmod_gemm_t gemm[Common->ompNumThreads];

                /* set device id, pointers */
                gpuid  		= omp_get_thread_num();			/* get gpuid */
                Int *Map  	= &h_Map[gpuid*n];			/* set map */
                double *C1 	= &h_C[gpuid*devBuffSize];		/* set Cbuff */

                const int nthreads = 1;

                cudaSetDevice(gpuid / Common->numGPU_parallel);					/* set device */

                repeat_supernode = FALSE;
                nscol3 = 0;

                /* get supernode dimensions */
                s = leaves[leaf_idx];
                node = start_global;
                while (s != EMPTY)
                {
                while (node < end_global && supernode_levels[node] != s)
                    node++;
                if (node < end_global && pending[node-start_global] <= 0 && omp_test_lock(&node_locks[node-start_global]))
                {
                k1 = Super [s];            		/* s contains columns k1 to k2-1 of L */
                k2 = Super [s+1];
                nscol = k2 - k1;           		/* # of columns in all of s */
                psi = Lpi [s];             		/* pointer to first row of s in Ls */
                psx = Lpx [s];             		/* pointer to first row of s in Lx */
                psend = Lpi [s+1];         		/* pointer just past last row of s in Ls */
                nsrow = psend - psi;       		/* # of rows in all of s */
                pend = psx + nsrow * nscol;       	/* s is nsrow-by-nscol */
                pk = psx;

                        printf ("checkpoint -6\n");


                /*
                 *  Initialize Supernode
                 *
                 *  Initializes the supernode with the following steps:
                 *
                 *  1. clear supernode (Lx) on device
                 *  2. create Map for supernode
                 *
                 */
                TEMPLATE2 ( CHOLMOD (gpu_initialize_supernode_root))( Common, gpu_p, nscol, nsrow, psi, psx, gpuid );

                        printf ("checkpoint -5\n");

                /* construct the scattered Map for supernode s */
#pragma omp parallel for num_threads(numThreads) if ( nsrow > 128 )
                for (k = 0 ; k < nsrow ; k++) {
                    Map [Ls [psi + k]] = k ;
                }

                        printf ("checkpoint -4\n");

//#pragma omp critical (head_next)
                {
                    /* reorder descendants in supernode by descreasing size */
                    TEMPLATE2 (CHOLMOD (gpu_reorder_descendants_root))(Common, gpu_p, k1, k2, Ls, Lpi, Lpos, Super, Head, &tail, Next, Previous, &ndescendants, &mapCreatedOnGpu, s, gpuid );

                    for ( d=Head[s]; d!=EMPTY; d=Next[d] ){
                        Next_local[d] = Next[d];
                        Previous_local[d] = Previous[d];
                        Lpos_local[d] = Lpos[d];
                    }

                    for ( d = Head[s]; d != EMPTY; d = Next_local[d] ) {

                        p = Lpos [d] ;          	 	/* offset of 1st row of d affecting s */
                        pdi = Lpi [d] ;         		/* pointer to first row of d in Ls */
                        pdi1 = pdi + p ;        	 	/* ptr to 1st row of d affecting s in Ls */
                        pdend = Lpi [d+1] ;     	 	/* pointer just past last row of d in Ls */
                        for (pdi2 = pdi1 ; pdi2 < pdend && Ls [pdi2] < k2 ; pdi2++) ;
                        ndrow = pdend - pdi ;   	 	/* # rows in all of d */
                        Lpos [d] = pdi2 - pdi ;

                        if (Lpos [d] < ndrow) {
                            dancestor = SuperMap [Ls [pdi2]] ;
#pragma omp critical (head_next)
                            {
                                Next [d] = Head [dancestor] ;
                                Head [dancestor] = d ;
                            }
                        }

                    }

                    /* prepare next supernode */
                    /* Lpos [s] is offset of first row of s affecting its parent */
                    if ( nsrow - nscol > 0 ) {
                        Lpos [s] = nscol ;
                        sparent = SuperMap [Ls [psi + nscol]] ;
                        /* place s in link list of its parent */
#pragma omp critical (head_next)
                        {
                            Next [s] = Head [sparent] ;
                            Head [sparent] = s ;
                        }
                        //Head[s] = EMPTY;
                    }

                } /* end pragma omp critical */

                        printf ("checkpoint -3\n");

                /* copy matrix into supernode s (lower triangular part only) */
#pragma omp parallel for private ( p, pend, pfend, pf, i, j, imap, q ) num_threads(numThreads) if ( k2-k1 > 64 )
                for (k = k1 ; k < k2 ; k++)
                {
                    /* copy the kth column of A into the supernode */
                    if (stype != 0)
                    {
                        p = Ap [k] ;
                        pend = (Apacked) ? (Ap [k+1]) : (p + Anz [k]) ;

                        for ( ; p < pend ; p++)
                        {
                            i = Ai [p] ;
                            if (i >= k)
                            {
                                imap = Map [i] ;					/* row i of L is located in row Map [i] of s */
                                if (imap >= 0 && imap < nsrow)
                                {
                                    L_ASSIGN (Lx,(imap+(psx+(k-k1)*nsrow)), Ax,Az,p) ;	/* Lx [Map [i] + pk] = Ax [p] ; */
                                }
                            }
                        }
                    }
                    /* copy the kth column of A*F into the supernode */
                    else
                    {
                        double fjk[2];
                        pf = Fp [k] ;
                        pfend = (Fpacked) ? (Fp [k+1]) : (p + Fnz [k]) ;
                        for ( ; pf < pfend ; pf++)
                        {
                            j = Fi [pf] ;
                            L_ASSIGN (fjk,0, Fx,Fz,pf) ;				/* fjk = Fx [pf] ; */
                            p = Ap [j] ;
                            pend = (Apacked) ? (Ap [j+1]) : (p + Anz [j]) ;
                            for ( ; p < pend ; p++)
                            {
                                i = Ai [p] ;
                                if (i >= k)
                                {
                                    imap = Map [i] ;
                                    if (imap >= 0 && imap < nsrow)
                                    {
                                        L_MULTADD (Lx,(imap+(psx+(k-k1)*nsrow)),Ax,Az,p, fjk) ;	/* Lx [Map [i] + pk] += Ax [p] * fjk ; */
                                    }
                                }
                            }
                        }
                    }
                }

                        printf ("checkpoint -2\n");

                /* add beta (only real part) to the diagonal of the supernode, if nonzero */
                if (beta [0] != 0.0)
                {
                    pk = psx ;
                    for (k = k1 ; k < k2 ; k++)
                    {
                        L_ASSEMBLE (Lx,pk, beta) ;	/* Lx [pk] += beta [0] ; */
                        pk += nsrow + 1 ;       	/* advance to the next diagonal entry */
                    }
                }
                        printf ("checkpoint -1\n");

                /* save/restore the list of supernodes */
                if (!repeat_supernode)
                {
                    for (d = Head [s] ; d != EMPTY ; d = Next_local [d])
                    {
                        Lpos_save [d] = Lpos_local [d] ;
                        Next_save [d] = Next_local [d] ;
                    }
                }
                else
                {
                    for (d = Head [s] ; d != EMPTY ; d = Next_local [d])
                    {
                        Lpos_local [d] = Lpos_save [d] ;
                        Next_local [d] = Next_save [d] ;
                    }
                }

                        printf ("checkpoint -0\n");

                /* initialize the buffer counter */
                Common->ibuffer[gpuid] = 0;
                supernodeUsedGPU = 0;
                idescendant = 0;
                dlarge = Head[s];
                dsmall = tail;
                GPUavailable = 1;



#ifdef MKLROOT
                mkl_set_num_threads_local(numThreads1);
#else
                openblas_set_num_threads(numThreads1);
#endif


                cuErrHost = cudaSuccess;
#ifdef QUERY_LX_EVENTS
                cuErrDev = cudaSuccess;
#endif

                /* loop over descendants d of supernode s */
                while( (idescendant < ndescendants) )
                {

                    iHostBuff = (Common->ibuffer[gpuid]) % CHOLMOD_HOST_SUPERNODE_BUFFERS;
                    iDevBuff  = (Common->ibuffer[gpuid]) % CHOLMOD_DEVICE_LX_BUFFERS;
                    iDevCBuff  = (Common->ibuffer[gpuid]) % CHOLMOD_DEVICE_C_BUFFERS;

                    Common->ibuffer[gpuid]++;
                    Common->ibuffer[gpuid] = Common->ibuffer[gpuid]%(CHOLMOD_HOST_SUPERNODE_BUFFERS*CHOLMOD_DEVICE_LX_BUFFERS*CHOLMOD_DEVICE_C_BUFFERS*CHOLMOD_DEVICE_STREAMS);


                    /* get next descendant */
                    if ( idescendant > 0 ) {

                        cuErrHost = cudaEventQuery ( Common->updateCBuffersFree[gpuid][iHostBuff] );
#ifdef QUERY_LX_EVENTS
                        cuErrDev = cudaEventQuery ( Common->updateCDevBuffersFree[gpuid][iDevBuff] );
#endif
                        while ( (cuErrHost != cudaSuccess
#ifdef QUERY_LX_EVENTS
                                    || cuErrDev != cudaSuccess
#endif
                                    ) && (ndescendants - idescendant < CHOLMOD_HOST_SUPERNODE_BUFFERS) )
                        {
                            iHostBuff = (Common->ibuffer[gpuid]) % CHOLMOD_HOST_SUPERNODE_BUFFERS;
                            iDevBuff  = (Common->ibuffer[gpuid]) % CHOLMOD_DEVICE_LX_BUFFERS;
                            iDevCBuff  = (Common->ibuffer[gpuid]) % CHOLMOD_DEVICE_C_BUFFERS;

                            Common->ibuffer[gpuid]++;
                            Common->ibuffer[gpuid] = Common->ibuffer[gpuid]%(CHOLMOD_HOST_SUPERNODE_BUFFERS*CHOLMOD_DEVICE_LX_BUFFERS*CHOLMOD_DEVICE_C_BUFFERS*CHOLMOD_DEVICE_STREAMS);

                            cuErrHost = cudaEventQuery ( Common->updateCBuffersFree[gpuid][iHostBuff] );
#ifdef QUERY_LX_EVENTS
                            cuErrDev = cudaEventQuery ( Common->updateCDevBuffersFree[gpuid][iDevBuff] );
#endif
                        }

                    }

                        if ( cuErrHost == cudaSuccess
#ifdef QUERY_LX_EVENTS
                                && cuErrDev == cudaSuccess
#endif
                           )
                        {

                            d = dlarge;
                            dlarge = Next_local[dlarge];

                            GPUavailable = 1;

                        }
                        else {
                            d = dsmall;
                            dsmall = Previous_local[dsmall];
                            GPUavailable = 0;
                        }

                        printf ("checkpoint s = %ld d = %ld idescendant = %ld ndescendants = %ld\n", s, d, idescendant, ndescendants);

                    /* get the size of supernode d */
                    kd1 = Super [d] ;      		/* d contains cols kd1 to kd2-1 of L */
                    kd2 = Super [d+1] ;
                    ndcol = kd2 - kd1 ; 		/* # of columns in all of d */
                    pdi = Lpi [d] ;         		/* pointer to first row of d in Ls */
                    pdx = Lpx [d] ;         	 	/* pointer to first row of d in Lx */
                    pdend = Lpi [d+1] ;     	 	/* pointer just past last row of d in Ls */
                    ndrow = pdend - pdi ;   	 	/* # rows in all of d */

                    /* find the range of rows of d that affect rows k1 to k2-1 of s */
                    p = Lpos_local [d] ;          	/* offset of 1st row of d affecting s */
                    pdi1 = pdi + p ;        	 	/* ptr to 1st row of d affecting s in Ls */
                    pdx1 = pdx + p ;        	 	/* ptr to 1st row of d affecting s in Lx */

                    for (pdi2 = pdi1 ; pdi2 < pdend && Ls [pdi2] < k2 ; (pdi2)++) ;
                    ndrow1 = pdi2 - pdi1 ;      	/* # rows in first part of d */
                    ndrow2 = pdend - pdi1 ;     	/* # rows in remaining d */

                    /* construct the update matrix C for this supernode d */
                    ndrow3 = ndrow2 - ndrow1 ;  	 	/* number of rows of C2 */


                    /*
                     *  Supernode Assembly
                     *
                     *  Assemble the supernode with the following steps:
                     *
                     *  1. perform dsyrk
                     *  2. perform dgemm
                     *  3. perform addUpdate
                     *
                     */
                    if ( GPUavailable == 1 )
                    {
                        printf ("checkpoint 0\n");
                        TEMPLATE2 (CHOLMOD (gpu_updateC_root)) (Common, gpu_p, Lx, ndrow1, ndrow2, ndrow, ndcol, nsrow, pdx1, pdi1, iHostBuff, iDevBuff, iDevCBuff, gpuid);
                        printf ("checkpoint 1\n");
                        supernodeUsedGPU = 1;   				/* GPU was used for this supernode*/
                        idescendant++;
                    }
                    else if ( GPUavailable == 0 )
                    {

                        int tid;

                        nvtxRangeId_t id2 = nvtxRangeStartA("CPU portion");

                        desc_count = 0;
                        syrk_count = 0;
                        gemm_count = 0;
                        counter = 0;

                        printf ("checkpoint 2\n");
                        /* loop over descendants */
                        for(tid = 0; tid < nthreads; tid++)
                        {

                            /* ensure there are remaining descendants to assemble */
                            if(idescendant < ndescendants)
                            {

                                if(tid > 0) {
                                    d = dsmall;
                                    dsmall = Previous_local[dsmall];
                                }

                                {

                                    /* get descendant dimensions */
                                    kd1 = Super [d] ;
                                    kd2 = Super [d+1] ;
                                    ndcol = kd2 - kd1 ;
                                    pdi = Lpi [d] ;
                                    pdx = Lpx [d] ;
                                    pdend = Lpi [d+1] ;
                                    ndrow = pdend - pdi ;

                                    p = Lpos_local [d] ;
                                    pdi1 = pdi + p ;
                                    pdx1 = pdx + p ;

                                    for (pdi2 = pdi1 ; pdi2 < pdend && Ls [pdi2] < k2 ; (pdi2)++);
                                    ndrow1 = pdi2 - pdi1 ;
                                    ndrow2 = pdend - pdi1 ;
                                    ndrow3 = ndrow2 - ndrow1 ;

                                    /* ensure there is sufficient C buffer space to hold Schur complement update */
                                    if ( sizeof(double) * L_ENTRY * (counter + ndrow1*ndrow2) <= Common->devBuffSize )
                                    {

                                        idescendant++;

                                        Int m   = ndrow2-ndrow1;
                                        Int n   = ndrow1;
                                        Int k   = ndcol;
                                        Int lda = ndrow;
                                        Int ldb = ndrow;
                                        Int ldc = ndrow2;
                                        printf ("checkpoint d = %ld lpos = %ld pdi = %ld pdi1 = %ld pdx = %ld pdx1 = %ld m = %ld n = %ld k = %ld lda = %ld ldb = %ld ldc = %ld\n", d, p, pdi, pdi1, pdx, pdx1, m, n, k, lda, ldb, ldc);

                                        /* store descendant dimensions */
                                        desc[desc_count].pdi1   = pdi1;
                                        desc[desc_count].ndrow1 = ndrow1;
                                        desc[desc_count].ndrow2 = ndrow2;
                                        desc[desc_count].C      = (double *)&C1[counter];
                                        desc_count++;

                                        /* store syrk dimensions & pointers */
                                        syrk[syrk_count].n     = n;
                                        syrk[syrk_count].k     = k;
                                        syrk[syrk_count].lda   = lda;
                                        syrk[syrk_count].ldc   = ldc;
                                        syrk[syrk_count].A     = (double *)(Lx + L_ENTRY*pdx1);
                                        syrk[syrk_count].C     = (double *)&C1[counter];
                                        syrk_count++;

                                        if (m > 0)
                                        {
                                        /* store gemm dimensions & pointers */
                                        gemm[gemm_count].m     = m;
                                        gemm[gemm_count].n     = n;
                                        gemm[gemm_count].k     = k;
                                        gemm[gemm_count].lda   = lda;
                                        gemm[gemm_count].ldb   = ldb;
                                        gemm[gemm_count].ldc   = ldc;
                                        gemm[gemm_count].A     = (double *)(Lx + L_ENTRY*(pdx1 + n));
                                        gemm[gemm_count].B     = (double *)(Lx + L_ENTRY*pdx1);
                                        gemm[gemm_count].C     = (double *)(&C1[counter] + L_ENTRY*n);
                                        gemm_count++;
                                        }

                                        /* increment pointer to C buff */
                                        counter += L_ENTRY*n*ldc;

                                    }
                                    else
                                    {
                                        dsmall = d;
                                        tid = nthreads; // ends the loop
                                    }
                                }
                            }
                        } /* end loop over parallel descendants (threads) */

                        printf ("checkpoint 3\n");
                        {
                            int i;
                            /*
                             *  DSYRK
                             *
                             *   Perform dsyrk on batch of descendants
                             *
                             */
                            /* loop over syrk's */
#pragma omp parallel for num_threads(nthreads)
                            for(i = 0; i < syrk_count; i++)
                            {

                                /* get syrk dimensions */
                                Int n   = syrk[i].n;
                                Int k   = syrk[i].k;
                                Int lda = syrk[i].lda;
                                Int ldc = syrk[i].ldc;

                                double *A = (double *)syrk[i].A;
                                double *C = (double *)syrk[i].C;

                                double one[2]  = {1.0, 0.0};
                                double zero[2] = {0.0, 0.0};


#ifdef REAL
                                BLAS_dsyrk ("L", "N",
                                        n, k,
                                        one,
                                        A, lda,
                                        zero,
                                        C, ldc) ;
#else
                                BLAS_zherk ("L", "N",
                                        n, k,
                                        one,
                                        A, lda,
                                        zero,
                                        C, ldc) ;
#endif
                            } /* end loop over syrk's */





                            /*
                             *  DGEMM
                             *
                             *  Perform dgemm on batch of descendants
                             *
                             */
                            /* loop over gemm's */
#pragma omp parallel for num_threads(nthreads)
                            for(i = 0; i < gemm_count; i++)
                            {
                                /* get gemm dimensions */
                                Int m   = gemm[i].m;
                                Int n   = gemm[i].n;
                                Int k   = gemm[i].k;
                                Int lda = gemm[i].lda;
                                Int ldb = gemm[i].ldb;
                                Int ldc = gemm[i].ldc;

                                double *A = (double *)gemm[i].A;
                                double *B = (double *)gemm[i].B;
                                double *C = (double *)gemm[i].C;

                                double one[2] = {1.0, 0.0};
                                double zero[2] = {0.0, 0.0};


                                if (m > 0)
                                {
#ifdef REAL
                                    BLAS_dgemm ("N","T",
                                            m, n, k,
                                            one,
                                            A, lda,
                                            B, ldb,
                                            zero,
                                            C, ldc) ;
#else
                                    BLAS_zgemm ("N", "C",
                                            m, n, k,
                                            one,
                                            A, lda,
                                            B, ldb,
                                            zero,
                                            C, ldc) ;
#endif

                                }
                            } /* end loop over gemm's */





                            /*
                             *  Assembly
                             *
                             *  Assemble schur complements of a batch of descendants
                             *
                             */
                            /* loop over descendants */
                            for(i = 0; i < desc_count; i++)
                            {

                                Int ii, j, q, px;


                                /* get descendant dimensions */
                                Int pdi1 = desc[i].pdi1;
                                Int ndrow1 = desc[i].ndrow1;
                                Int ndrow2 = desc[i].ndrow2;

                                double *C = (double *)desc[i].C;


#pragma omp parallel for private ( j, ii, px, q ) num_threads(numThreads1) if (ndrow1 > 64 )
                                for (j = 0 ; j < ndrow1 ; j++)
                                {
                                    px = psx + Map [Ls [pdi1 + j]]*nsrow ;
                                    for (ii = j ; ii < ndrow2 ; ii++)
                                    {
                                        q = px + Map [Ls [pdi1 + ii]] ;
                                        L_ASSEMBLESUB (Lx,q, C, ii+ndrow2*j) ;
                                    }
                                }
                            } /* end loop over descendants */
                        }
                        printf ("checkpoint 4\n");

                        nvtxRangeEnd(id2);


                    }

                } /* end loop over descendants */






                /*
                 *  Final Supernode Assembly
                 *
                 *  Sum CPU and GPU assembly's of supernode:
                 *
                 */
                iHostBuff = (Common->ibuffer[gpuid])%CHOLMOD_HOST_SUPERNODE_BUFFERS;
                iDevBuff = (Common->ibuffer[gpuid])%CHOLMOD_DEVICE_LX_BUFFERS;
                TEMPLATE2 ( CHOLMOD (gpu_final_assembly_root ))( Common, gpu_p, Lx, psx, nscol, nsrow, supernodeUsedGPU, gpuid );




                /*
                 *  Cholesky Factorization
                 *
                 *  Factorize diagonal block of spuernode s in LL' in the following steps:
                 *  1. perform dpotrf
                 *
                 */
                nscol2 = (repeat_supernode) ? (nscol3) : (nscol) ;
                if ( !(supernodeUsedGPU) || !TEMPLATE2 (CHOLMOD (gpu_lower_potrf_root))(Common, gpu_p, Lx, &info, nscol2, nsrow, psx, gpuid))
                {

                    supernodeUsedGPU = 0;

#ifdef REAL
                    LAPACK_dpotrf ("L",
                            nscol2,                    	/* N: nscol2 */
                            Lx + L_ENTRY*psx, nsrow,    	/* A, LDA: S1, nsrow */
                            info) ;                    	/* INFO */
#else
                    LAPACK_zpotrf ("L",
                            nscol2,                     /* N: nscol2 */
                            Lx + L_ENTRY*psx, nsrow,    /* A, LDA: S1, nsrow */
                            info) ;                     /* INFO */
#endif


                }




                /* check if the matrix is not positive definite */
                if (repeat_supernode)
                {
                    /* the leading part has been refactorized; it must have succeeded */
                    info = 0 ;

                    /* zero out the rest of this supernode */
                    p = psx + nsrow * nscol3 ;
                    pend = psx + nsrow * nscol ;
                    for ( ; p < pend ; p++)
                    {
                        L_CLEAR (Lx,p) ;				/* Lx [p] = 0 ; */
                    }
                }




                /* info is set to one in LAPACK_*potrf if blas_ok is FALSE. It is
                 * set to zero in dpotrf/zpotrf if the factorization was successful. */
                if (CHECK_BLAS_INT && !Common->blas_ok)
                {
                    ERROR (CHOLMOD_TOO_LARGE, "problem too large for the BLAS") ;
                }




                /* check if the matrix is not positive definite */
                if (info != 0)
                {

                    /* Matrix is not positive definite. dpotrf/zpotrf do NOT report an
                     * error if the diagonal of L has NaN's, only if it has a zero.
                     */
                    if (Common->status == CHOLMOD_OK)
                    {
                        ERROR (CHOLMOD_NOT_POSDEF, "matrix not positive definite") ;
                    }


                    /* L->minor is the column of L that contains a zero or negative
                     * diagonal term.
                     */
                    L->minor = k1 + info - 1 ;


                    /* clear the link lists of all subsequent supernodes */
                    for (ss = s+1 ; ss < L->nsuper ; ss++)
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
                     * where the 1-based p is identical to the 0-based L->minor. Since
                     * L->minor is in the current supernode s, it and any columns to the
                     * left of it in supernode s are also all zero. This differs from
                     * [R,p]=chol(A), which contains nonzero rows 1 to p-1. Fix this
                     * by setting repeat_supernode to TRUE, and repeating supernode s.
                     *
                     * If Common->quick_return_if_not_posdef is true, then the entire
                     * supernode s is not factorized; it is left as all zero.
                     */
                    if (info == 1 || Common->quick_return_if_not_posdef)
                    {
                        /* If the first column of supernode s contains a zero or
                         * negative diagonal entry, then it is already properly set to
                         * zero. Also, info will be 1 if integer overflow occured in
                         * the BLAS. */
                        Head [s] = EMPTY ;

                        /* finalize GPU */
                        CHOLMOD (gpu_end) (Common);
                        /*return Common->status;*/

                    }
                    else
                    {
                        /* Repeat supernode s, but only factorize it up to but not
                         * including the column containing the problematic diagonal
                         * entry. */
                        repeat_supernode = TRUE ;
                        //s-- ;
                        nscol3 = info - 1 ;
                    }

                } /* end if info */


                if (!repeat_supernode)
                {


                    /*
                     *  Triangular Solve
                     *
                     *  Compute the subdiagonal block in the following steps:
                     *  1. perform dtrsm
                     *  2. copy result back into factor Lx
                     *  3. prepare next supernode
                     *
                     */
                    nsrow2 = nsrow - nscol2 ;
                    if (nsrow2 > 0)
                    {
                        /* The current supernode is columns k1 to k2-1 of L. Let L1 be the
                         * diagonal block (factorized by dpotrf/zpotrf above; rows/cols
                         * k1:k2-1), and L2 be rows k2:n-1 and columns k1:k2-1 of L. The
                         * triangular system to solve is L2*L1' = S2, where S2 is
                         * overwritten with L2. More precisely, L2 = S2 / L1' in MATLAB
                         * notation.
                         */
                        if ( !(supernodeUsedGPU) || !TEMPLATE2 (CHOLMOD(gpu_triangular_solve_root)) (Common, gpu_p, Lx, nsrow2, nscol2, nsrow, psx ,gpuid) )
                        {
#ifdef REAL
                            BLAS_dtrsm ("R", "L", "T", "N",
                                    nsrow2, nscol2,                 	/* M, N */
                                    one,                            	/* ALPHA: 1 */
                                    Lx + L_ENTRY*psx, nsrow,        	/* A, LDA: L1, nsrow */
                                    Lx + L_ENTRY*(psx + nscol2),    	/* B, LDB, L2, nsrow */
                                    nsrow) ;
#else
                            BLAS_ztrsm ("R", "L", "C", "N",
                                    nsrow2, nscol2,                 /* M, N */
                                    one,                            /* ALPHA: 1 */
                                    Lx + L_ENTRY*psx, nsrow,        /* A, LDA: L1, nsrow */
                                    Lx + L_ENTRY*(psx + nscol2),    /* B, LDB, L2, nsrow */
                                    nsrow) ;
#endif


                        }


                        if (CHECK_BLAS_INT && !Common->blas_ok)
                        {
                            ERROR (CHOLMOD_TOO_LARGE, "problem too large for the BLAS") ;
                        }

                    }
                    /* copy supernode back to factor Lx anyways */
                    else
                    {
                        TEMPLATE2 ( CHOLMOD (gpu_copy_supernode_root) )( Common, gpu_p, Lx, psx, nscol, nscol2, nsrow, supernodeUsedGPU, gpuid);
                    }

                    Head [s] = EMPTY ; /* link list for supernode s no longer needed */


                    if (repeat_supernode)
                    {
                        /* matrix is not positive definite; finished clean-up for supernode
                         * containing negative diagonal */
                        /*return Common->status;*/
                    }

                    /* Mark the supernode complete */
                    sparent = SuperMap [Ls [psi + nscol]];
                    if (sparent > s && sparent < L->nsuper)
                    {
                        Int inode;
                        inode = node;
                        while (inode < end_global && supernode_levels[inode] != sparent)
                            inode++;
                        if (inode < end_global)
                        {
#pragma omp atomic
                            pending[inode-start_global]--;
                            s = sparent;
                        }
                        else
                            s = EMPTY;
                    }
                }
            }
                else
                {
                    s = EMPTY;
                }
                }

            } /* end loop over supenodes */
        }

        free ( Next_local );
        free ( Lpos_local );

        free ( pending );
        free ( leaves );
        for ( node=0; node<event_len; node++ ) {
            omp_destroy_lock(&node_locks[node]);
        }
        free ( node_locks );

    } /* end loop over levels */


#endif

    /* return ok */
    return Common->status;/*(Common->status >= CHOLMOD_OK) ;*/

}


/*
#undef REAL
#undef COMPLEX
#undef ZOMPLEX
*/











