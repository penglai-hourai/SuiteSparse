/* ========================================================================== */
/* === GPU/t_factorize_subtree.c ============================================ */
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
 *   t_factorize_subtree
 *
 * Description:
 *   Contains functions for factorization
 *   of the GPU subtree algorithm.
 *
 */


/* includes */
#include <string.h>
#include <time.h>


/*
 * Function:
 *   gpu_factorize_subtree
 *
 * Description:
 *   Factorizes subtrees of the elimination tree on the GPU.
 *
 */
void TEMPLATE2 (CHOLMOD (gpu_factorize_subtree))
  (
    cholmod_common *Common,
    cholmod_global_pointers *gb_p,
    cholmod_gpu_pointers *gpu_p,
    cholmod_cpu_pointers *cpu_p,
    cholmod_tree_pointers *tree_p,
    cholmod_profile_pointers *prof_p,
    cholmod_factor *L,
    int deviceid,
    Int subtree,
    Int *LpxSub
  )
{

  cudaDeviceSynchronize();

#ifdef SUITESPARSE_CUDA
  /* local variables */
  Int    i, j, syrk_count, gemm_count, potrf_count, trsm_count, update_count, desc_count, super_count, count, count_prev, nbatch, level, gpuid;
  Int    strideSize, maxDim1[3] = {0,0,0};
  Int	 *d_dimSuper, *d_dimDesc, *h_dimSuper, *h_dimDesc;
  Int    sparent, p, s, d, n, k1, k2, nscol, psi, psx, psend, nsrow, nsrow2, kd1, kd2, pdi, pdi1, pdi2, pdx, pdx1, pdend, ndcol, ndrow, ndrow1, ndrow2,
         ndrow3, spsend, dancestor, dlarge, idescendant, node_ptr, node, start, end, diff, maxbatch, maxnsrow, maxkdif, maxnz, maxnsrownscol;
  Int    *Ls, *Lpi, *Lpx, *Lpos, *Lpos_save, *Ap, *Super, *SuperMap, *Head, *Next, *factor_size, *ndescendants, *supernode_batch, *supernode_levels, *supernode_levels_ptrs,
         *supernode_levels_subtree_ptrs, *supernode_num_levels, *level_num_desc, *level_num_desc_ptrs, *level_descendants, *level_descendants_ptrs;
  double *devPtrC, *d_C, *d_Lx, *tstart, *tend, *syrk_time, *gemm_time, *potrf_time, *trsm_time, *syrk_flops, *gemm_flops, *potrf_flops, *trsm_flops;
  double **d_ptrSuper, **h_ptrSuper, **d_ptrDesc, **h_ptrDesc;
  double tstart1;

  struct cholmod_syrk_ptrs_t 	*h_syrk;
  struct cholmod_gemm_ptrs_t 	*h_gemm;
  struct cholmod_potrf_ptrs_t 	*h_potrf;
  struct cholmod_trsm_ptrs_t 	*h_trsm;
  struct cholmod_desc_ptrs_t 	*h_desc;
  struct cholmod_super_ptrs_t 	*h_super;
  struct cholmod_syrk_t 	syrk[gb_p->maxndesc];
  struct cholmod_gemm_t 	gemm[gb_p->maxndesc];
  struct cholmod_potrf_t 	potrf[gb_p->maxbatch];
  struct cholmod_trsm_t 	trsm[gb_p->maxbatch];
  struct cholmod_desc_t 	desc[gb_p->maxndesc];
  struct cholmod_super_t 	super[gb_p->maxbatch];

  int vgpuid;

  Int d_itr;
  Int update_count_factorized, update_count_factorized_small;
  struct cholmod_descendant_score_t *update_factorized = gpu_p->h_darray[deviceid];

  const double minus_one = -1;
  const double zero = 0;


  cudaSetDevice(deviceid);


  /*
   * Set variables & pointers
   */
  /* set host variables */
  n 		= L->n;
  gpuid		= deviceid;
  maxbatch 	= gb_p->maxbatch;

  /* set host pointers */
  Ls 		= cpu_p->Ls;
  Lpi 		= cpu_p->Lpi;
  Lpx 		= cpu_p->Lpx;
  Lpos 		= cpu_p->Lpos;
  Lpos_save = cpu_p->Lpos_save;
  Ap		= cpu_p->Ap;
  Super 	= cpu_p->Super;
  SuperMap 	= cpu_p->SuperMap;
  Head 		= cpu_p->Head;
  Next 		= cpu_p->Next;
  h_dimSuper    = gpu_p->h_dimSuper[gpuid];
  h_ptrSuper    = gpu_p->h_ptrSuper[gpuid];
  h_dimDesc     = gpu_p->h_dimDesc[gpuid];
  h_ptrDesc     = gpu_p->h_ptrDesc[gpuid];
  h_syrk    = &gpu_p->h_syrk[gpuid];
  h_gemm    = &gpu_p->h_gemm[gpuid];
  h_potrf   = &gpu_p->h_potrf[gpuid];
  h_trsm    = &gpu_p->h_trsm[gpuid];
  h_desc    = &gpu_p->h_desc[gpuid];
  h_super   = &gpu_p->h_super[gpuid];

  /* set tree pointers */
  factor_size			= tree_p->factor_size;
  ndescendants                  = tree_p->ndescendants;
  supernode_batch               = tree_p->supernode_batch;
  supernode_levels              = tree_p->supernode_levels;
  supernode_levels_ptrs         = tree_p->supernode_levels_ptrs;
  supernode_levels_subtree_ptrs  = tree_p->supernode_levels_subtree_ptrs;
  supernode_num_levels          = tree_p->supernode_num_levels;
  level_num_desc		= tree_p->level_num_desc;
  level_num_desc_ptrs		= tree_p->level_num_desc_ptrs;
  level_descendants		= tree_p->level_descendants;
  level_descendants_ptrs	= tree_p->level_descendants_ptrs;

  /* set gpu pointers */
  d_Lx 		= gpu_p->d_Lx[gpuid];
  d_C		= gpu_p->d_C[gpuid];
  d_dimSuper	= gpu_p->d_dimSuper[gpuid];
  d_ptrSuper	= gpu_p->d_ptrSuper[gpuid];
  d_dimDesc     = gpu_p->d_dimDesc[gpuid];
  d_ptrDesc     = gpu_p->d_ptrDesc[gpuid];

  /* set timer pointers */
  tstart	= prof_p->f_start[deviceid];
  tend		= prof_p->f_end[deviceid];
  syrk_time	= prof_p->syrk_time[deviceid];
  gemm_time	= prof_p->gemm_time[deviceid];
  potrf_time	= prof_p->potrf_time[deviceid];
  trsm_time	= prof_p->trsm_time[deviceid];
  syrk_flops	= prof_p->syrk_flop[deviceid];
  gemm_flops	= prof_p->gemm_flop[deviceid];
  potrf_flops	= prof_p->potrf_flop[deviceid];
  trsm_flops	= prof_p->trsm_flop[deviceid];

  /* clear global flops */
  CLEAR1(syrk_flops,0);
  CLEAR1(gemm_flops,0);
  CLEAR1(potrf_flops,0);
  CLEAR1(trsm_flops,0);
  PRINTF("\n\nlevel:\tbatch:\tnsuper:\tsyrkTime:\tgemmTime:\tpotrfTime:\ttrsmTime:\tsyrkFlops:\tgemmFlops:\tpotrfFlops:\ttrsmFlops:\n");





  //printf ("# nodes in subtree = %ld\n", supernode_levels_ptrs[supernode_levels_subtree_ptrs[subtree]+supernode_num_levels[subtree]] - supernode_levels_ptrs[supernode_levels_subtree_ptrs[subtree]]);
  /* loop over levels in subtree */
  for(level = 0; level < supernode_num_levels[subtree]; level++) {


  #ifdef USE_NVTX
    char subtreestrng[0];
    sprintf (subtreestrng, "level %d ", level);
    nvtxRangeId_t range1 = nvtxRangeStartA (subtreestrng);
  #endif

    /* set batching parameters */
    start = supernode_levels_ptrs[supernode_levels_subtree_ptrs[subtree]+level];          /* starting supernode of level */
    end   = supernode_levels_ptrs[supernode_levels_subtree_ptrs[subtree]+level+1];        /* ending supernode of level */
    diff  = (end - start);                   	                                        /* # supernodes in level */
    strideSize = level_num_desc[level_num_desc_ptrs[subtree]+level];	 	        /* largest number of descendants in a batch in current level */
    nbatch = supernode_batch[supernode_levels_subtree_ptrs[subtree] + level];		/* size of batch for current level */
    node = 0;

    /* clear local flops */
    CLEAR1(syrk_flops,1);
    CLEAR1(gemm_flops,1);
    CLEAR1(potrf_flops,1);
    CLEAR1(trsm_flops,1);


    /* loop over groups of batches */
    while(node < (end - start)) {


      /* check if batch overexceeds */
      if( (node + nbatch) > (end-start) )
        nbatch = (end-start) - node;


      TIMER_START(tstart,0);





      /* clear counters */
      super_count 	= 0;
      desc_count 	= 0;
      syrk_count 	= 0;
      gemm_count 	= 0;
      potrf_count 	= 0;
      trsm_count 	= 0;
      update_count 	= 0;

      /* clear max variables */
      maxnsrownscol 	= 0;
      maxnsrow 		= 0;
      maxkdif 		= 0;
      maxnz 		= 0;
      maxDim1[0] = 0;
      maxDim1[1] = 0;
      maxDim1[2] = 0;











      /* set pointers for dimensions of batch supernode */
      h_potrf->n   	= h_dimSuper + 0*maxbatch;
      h_potrf->lda 	= h_dimSuper + 1*maxbatch;
      h_trsm->m   	= h_dimSuper + 2*maxbatch;
      h_trsm->n   	= h_dimSuper + 3*maxbatch;
      h_trsm->lda 	= h_dimSuper + 4*maxbatch;
      h_trsm->ldb 	= h_dimSuper + 5*maxbatch;
      h_super->s     	= h_dimSuper + 6*maxbatch;
      h_super->k1    	= h_dimSuper + 7*maxbatch;
      h_super->k2    	= h_dimSuper + 8*maxbatch;
      h_super->psi   	= h_dimSuper + 9*maxbatch;
      h_super->psx   	= h_dimSuper + 10*maxbatch;
      h_super->nscol 	= h_dimSuper + 11*maxbatch;
      h_super->nsrow 	= h_dimSuper + 12*maxbatch;

      /* set pointers for pointers of batch supernode */
      h_potrf->A 	= h_ptrSuper + 0*maxbatch;
      h_trsm->A 	= h_ptrSuper + 1*maxbatch;
      h_trsm->B 	= h_ptrSuper + 2*maxbatch;

      /* set pointers for dimensions of batch descendants */
      h_syrk->n      	= h_dimDesc + 0*strideSize;
      h_syrk->k      	= h_dimDesc + 1*strideSize;
      h_syrk->lda    	= h_dimDesc + 2*strideSize;
      h_syrk->ldc    	= h_dimDesc + 3*strideSize;
      h_gemm->m	     	= h_dimDesc + 4*strideSize;
      h_gemm->n      	= h_dimDesc + 5*strideSize;
      h_gemm->k      	= h_dimDesc + 6*strideSize;
      h_gemm->lda    	= h_dimDesc + 7*strideSize;
      h_gemm->ldb    	= h_dimDesc + 8*strideSize;
      h_gemm->ldc    	= h_dimDesc + 9*strideSize;
      h_desc->ndrow1 	= h_dimDesc + 10*strideSize;
      h_desc->ndrow2 	= h_dimDesc + 11*strideSize;
      h_desc->pdi1   	= h_dimDesc + 12*strideSize;
      h_desc->s      	= h_dimDesc + 13*strideSize;

      /* set pointers for pointers of batch descendants */
      h_syrk->A 	= h_ptrDesc + 0*strideSize;
      h_syrk->C 	= h_ptrDesc + 1*strideSize;
      h_gemm->A 	= h_ptrDesc + 2*strideSize;
      h_gemm->B 	= h_ptrDesc + 3*strideSize;
      h_gemm->C 	= h_ptrDesc + 4*strideSize;
      h_desc->C 	= h_ptrDesc + 5*strideSize;










      /*
       *  Store supernode dimensions
       *
       *  Stores dimensions of supernodes in the following order:
       *    1. potrf's to be sent to cuBlas
       *    2. trsm's to be sent to cuBlas
       *    3. potrf's & trsm's to be batched
       *
       *  The dimensions are stored in lists, in the order specified.
       */


      TIMER_START(tstart,1);

      /*
       * Gather dimensions of all supernodes
       */
      /* loop over batch of supernodes */
      for(i = 0; i < nbatch; i++) {

        node_ptr = start + node + i;
        s = supernode_levels[node_ptr];

        /* get supernode dimensions */
        k1 	= Super [s] ;
        k2 	= Super [s+1] ;
        nscol 	= k2 - k1 ;
        psi 	= Lpi [s] ;
        psx 	= LpxSub [s] ;
        psend 	= Lpi [s+1] ;
        nsrow 	= psend - psi ;
        nsrow2  = nsrow - nscol ;

        /* store maximum supernode dimension (for initLx & initMap & copyLx) */
        if(nsrow > maxnsrow) maxnsrow = nsrow;
        if(nscol > maxkdif) maxkdif = nscol;
        if(nscol*nsrow > maxnsrownscol) maxnsrownscol = nscol*nsrow;
        Int kk;
        for (kk = k1 ; kk < k2; kk++){
          Int Apdiff = Ap[kk+1]-Ap[kk];
          if(Apdiff > maxnz) maxnz = Apdiff;
        }


        Int m   = nsrow - nscol ;
        Int n   = nscol;
        Int lda = nsrow;
        Int ldb = nsrow;

        /* store supernode dimensions */
        super[super_count].s 	  = s;
        super[super_count].k1 	  = k1;
        super[super_count].k2 	  = k2;
        super[super_count].psi    = psi;
        super[super_count].psx    = psx;
        super[super_count].nscol  = nscol;
        super[super_count].nsrow  = nsrow;

        /* store potrf dimensions & pointers */
        potrf[super_count].score  = (double)(n);
        potrf[super_count].n 	  = n;
        potrf[super_count].lda    = lda;
        potrf[super_count].A 	  = (double *)(d_Lx + psx);

        /* store trsm dimensions & pointers */
        trsm[super_count].score   = (double)(m*n);
        trsm[super_count].m 	  = m;
        trsm[super_count].n 	  = n;
        trsm[super_count].lda 	  = lda;
        trsm[super_count].ldb 	  = ldb;
        trsm[super_count].A 	  = (double *)(d_Lx + psx);
        trsm[super_count].B 	  = (double *)(d_Lx + psx + n);

        /* increment flops */
        SUM(potrf_flops,1,(double)(n*n*n/3.0));
        SUM(trsm_flops,1,(double)(m*n*n));

        super_count++;

        for (sparent = tree_p->supernode_parent[s]; sparent != EMPTY; sparent = tree_p->supernode_parent[sparent])
        {
#pragma omp atomic
            tree_p->supernode_size[sparent] -= tree_p->supernode_size[s];
        }
        tree_p->supernode_size[s] = 0;

      } /* end loop over supernodes */

      //super_count = nbatch; // not necessary, (super_count == nbatch) anyway




      /*
       * Sort supernodes in descending order
       */
      /*
      qsort ( potrf, super_count, sizeof(struct cholmod_potrf_t), (__compar_fn_t) CHOLMOD(sort_potrf) );
      qsort ( trsm, super_count, sizeof(struct cholmod_trsm_t), (__compar_fn_t) CHOLMOD(sort_trsm) );
      */




      /*
       * Store dimensions of all supernodes in lists
       */

      /* store potrf dimensions & pointers */
      j = 0;
      for(i = 0; i < super_count; i++)
      {
        Int n = potrf[i].n;

        if(n < 64) continue;
        h_potrf->n[j]         = potrf[i].n;
        h_potrf->lda[j]       = potrf[i].lda;
        h_potrf->A[j]         = potrf[i].A;
        j++;
      }
      potrf_count = j;
      for(i = 0; i < super_count; i++)
      {
        Int n = potrf[i].n;

        if(n >= 64) continue;
        h_potrf->n[j]         = potrf[i].n;
        h_potrf->lda[j]       = potrf[i].lda;
        h_potrf->A[j]         = potrf[i].A;
        j++;
      }


      /* store trsm dimensions & pointers */
      j = 0;
      for(i = 0; i < super_count; i++)
      {
        Int m = trsm[i].m;
        Int n = trsm[i].n;

        if(m < 64 && n < 64) continue;
        h_trsm->m[j]           = trsm[i].m;
        h_trsm->n[j]           = trsm[i].n;
        h_trsm->lda[j]         = trsm[i].lda;
        h_trsm->ldb[j]         = trsm[i].ldb;
        h_trsm->A[j]           = trsm[i].A;
        h_trsm->B[j]           = trsm[i].B;
        j++;
      }
      trsm_count = j;
      for(i = 0; i < super_count; i++)
      {
        Int m = trsm[i].m;
        Int n = trsm[i].n;

        if(m >= 64 || n >= 64) continue;
        h_trsm->m[j]           = trsm[i].m;
        h_trsm->n[j]           = trsm[i].n;
        h_trsm->lda[j]         = trsm[i].lda;
        h_trsm->ldb[j]         = trsm[i].ldb;
        h_trsm->A[j]           = trsm[i].A;
        h_trsm->B[j]           = trsm[i].B;
        j++;
      }


      /* store supernode dimensions */
      for(i = 0; i < super_count; i++)
      {
        h_super->s[i]         = super[i].s;
        h_super->k1[i]        = super[i].k1;
        h_super->k2[i]        = super[i].k2;
        h_super->psi[i]       = super[i].psi;
        h_super->psx[i]       = super[i].psx;
        h_super->nscol[i]     = super[i].nscol;
        h_super->nsrow[i]     = super[i].nsrow;
      } /* end loop over supernodes */

      TIMER_END(tstart,tend,1);










      /*
       *  Store descendant dimensions:
       *
       *  Stores dimensions of descendants in the following order:
       *    1. syrk's to be sent to cuBlas
       *    2. gemm's to be sent to cuBlas
       *    3. syrk's & gemm's to be batched
       *
       *  First gather all dimensions and sort them by size. Then
       *  loop over ordered list of descendants and store its dimensions
       *  into lists, in the order specified.
       *
       */


       TIMER_START(tstart,2);

//#pragma omp critical (head_next)
{

      /*
       * Gather dimensions of all descendants
       */
      devPtrC = (double *)(d_C);

      update_count_factorized = 0;
      update_count_factorized_small = 0;

      /* loop over batch of supernodes */
      for(i = 0; i < nbatch; i++) {

        s = h_super->s[i];
        k2 = h_super->k2[i];
        nsrow2 = h_super->nsrow[i] - h_super->nscol[i];
        nscol = h_super->nscol[i];
        psi = h_super->psi[i];

        idescendant = 0;
        dlarge = Head[s];

        /* loop over descendants */
        //while (dlarge != EMPTY)
        while(idescendant < ndescendants[s])
        {

            d = dlarge;
            dlarge = Next[dlarge];

          /* get descendant dimensions */
          kd1 	= Super [d] ;
          kd2 	= Super [d+1] ;
          ndcol = kd2 - kd1 ;
          pdi 	= Lpi [d] ;
          pdx 	= LpxSub [d] ;
          pdend = Lpi [d+1] ;
          ndrow = pdend - pdi ;

          if (tree_p->factorized[d] == -1)
          {
              dancestor = SuperMap [Ls [pdi + Lpos[d]]];
              while (dancestor != EMPTY && Lpos[d] < ndrow)
              {
#pragma omp atomic
                  ndescendants[dancestor]--;

                  for (pdi2 = pdi + Lpos[d]; pdi2 < pdend && Ls [pdi2] < Super[dancestor+1]; pdi2++);
                  Lpos[d] = pdi2 - pdi;
                  dancestor = SuperMap [Ls [pdi + Lpos[d]]];
              }
          }
          else if (tree_p->factorized[d] == 1)
          {
              Lpos_save[d] = Lpos[d];

              dancestor = SuperMap [Ls [pdi + Lpos[d]]];
              while (dancestor != EMPTY && LpxSub[dancestor] >= 0 && Lpos[d] < ndrow)
              {
#pragma omp atomic
                  ndescendants[dancestor]--;

                  for (pdi2 = pdi + Lpos[d]; pdi2 < pdend && Ls [pdi2] < Super[dancestor+1]; pdi2++);
                  Lpos[d] = pdi2 - pdi;
                  dancestor = SuperMap [Ls [pdi + Lpos[d]]];
              }

              if (dancestor != EMPTY && Lpos[d] < ndrow)
              {
#pragma omp critical (head_next)
                  {
                      Next [d] = Head [dancestor] ;
                      Head [dancestor] = d ;
                  }
                  tree_p->factorized[d] = 1;
              }
              else
              {
                  tree_p->factorized[d] = -1;
              }

              if (pdend - (pdi + Lpos_save[d])> 0)
              {
                  update_factorized[update_count_factorized].d = d;
                  if ((pdend - (pdi + Lpos_save[d])) <= UPDATE_BATCH_DIMENSION_M && (pdi2 - (pdi + Lpos_save[d])) <= UPDATE_BATCH_DIMENSION_N && ndcol <= UPDATE_BATCH_DIMENSION_K)
                  {
                      update_factorized[update_count_factorized].score = ndcol * (pdend - (pdi + Lpos_save[d]));
                      update_count_factorized_small++;
                  }
                  else
                  {
                      update_factorized[update_count_factorized].score = - ndcol * (pdend - (pdi + Lpos_save[d]));
                  }
                  update_count_factorized++;
              }
          }
          else
          {
          idescendant++;

          p 	= Lpos[d] ;
          pdi1 	= pdi + p ;
          pdx1 	= pdx + p ;

          for (pdi2 = pdi1; pdi2 < pdend && Ls [pdi2] < k2; pdi2++) ;
          ndrow1 = pdi2 - pdi1 ;
          ndrow2 = pdend - pdi1 ;
          ndrow3 = ndrow2 - ndrow1 ;

          Int m   = ndrow2-ndrow1;
          Int n   = ndrow1;
          Int k   = ndcol;
          Int lda = ndrow;
          Int ldb = ndrow;
          Int ldc = ndrow2;

          /* store descendant dimensions & pointers (for addUpdate) */
          desc[desc_count].score 	= (double)(ndrow1*ndrow2);
          desc[desc_count].s 		= i;
          desc[desc_count].pdi1 	= pdi1;
          desc[desc_count].pdx1 	= pdx1;
          desc[desc_count].ndrow1 	= ndrow1;
          desc[desc_count].ndrow2 	= ndrow2;
          desc[desc_count].C 		= (double *)(devPtrC);

          /* store syrk dimensions & pointers */
          syrk[desc_count].score 	= (double)(n*k);
          syrk[desc_count].n 		= n;
          syrk[desc_count].k 		= k;
          syrk[desc_count].lda 		= lda;
          syrk[desc_count].ldc 		= ldc;
          syrk[desc_count].A 		= (double *)(d_Lx + pdx1);
          syrk[desc_count].C 		= (double *)(devPtrC);

          /* store gemm dimensions & pointers */
          gemm[desc_count].score 	= (double)(2.0*m*n*k);
          gemm[desc_count].m 		= m;
          gemm[desc_count].n 		= n;
          gemm[desc_count].k 		= k;
          gemm[desc_count].lda 		= lda;
          gemm[desc_count].ldb 		= ldb;
          gemm[desc_count].ldc 		= ldc;
          gemm[desc_count].A 		= (double *)(d_Lx + pdx1 + ndrow1);
          gemm[desc_count].B 		= (double *)(d_Lx + pdx1);
          gemm[desc_count].C 		= (double *)(devPtrC + ndrow1);

          /* update flops */
          SUM(syrk_flops,1,(double)(n*n*k));
          SUM(gemm_flops,1,(double)(2*m*n*k));

          devPtrC+=ndrow1*ndrow2;
          desc_count++;

          /* prepare for next descendant */
          Lpos [d] = pdi2 - pdi ;

          if (Lpos [d] < ndrow) {
              dancestor = SuperMap [Ls [pdi2]] ;
#pragma omp critical (head_next)
              {
                  Next [d] = Head [dancestor] ;
                  Head [dancestor] = d ;
              }
              tree_p->factorized[d] = 2;
          }
          else
          {
              tree_p->factorized[d] = -2;
          }
          }

        } /* end loop over descendants */

        /* prepare for next supernode */
        if(nsrow2 > 0) {
          Lpos [s] = nscol;
          sparent = SuperMap [Ls [psi + nscol]] ;
#pragma omp critical (head_next)
          {
              Next [s] = Head [sparent] ;
              Head [sparent] = s ;
          }
              tree_p->factorized[s] = 2;
        }
        else
        {
              tree_p->factorized[s] = -2;
        }
        //Head [s] = EMPTY ;


      } /* end loop over supernodes */

}/* end pragma omp critical */




      /*
       * Sort descendants in descending order
       */
      /*
      qsort ( desc, desc_count, sizeof(struct cholmod_desc_t), (__compar_fn_t) CHOLMOD(sort_desc) );
      qsort ( syrk, desc_count, sizeof(struct cholmod_syrk_t), (__compar_fn_t) CHOLMOD(sort_syrk) );
      qsort ( gemm, desc_count, sizeof(struct cholmod_gemm_t), (__compar_fn_t) CHOLMOD(sort_gemm) );
      */


    if (update_count_factorized > 0)
    {
        qsort (update_factorized, update_count_factorized, sizeof(struct cholmod_descendant_score_t), (__compar_fn_t) CHOLMOD(score_comp));
        {
            Int idx_l, idx_h;
            idx_l = 0;
            idx_h = 1;
            while (idx_h < update_count_factorized)
            {
                while (idx_h < update_count_factorized && update_factorized[idx_l].d == update_factorized[idx_h].d)
                    idx_h++;
                if (idx_h < update_count_factorized)
                {
                    idx_l++;
                    if (idx_l < idx_h)
                        update_factorized[idx_l] = update_factorized[idx_h];
                }
            }
            idx_l++;
            update_count_factorized = idx_l;
            update_count_factorized_small = 0;
            for (idx_h = 0; idx_h < update_count_factorized; idx_h++)
                if (update_factorized[idx_h].score > 0)
                    update_count_factorized_small++;
        }
    }


      /*
       * Store dimensions of all descendants in lists
       */

      /* store syrk dimensions & pointers */
      j = 0;
      for(i = 0; i < desc_count; i++)
      {
        Int n = syrk[i].n;

	if(n < 128) continue;
        h_syrk->n[j]         	= syrk[i].n;
        h_syrk->k[j]         	= syrk[i].k;
        h_syrk->lda[j]       	= syrk[i].lda;
        h_syrk->ldc[j]       	= syrk[i].ldc;
        h_syrk->A[j]         	= syrk[i].A;
        h_syrk->C[j]     	    = syrk[i].C;
        j++;
      }
      syrk_count = j;
      for(i = 0; i < desc_count; i++)
      {
        Int n = syrk[i].n;

	if( n >= 128) continue;
        h_syrk->n[j]            = syrk[i].n;
        h_syrk->k[j]            = syrk[i].k;
        h_syrk->lda[j]          = syrk[i].lda;
        h_syrk->ldc[j]          = syrk[i].ldc;
        h_syrk->A[j]            = syrk[i].A;
        h_syrk->C[j]            = syrk[i].C;
        j++;
      }


      /* store gemm dimensions & pointers */
      j = 0;
      for(i = 0; i < desc_count; i++)
      {
        Int m = gemm[i].m;
        Int n = gemm[i].n;
        Int k = gemm[i].k;

        if(m < 128 && n < 128 && k < 128) continue;
        h_gemm->m[j]         	= gemm[i].m;
        h_gemm->n[j]         	= gemm[i].n;
        h_gemm->k[j]         	= gemm[i].k;
        h_gemm->lda[j]       	= gemm[i].lda;
        h_gemm->ldb[j]       	= gemm[i].ldb;
        h_gemm->ldc[j]       	= gemm[i].ldc;
        h_gemm->A[j]         	= gemm[i].A;
        h_gemm->B[j]         	= gemm[i].B;
        h_gemm->C[j]         	= gemm[i].C;

        j++;
      }
      gemm_count = j;
      for(i = 0; i < desc_count; i++)
      {
        Int m = gemm[i].m;
        Int n = gemm[i].n;
        Int k = gemm[i].k;

        if(m >= 128 || n >= 128 || k >= 128) continue;
        h_gemm->m[j]            = gemm[i].m;
        h_gemm->n[j]            = gemm[i].n;
        h_gemm->k[j]            = gemm[i].k;
        h_gemm->lda[j]          = gemm[i].lda;
        h_gemm->ldb[j]          = gemm[i].ldb;
        h_gemm->ldc[j]          = gemm[i].ldc;
        h_gemm->A[j]            = gemm[i].A;
        h_gemm->B[j]            = gemm[i].B;
        h_gemm->C[j]            = gemm[i].C;

        j++;
      }


      /* store descendant dimensions & pointers (for addUpdate) */
      j = 0;
      for(i = 0; i < desc_count; i++)
      {
        Int ndrow1 = desc[i].ndrow1;
        Int ndrow2 = desc[i].ndrow2;

	if(ndrow1 < 128 && ndrow2 < 128) continue;
        h_desc->s[j]         = desc[i].s;
        h_desc->ndrow1[j]    = desc[i].ndrow1;
        h_desc->ndrow2[j]    = desc[i].ndrow2;
        h_desc->pdi1[j]      = desc[i].pdi1;
        h_desc->C[j]         = desc[i].C;

        j++;
      }
      update_count = j;
      for(i = 0; i < desc_count; i++)
      {
        Int ndrow1 = desc[i].ndrow1;
        Int ndrow2 = desc[i].ndrow2;
        Int ndrow3 = ndrow2-ndrow1;

        if(ndrow1 >= 128 || ndrow2 >= 128) continue;
        h_desc->s[j]         = desc[i].s;
        h_desc->ndrow1[j]    = desc[i].ndrow1;
        h_desc->ndrow2[j]    = desc[i].ndrow2;
        h_desc->pdi1[j]      = desc[i].pdi1;
        h_desc->C[j]         = desc[i].C;

        /* store maximum descendant dimensions (for addUpdateC) */
        if(ndrow3 > maxDim1[0]) maxDim1[0] = ndrow3;
        if(ndrow1 > maxDim1[1]) maxDim1[1] = ndrow1;
        if(ndrow2 > maxDim1[2]) maxDim1[2] = ndrow2;

        j++;
      }
      TIMER_END(tstart,tend,2);


      /*
       *  Copy dimensions & pointers from host to device
       *
       */
      /* memcpy dimensions and pointers of a batch of supernodes from host to device */
      cudaMemcpyAsync(d_dimSuper, h_dimSuper,13*maxbatch*sizeof(Int),cudaMemcpyHostToDevice,Common->gpuStream[gpuid * Common->numGPU_parallel][CHOLMOD_DEVICE_STREAMS]);
      cudaMemcpyAsync(d_ptrSuper, h_ptrSuper,3*maxbatch*sizeof(double *), cudaMemcpyHostToDevice,Common->gpuStream[gpuid * Common->numGPU_parallel][CHOLMOD_DEVICE_STREAMS]);

      /* memcpy dimensions and pointers of a batch of descendants from host to device */
      cudaMemcpyAsync(d_dimDesc, h_dimDesc,14*strideSize*sizeof(Int),cudaMemcpyHostToDevice,Common->gpuStream[gpuid * Common->numGPU_parallel][CHOLMOD_DEVICE_STREAMS]);
      cudaMemcpyAsync(d_ptrDesc, h_ptrDesc,6*strideSize*sizeof(double *), cudaMemcpyHostToDevice,Common->gpuStream[gpuid * Common->numGPU_parallel][CHOLMOD_DEVICE_STREAMS]);


      /*
       *  Initialize Supernode - BATCHED
       *
       *  Initializes the supernode with the following steps:
       *
       *  1. create Map for each supernode (in the batch)
       *  2. initialize Lx for each supernode
       *
       */
      TIMER_START(tstart,3);
      TEMPLATE2 (CHOLMOD(gpu_initialize_supernode_batch))(
              Common,
              gb_p,
              gpu_p,
              n,
              maxnsrow,
              maxkdif,
              maxnz,
              strideSize,
              super_count,
              gpuid);
     TIMER_END(tstart,tend,3);

     Common->ibuffer[gpuid] = 0;
     int Map_idx_next[IBUFF_LOOPSIZE];
     Int s_MapCached[IBUFF_LOOPSIZE][MAP_CACHESIZE];

     int buff_itr;
     int cache_itr;
     for (buff_itr = 0; buff_itr < IBUFF_LOOPSIZE; buff_itr++)
     {
         Map_idx_next[buff_itr] = 0;
         for (cache_itr = 0; cache_itr < MAP_CACHESIZE; cache_itr++)
         {
             s_MapCached[buff_itr][cache_itr] = EMPTY;
         }
     }

    for (d_itr = update_count_factorized_small; d_itr < update_count_factorized; d_itr++)
    {
        int iBuff; 

        const int stype = cpu_p->stype;

        Int s, d;
        Int i, j, k, p, pend, imap;
        Int pdi0, pdx0, ndrow0;
        Int *d_Ls, *d_Ap, *d_Ai;
        double *d_Ax, *h_LxFactorized, *d_LxFactorized;

#undef COMBINED_MULTIPLICATION
//#define COMBINED_MULTIPLICATION

#ifdef COMBINED_MULTIPLICATION
        Int lpos_save;
#endif

        Int k1, k2, psi, psend, psx, nsrow;
        double *d_A, *d_B, *d_C, *d_D;

        iBuff = Common->ibuffer[gpuid] % IBUFF_LOOPSIZE;
        Common->ibuffer[gpuid] = (Common->ibuffer[gpuid] + 1) % IBUFF_LOOPSIZE;

        d_Ls = gpu_p->d_Ls[gpuid];
        d_Ap = gpu_p->d_Ap[gpuid];
        d_Ai = gpu_p->d_Ai[gpuid];
        d_Ax = gpu_p->d_Ax[gpuid];
        h_LxFactorized = gpu_p->h_LxFactorized[gpuid][iBuff];
        d_LxFactorized = gpu_p->d_LxFactorized[gpuid][iBuff];

        d = update_factorized[d_itr].d;

        /* get descendant dimensions */
        kd1 	= Super [d] ;
        kd2 	= Super [d+1] ;
        ndcol = kd2 - kd1 ;
        pdi 	= Lpi [d] ;
        pdi0 = pdi + Lpos_save[d];
        pdend = Lpi [d+1] ;
        pdx0 	= Lpx [d] + Lpos_save[d] ;
        ndrow = pdend - pdi ;
        ndrow0 = pdend - pdi0 ;

#ifdef COMBINED_MULTIPLICATION
        lpos_save = Lpos_save[d];
        s = SuperMap [Ls [pdi + lpos_save]];
        while (s != EMPTY && LpxSub[s] >= 0 && lpos_save < ndrow)
        {
            k1 = Super[s];
            k2 = Super[s+1];
            psi = Lpi[s];
            psend = Lpi[s+1];
            psx = LpxSub[s];
            nsrow = psend - psi;

            p 	    = lpos_save ;
            pdi1    = pdi + p ;
            pdx1    = pdx + p ;

            for (pdi2 = pdi1; pdi2 < pdend && Ls [pdi2] < k2; pdi2++) ;

            ndrow1 = pdi2 - pdi1 ;
            ndrow2 = pdend - pdi1 ;
            ndrow3 = ndrow2 - ndrow1 ;

            /* prepare for next descendant */
            lpos_save = pdi2 - pdi ;

            s = SuperMap [Ls [pdi + lpos_save]];
        }
#endif

        cudaEventSynchronize(Common->updateCBuffersFree[gpuid * Common->numGPU_parallel][iBuff]);

#pragma omp parallel for num_threads (Common->ompNumThreads) private (i, j) if (ndcol > 32)
        for (j = 0; j < ndcol; j++)
        {
            for (i = 0; i < ndrow0; i++)
            {
                h_LxFactorized[i+j*ndrow0] = ((double *) (L->x) + pdx0)[i+j*ndrow];
            }
        }

        cudaMemcpyAsync (
                d_LxFactorized,
                h_LxFactorized,
                sizeof(double) * ndcol * ndrow0,
                cudaMemcpyHostToDevice,
                Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);

        cudaEventRecord (Common->updateCBuffersFree[gpuid * Common->numGPU_parallel][iBuff], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);

        cudaStreamWaitEvent (Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff], Common->updateCKernelsComplete[gpuid * Common->numGPU_parallel], 0);

#ifdef COMBINED_MULTIPLICATION
        ndrow1 = pdi2 - pdi0;
        ndrow2 = pdend - pdi0;
        ndrow3 = ndrow2 - ndrow1;

        d_A = d_LxFactorized;
        d_B = d_A + ndrow1;
        d_C = gpu_p->d_C[gpuid];
        d_D = d_C + ndrow1;

        cublasSetStream (Common->cublasHandle[gpuid], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
        cublasDsyrk (
                Common->cublasHandle[gpuid],
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                ndrow1,
                ndcol,
                &minus_one,
                d_A,
                ndrow0,
                &zero,
                d_C,
                ndrow0);
        if (ndrow3 > 0)
        {
            cublasDgemm (
                    Common->cublasHandle[gpuid],
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    ndrow3, ndrow1, ndcol,
                    &minus_one,
                    d_B,
                    ndrow0,
                    d_A,
                    ndrow0,
                    &zero,
                    d_D,
                    ndrow0);
        }
#endif

        s = SuperMap [Ls [pdi + Lpos_save[d]]];
        while (s != EMPTY && LpxSub[s] >= 0 && Lpos_save[d] < ndrow)
        {
            int cache_itr;
            Int *d_MapFactorized;

            k1 = Super[s];
            k2 = Super[s+1];
            psi = Lpi[s];
            psend = Lpi[s+1];
            psx = LpxSub[s];
            nsrow = psend - psi;

            p = Lpos_save[d];
            pdi1    = pdi + p ;
            pdx1    = pdx + p ;

            for (pdi2 = pdi1; pdi2 < pdend && Ls [pdi2] < k2; pdi2++) ;

            ndrow1 = pdi2 - pdi1 ;
            ndrow2 = pdend - pdi1 ;
            ndrow3 = ndrow2 - ndrow1 ;

            d_A = d_LxFactorized + (pdi1 - pdi0);
            d_B = d_A + ndrow1;
            d_C = gpu_p->d_C[gpuid] + ndrow0 * (pdi1 - pdi0) + (pdi1 - pdi0);
            d_D = d_C + ndrow1;

#ifndef COMBINED_MULTIPLICATION
            cublasSetStream (Common->cublasHandle[gpuid], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
            cublasDsyrk (
                    Common->cublasHandle[gpuid],
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    ndrow1,
                    ndcol,
                    &minus_one,
                    d_A,
                    ndrow0,
                    &zero,
                    d_C,
                    ndrow0);
            if (ndrow3 > 0)
            {
                cublasDgemm (
                        Common->cublasHandle[gpuid],
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        ndrow3, ndrow1, ndcol,
                        &minus_one,
                        d_B,
                        ndrow0,
                        d_A,
                        ndrow0,
                        &zero,
                        d_D,
                        ndrow0);
            }
#endif

            for (cache_itr = 0; cache_itr < MAP_CACHESIZE && s_MapCached[iBuff][cache_itr] != s; cache_itr++);

            if (cache_itr >= MAP_CACHESIZE)
            {
                cache_itr = Map_idx_next[iBuff];
                Map_idx_next[iBuff] = (cache_itr + 1) % MAP_CACHESIZE;
                s_MapCached[iBuff][cache_itr] = s;
                d_MapFactorized = gpu_p->d_MapFactorized[gpuid][iBuff][cache_itr];
                createMapOnDevice_factorized (d_MapFactorized, d_Ls, psi, nsrow, Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
            }
            else
            {
                d_MapFactorized = gpu_p->d_MapFactorized[gpuid][iBuff][cache_itr];
            }

            addUpdateOnDevice_factorized (gpu_p->d_Lx[gpuid] + psx, d_C, gpu_p->d_Ls[gpuid], d_MapFactorized, k1, k2, psi, psend, nsrow, pdi1, ndrow0, ndrow1, ndrow2, Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);

            /* prepare for next descendant */
            Lpos_save[d] = pdi2 - pdi ;

            s = SuperMap [Ls [pdi + Lpos_save[d]]];
        }

        cudaEventRecord (Common->updateCKernelsComplete[gpuid * Common->numGPU_parallel], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
    }

    const size_t batch_size_A = sizeof(double) * UPDATE_BATCH_SIZE_A;
    const size_t batch_size_C = sizeof(double) * UPDATE_BATCH_SIZE_C;
    Int max_batch_count;
    max_batch_count = UPDATE_MAX_BATCH_COUNT;
    if (max_batch_count > gb_p->CSize / batch_size_C)
        max_batch_count = gb_p->CSize / batch_size_C;
    if (max_batch_count > gb_p->LxSizeFactorized / (batch_size_A + 2 * sizeof(double*)))
        max_batch_count = gb_p->LxSizeFactorized / (batch_size_A + 2 * sizeof(double*));

    for (d_itr = 0; d_itr < update_count_factorized_small; d_itr += max_batch_count)
    {
        int iBuff; 
        Int batch_count;
        Int d_itr_offset;

        const int stype = cpu_p->stype;

        Int s, d;
        Int i, j, k, p, pend, imap;
        Int pdi0, pdx0, ndrow0;
        Int *d_Ls, *d_Ap, *d_Ai;
        double *d_Ax, *h_LxFactorized, *d_LxFactorized;

        Int k1, k2, psi, psend, psx, nsrow;
        double *d_A, *d_B, *d_C, *d_D;
        double **h_A_array, **h_C_array;
        double **d_A_array, **d_C_array;

        batch_count = MIN (max_batch_count, (update_count_factorized_small - d_itr));

        iBuff = Common->ibuffer[gpuid] % IBUFF_LOOPSIZE;
        Common->ibuffer[gpuid] = (Common->ibuffer[gpuid] + 1) % IBUFF_LOOPSIZE;

        d_Ls = gpu_p->d_Ls[gpuid];
        d_Ap = gpu_p->d_Ap[gpuid];
        d_Ai = gpu_p->d_Ai[gpuid];
        d_Ax = gpu_p->d_Ax[gpuid];
        h_LxFactorized = gpu_p->h_LxFactorized[gpuid][iBuff];
        d_LxFactorized = gpu_p->d_LxFactorized[gpuid][iBuff];

        cudaMemsetAsync (d_LxFactorized, 0, batch_count * batch_size_A, Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);

        cudaEventSynchronize(Common->updateCBuffersFree[gpuid * Common->numGPU_parallel][iBuff]);

        for (d_itr_offset = 0; d_itr_offset < batch_count; d_itr_offset++)
        {
        d = update_factorized[d_itr + d_itr_offset].d;

        /* get descendant dimensions */
        kd1 	= Super [d] ;
        kd2 	= Super [d+1] ;
        ndcol = kd2 - kd1 ;
        pdi 	= Lpi [d] ;
        pdi0 = pdi + Lpos_save[d];
        pdend = Lpi [d+1] ;
        pdx0 	= Lpx [d] + Lpos_save[d] ;
        ndrow = pdend - pdi ;
        ndrow0 = pdend - pdi0 ;

#pragma omp parallel for num_threads (Common->ompNumThreads) private (i, j) if (ndcol > 32)
        for (j = 0; j < ndcol; j++)
        {
            for (i = 0; i < ndrow0; i++)
            {
                (h_LxFactorized + d_itr_offset * UPDATE_BATCH_SIZE_A)[i+j*ndrow0] = ((double *) (L->x) + pdx0)[i+j*ndrow];
            }
        }

        cudaMemcpy2DAsync (
                d_LxFactorized + d_itr_offset * UPDATE_BATCH_SIZE_A,
                sizeof(double) * UPDATE_BATCH_DIMENSION_M,
                h_LxFactorized + d_itr_offset * UPDATE_BATCH_SIZE_A,
                sizeof(double) * ndrow0,
                sizeof(double) * ndrow0,
                ndcol,
                cudaMemcpyHostToDevice,
                Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
        }

        cudaEventRecord (Common->updateCBuffersFree[gpuid * Common->numGPU_parallel][iBuff], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);

        cudaStreamWaitEvent (Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff], Common->updateCKernelsComplete[gpuid * Common->numGPU_parallel], 0);

#undef COMBINED_MULTIPLICATION
#define COMBINED_MULTIPLICATION

#ifdef COMBINED_MULTIPLICATION
        h_A_array = h_LxFactorized + UPDATE_MAX_BATCH_COUNT * UPDATE_BATCH_SIZE_A;
        h_C_array = h_A_array + UPDATE_MAX_BATCH_COUNT;
        d_A_array = d_LxFactorized + UPDATE_MAX_BATCH_COUNT * UPDATE_BATCH_SIZE_A;
        d_C_array = d_A_array + UPDATE_MAX_BATCH_COUNT;
        for (d_itr_offset = 0; d_itr_offset < batch_count; d_itr_offset++)
        {
            h_A_array[d_itr_offset] = d_LxFactorized + d_itr_offset * UPDATE_BATCH_SIZE_A;
            h_C_array[d_itr_offset] = gpu_p->d_C[gpuid] + d_itr_offset * UPDATE_BATCH_SIZE_C;
        }
        cudaMemcpyAsync (d_A_array, h_A_array, sizeof (double *) * batch_count, cudaMemcpyHostToDevice, Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
        cudaMemcpyAsync (d_C_array, h_C_array, sizeof (double *) * batch_count, cudaMemcpyHostToDevice, Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);

        cublasSetStream (Common->cublasHandle[gpuid], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
        cublasDgemmBatched (
                Common->cublasHandle[gpuid],
                CUBLAS_OP_N, CUBLAS_OP_T,
                UPDATE_BATCH_DIMENSION_M, UPDATE_BATCH_DIMENSION_N, UPDATE_BATCH_DIMENSION_K,
                &minus_one,
                d_A_array,
                UPDATE_BATCH_DIMENSION_M,
                d_A_array,
                UPDATE_BATCH_DIMENSION_M,
                &zero,
                d_C_array,
                UPDATE_BATCH_DIMENSION_M,
                batch_count);
#endif

        for (d_itr_offset = 0; d_itr_offset < batch_count; d_itr_offset++)
        {
        d = update_factorized[d_itr + d_itr_offset].d;

        /* get descendant dimensions */
        kd1 	= Super [d] ;
        kd2 	= Super [d+1] ;
        ndcol = kd2 - kd1 ;
        pdi 	= Lpi [d] ;
        pdi0 = pdi + Lpos_save[d];
        pdend = Lpi [d+1] ;
        pdx0 	= Lpx [d] + Lpos_save[d] ;
        ndrow = pdend - pdi ;
        ndrow0 = pdend - pdi0 ;

        s = SuperMap [Ls [pdi + Lpos_save[d]]];
        while (s != EMPTY && LpxSub[s] >= 0 && Lpos_save[d] < ndrow)
        {
            int cache_itr;
            Int *d_MapFactorized;

            k1 = Super[s];
            k2 = Super[s+1];
            psi = Lpi[s];
            psend = Lpi[s+1];
            psx = LpxSub[s];
            nsrow = psend - psi;

            p = Lpos_save[d];
            pdi1    = pdi + p ;
            pdx1    = pdx + p ;

            for (pdi2 = pdi1; pdi2 < pdend && Ls [pdi2] < k2; pdi2++) ;

            ndrow1 = pdi2 - pdi1 ;
            ndrow2 = pdend - pdi1 ;
            ndrow3 = ndrow2 - ndrow1 ;

            d_A = d_LxFactorized + d_itr_offset * UPDATE_BATCH_SIZE_A + (pdi1 - pdi0);
            d_B = d_A + ndrow1;
            d_C = gpu_p->d_C[gpuid] + d_itr_offset * UPDATE_BATCH_SIZE_C + (pdi1 - pdi0) * UPDATE_BATCH_DIMENSION_M + (pdi1 - pdi0);
            d_D = d_C + ndrow1;

#ifndef COMBINED_MULTIPLICATION
            cublasSetStream (Common->cublasHandle[gpuid], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
            cublasDsyrk (
                    Common->cublasHandle[gpuid],
                    CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N,
                    ndrow1,
                    ndcol,
                    &minus_one,
                    d_A,
                    UPDATE_BATCH_DIMENSION_M,
                    &zero,
                    d_C,
                    UPDATE_BATCH_DIMENSION_M);
            if (ndrow3 > 0)
            {
                cublasDgemm (
                        Common->cublasHandle[gpuid],
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        ndrow3, ndrow1, ndcol,
                        &minus_one,
                        d_B,
                        UPDATE_BATCH_DIMENSION_M,
                        d_A,
                        UPDATE_BATCH_DIMENSION_M,
                        &zero,
                        d_D,
                        UPDATE_BATCH_DIMENSION_M);
            }
#endif

            for (cache_itr = 0; cache_itr < MAP_CACHESIZE && s_MapCached[iBuff][cache_itr] != s; cache_itr++);

            if (cache_itr >= MAP_CACHESIZE)
            {
                cache_itr = Map_idx_next[iBuff];
                Map_idx_next[iBuff] = (cache_itr + 1) % MAP_CACHESIZE;
                s_MapCached[iBuff][cache_itr] = s;
                d_MapFactorized = gpu_p->d_MapFactorized[gpuid][iBuff][cache_itr];
                createMapOnDevice_factorized (d_MapFactorized, d_Ls, psi, nsrow, Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
            }
            else
            {
                d_MapFactorized = gpu_p->d_MapFactorized[gpuid][iBuff][cache_itr];
            }

            addUpdateOnDevice_factorized (gpu_p->d_Lx[gpuid] + psx, d_C, gpu_p->d_Ls[gpuid], d_MapFactorized, k1, k2, psi, psend, nsrow, pdi1, UPDATE_BATCH_DIMENSION_M, ndrow1, ndrow2, Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);

            /* prepare for next descendant */
            Lpos_save[d] = pdi2 - pdi ;

            s = SuperMap [Ls [pdi + Lpos_save[d]]];
        }
        }

        cudaEventRecord (Common->updateCKernelsComplete[gpuid * Common->numGPU_parallel], Common->gpuStream[gpuid * Common->numGPU_parallel][iBuff]);
    }

    cudaEventSynchronize(Common->updateCKernelsComplete[gpuid * Common->numGPU_parallel]);

      /*
       *  Supernode Assembly - BATCHED
       *
       *  Assemble the supernodes with the following steps:
       *
       *  1. perform batched dsyrk
       *  2. perform batched dgemm
       *  3. perform batched addUpdate
       *
       */

      /* check if descendants > 0 (which is when level > 0) */
      if(level_descendants[level_descendants_ptrs[subtree]+level]>0) {

          TEMPLATE2 (CHOLMOD(gpu_updateC_batch))( Common,
                  gb_p,
                  gpu_p,
                  cpu_p,
                  tree_p,
                  prof_p,
                  n,
                  subtree,
                  level,
                  node,
                  desc_count,
                  syrk_count,
                  gemm_count,
                  update_count,
                  gpuid,
                  1,
                  start,
                  end,
                  maxDim1,
                  LpxSub,
                  L->px);

      }


      /*
       *  Cholesky Factorization - BATCHED
       *
       *  Perform batched dpotrf.
       *
       */
      TEMPLATE2 (CHOLMOD(gpu_lower_potrf_batch))(
              Common,
              gb_p,
              gpu_p,
              prof_p,
              super_count,
              potrf_count,
              gpuid);


      /*
       *  Triangular Solve - BATCHED
       *
       *  Perform batched dtrsm.
       *
       */
      TEMPLATE2 (CHOLMOD(gpu_triangular_solve_batch))(
              Common,
              gb_p,
              gpu_p,
              prof_p,
              super_count,
              trsm_count,
              gpuid);



      /*
       *  Copy Supernodes
       *
       *  Copies factored supernodes from device to pinned to regular memory.
       *
       */

      TEMPLATE2 (CHOLMOD(gpu_copy_supernode2))(
              Common,
              gb_p,
              gpu_p,
              super_count,
              maxnsrownscol,
              gpuid);



      /* increment supernode id */
      node += nbatch;
      tend[0] += SuiteSparse_time() - tstart[0];
      TIMER_END(tstart,tend,0);

    } /* end loop over group streams */



    /* synchronize all streams - need this? or just redundancy? */
    //cudaStreamSynchronize (Common->gpuStream[gpuid * Common->numGPU_parallel][CHOLMOD_DEVICE_STREAMS]);


    #ifdef USE_NVTX
      nvtxRangeEnd(range1);
    #endif



    /* increment flops */
    SUM(syrk_flops,0,(double)(syrk_flops[1]));
    SUM(gemm_flops,0,(double)(gemm_flops[1]));
    SUM(potrf_flops,0,(double)(potrf_flops[1]));
    SUM(trsm_flops,0,(double)(trsm_flops[1]));

    /* print benchmarks per level */
    PRINTFV("%d\t",level);
    PRINTFV("%d\t",nbatch);
    PRINTFV("%d\t",end-start);
    PRINTFV("%f\t",syrk_time[1]);
    PRINTFV("%f\t",gemm_time[1]);
    PRINTFV("%f\t",potrf_time[1]);
    PRINTFV("%f\t",trsm_time[1]);
    PRINTFV("%f\t",1.0e-9*syrk_flops[1]/syrk_time[1]);
    PRINTFV("%f\t",1.0e-9*gemm_flops[1]/gemm_time[1]);
    PRINTFV("%f\t",1.0e-9*potrf_flops[1]/potrf_time[1]);
    PRINTFV("%f\n",1.0e-9*trsm_flops[1]/trsm_time[1]);


    /* clear timers */
    CLEAR1(syrk_time,1);
    CLEAR1(gemm_time,1);
    CLEAR1(potrf_time,1);
    CLEAR1(trsm_time,1);

  } /* end loop  over levels */


  /* clear factor on GPU */
  cudaMemsetAsync ( d_Lx, 0, factor_size[subtree]*sizeof(double),Common->gpuStream[gpuid * Common->numGPU_parallel][CHOLMOD_DEVICE_STREAMS+1]);


  /*
   *  Store supernode
   *
   *  Copies factored supernodes from pinned (h_Lx) to regular memory (Lx).
   *  Ony for last level.
   *
   */
  if (supernode_num_levels[subtree] > 0)
  TEMPLATE2 (CHOLMOD(gpu_copy_supernode))(
          Common,
          gpu_p,
          cpu_p,
          tree_p,
          subtree,
          level,
          gpuid,
          2,
          start,
          end,
          LpxSub,
          L->px);


  //cudaStreamSynchronize (Common->gpuStream[gpuid * Common->numGPU_parallel][0]);


 /* print overall benchmarks */
  PRINTFV("\n\nSubtree %d benchmarks:\n",subtree);
  PRINTF("\n- time -\n");
  PRINTFV("total:        %f\n",tend[0]);
  PRINTFV("superDim:     %f\n",tend[1]);
  PRINTFV("descDim:      %f\n",tend[2]);
  PRINTFV("initLx:       %f\n",tend[3]);
  PRINTFV("dsyrk:        %f\n",syrk_time[0]);
  PRINTFV("dgemm:        %f\n",gemm_time[0]);
  PRINTFV("assembly:     %f\n",tend[4]);
  PRINTFV("dpotrf:       %f\n",potrf_time[0]);
  PRINTFV("dtrsm:        %f\n",trsm_time[0]);
  PRINTF("\n- flops -\n");
  PRINTFV("dsyrk:        %f\n",1.0e-9*syrk_flops[0]/syrk_time[0]);
  PRINTFV("dgemm:        %f\n",1.0e-9*gemm_flops[0]/gemm_time[0]);
  PRINTFV("dpotrf:       %f\n",1.0e-9*potrf_flops[0]/potrf_time[0]);
  PRINTFV("dtrsm:        %f\n",1.0e-9*trsm_flops[0]/trsm_time[0]);
  PRINTF("\n");

#endif

}
