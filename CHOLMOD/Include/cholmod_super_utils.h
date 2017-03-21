#ifndef CHOLMOD_SUPER_UTILS_H
#define CHOLMOD_SUPER_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

void cholmod_qSort (int *key, int *value, int low, int high);
void cholmod_l_qSort (SuiteSparse_long *key, SuiteSparse_long *value, SuiteSparse_long low, SuiteSparse_long high);
void cholmod_qRevSort (int *key, int *value, int low, int high);
void cholmod_l_qRevSort (SuiteSparse_long *key, SuiteSparse_long *value, SuiteSparse_long low, SuiteSparse_long high);
void cholmod_init_gpus (int for_whom, cholmod_common *Common, int pdev);
void cholmod_l_init_gpus (int for_whom, cholmod_common *Common, int pdev);

#ifdef __cplusplus
}
#endif

#endif
