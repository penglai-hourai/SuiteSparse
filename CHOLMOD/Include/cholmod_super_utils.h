#ifndef CHOLMOD_SUPER_UTILS_H
#define CHOLMOD_SUPER_UTILS_H

#include "cholmod_internal.h"

void CHOLMOD (qSort) (Int *key, Int *value, Int low, Int high);
void CHOLMOD (qRevSort) (Int *key, Int *value, Int low, Int high);

#endif
