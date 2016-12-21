#ifndef NGPL
#ifndef NSUPERNODAL

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

#endif
#endif
