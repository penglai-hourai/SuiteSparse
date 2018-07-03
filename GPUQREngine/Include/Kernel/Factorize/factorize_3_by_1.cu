// =============================================================================
// === GPUQREngine/Include/Kernel/Factorize/factorize_3_by_1.cu ================
// =============================================================================

//------------------------------------------------------------------------------
// 96-by-32 factorize, no VT or tiles, edge case.  384 threads   WHOLE FRONT
//------------------------------------------------------------------------------

#define FACTORIZE   factorize_small_front
#define M           (PANELSIZE * TILESIZE)
#define N           (TILESIZE)
#define BITTYROWS   (8)
#define WHOLE_FRONT
#include "Kernel/Factorize/factorize_vt.cu"
