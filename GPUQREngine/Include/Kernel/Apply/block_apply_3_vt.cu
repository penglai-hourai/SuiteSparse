// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/block_apply_3.cu =======================
// =============================================================================

//------------------------------------------------------------------------------
// block_apply_3: handles all edge cases and any number of column tiles
//------------------------------------------------------------------------------

#define BLOCK_APPLY block_apply_3_vt
#define ROW_PANELSIZE 3
#define COL_PANELSIZE 2
#define USE_VT
#include "block_apply.cu"
