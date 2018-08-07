// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/block_apply_2_by_1.cu ==================
// =============================================================================

//------------------------------------------------------------------------------
// block_apply_2_by_1: handles all edge cases, but just a single column tile
//------------------------------------------------------------------------------

#define BLOCK_APPLY block_apply_2_by_1_vt
#define ROW_PANELSIZE 2
#define COL_PANELSIZE 1
#define USE_VT
#include "block_apply.cu"
