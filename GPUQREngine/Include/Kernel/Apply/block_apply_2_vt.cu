// =============================================================================
// === GPUQREngine/Include/Kernel/Apply/block_apply_2.cu =======================
// =============================================================================

//------------------------------------------------------------------------------
// block_apply_2: handles all edge cases and any number of column tiles
//------------------------------------------------------------------------------

#define BLOCK_APPLY block_apply_2_vt
#define ROW_PANELSIZE 2
#define COL_PANELSIZE 2
#define APPLY_VT
#include "block_apply.cu"
#undef APPLY_VT
