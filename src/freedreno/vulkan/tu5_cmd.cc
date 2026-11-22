/*
 * Copyright © 2016 Rob Clark <robclark@freedesktop.org>
 * Copyright © 2026 hiru
 * SPDX-License-Identifier: MIT
 *
 * The command stream of Adreno 5xx. The register programming is the gallium
 * driver's (a5xx/fd5_emit.c, fd5_program.c, fd5_gmem.c), with the state taken
 * from the Vulkan objects.
 */

#include "tu5.h"

#include "util/half_float.h"
#include "vk_format.h"

#include "ir3/ir3_shader.h"

#include "tu_cmd_buffer.h"
#include "tu_cs.h"
#include "tu_device.h"
#include "tu_image.h"
#include "tu_pass.h"
#include "tu_shader.h"

/* The registers of a5xx and of a6xx, which the headers above bring in, have
 * enumerators of the same names: keep those of a5xx apart.
 */
namespace a5xx_regs {
#include "fd5_hw.h"
}
using namespace a5xx_regs;

#include "tu5_format.h"

#define A5XX_MAX_RENDER_TARGETS 8

/* What the gallium driver writes its command stream with */
#define OUT_PKT4(cs, reg, cnt) tu_cs_emit_pkt4(cs, reg, cnt)
#define OUT_PKT7(cs, opc, cnt) tu_cs_emit_pkt7(cs, opc, cnt)
#define OUT_RING(cs, val)      tu_cs_emit(cs, val)
#ifndef COND
#define COND(bool, val) ((bool) ? (val) : 0)
#endif

static void
tu5_event_write(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                enum vgt_event_type evt, bool timestamp)
{
   OUT_PKT7(cs, CP_EVENT_WRITE, timestamp ? 4 : 1);
   OUT_RING(cs, CP_EVENT_WRITE_0_EVENT(evt));
   if (timestamp) {
      tu_cs_emit_qw(cs, global_iova(cmd, seqno_dummy));
      OUT_RING(cs, 0x00000000);
   }
}

static void
tu5_set_render_mode(struct tu_cs *cs, enum render_mode_cmd mode)
{
   OUT_PKT7(cs, CP_SET_RENDER_MODE, 5);
   OUT_RING(cs, CP_SET_RENDER_MODE_0_MODE(mode));
   OUT_RING(cs, 0x00000000); /* ADDR_LO */
   OUT_RING(cs, 0x00000000); /* ADDR_HI */
   OUT_RING(cs, COND(mode == GMEM, CP_SET_RENDER_MODE_3_GMEM_ENABLE) |
                   COND(mode == BINNING, CP_SET_RENDER_MODE_3_VSC_ENABLE));
   OUT_RING(cs, 0x00000000);
}

static void
tu5_cache_flush(struct tu_cs *cs)
{
   OUT_PKT4(cs, REG_A5XX_UCHE_CACHE_INVALIDATE_MIN_LO, 5);
   OUT_RING(cs, 0x00000000); /* UCHE_CACHE_INVALIDATE_MIN_LO */
   OUT_RING(cs, 0x00000000); /* UCHE_CACHE_INVALIDATE_MIN_HI */
   OUT_RING(cs, 0x00000000); /* UCHE_CACHE_INVALIDATE_MAX_LO */
   OUT_RING(cs, 0x00000000); /* UCHE_CACHE_INVALIDATE_MAX_HI */
   OUT_RING(cs, 0x00000012); /* UCHE_CACHE_INVALIDATE */
   tu_cs_emit_wfi(cs);
}

static void
tu5_emit_lrz_flush(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   OUT_PKT4(cs, REG_A5XX_GRAS_LRZ_CNTL, 1);
   OUT_RING(cs, A5XX_GRAS_LRZ_CNTL_ENABLE);

   tu5_event_write(cmd, cs, LRZ_FLUSH, false);

   OUT_PKT4(cs, REG_A5XX_GRAS_LRZ_CNTL, 1);
   OUT_RING(cs, 0x0);
}

static enum a4xx_state_block
tu5_stage2shadersb(gl_shader_stage type)
{
   switch (type) {
   case MESA_SHADER_VERTEX:
      return SB4_VS_SHADER;
   case MESA_SHADER_FRAGMENT:
      return SB4_FS_SHADER;
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_KERNEL:
      return SB4_CS_SHADER;
   default:
      unreachable("bad shader type");
   }
}

/* Have the shader's instructions loaded from where the pipeline put them */
static void
tu5_emit_shader(struct tu_cs *cs, const struct ir3_shader_variant *so,
                uint64_t iova)
{
   OUT_PKT7(cs, CP_LOAD_STATE4, 3);
   OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(0) |
                   CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) |
                   CP_LOAD_STATE4_0_STATE_BLOCK(tu5_stage2shadersb(so->type)) |
                   CP_LOAD_STATE4_0_NUM_UNIT(so->instrlen));
   OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_SHADER) | (uint32_t)iova);
   OUT_RING(cs, iova >> 32);
}

static void
tu5_emit_shader_obj(struct tu_cs *cs, const struct tu_shader *shader,
                    uint32_t shader_obj_reg)
{
   OUT_PKT4(cs, shader_obj_reg, 6);
   tu_cs_emit_qw(cs, shader->binary_iova); /* SP_VS_OBJ_START_LO/HI */
   OUT_RING(cs, A5XX_SP_VS_PVT_MEM_PARAM_MEMSIZEPERITEM(
                   shader->pvtmem.per_fiber_size) |
                   A5XX_SP_VS_PVT_MEM_PARAM_HWSTACKOFFSET(
                      shader->pvtmem.per_sp_size));
   tu_cs_emit_qw(cs, shader->pvtmem.iova); /* SP_xS_PVT_MEM_ADDR */
   OUT_RING(cs,
            A5XX_SP_VS_PVT_MEM_SIZE_TOTALPVTMEMSIZE(shader->pvtmem.per_sp_size));
}

/* regid:      base const register
 * dwords:     the constant values
 * sizedwords: size of them
 */
static void
tu5_emit_const(struct tu_cs *cs, const struct ir3_shader_variant *v,
               uint32_t regid, uint32_t sizedwords, const uint32_t *dwords)
{
   assert((regid % 4) == 0);
   assert((sizedwords % 4) == 0);

   OUT_PKT7(cs, CP_LOAD_STATE4, 3 + sizedwords);
   OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(regid / 4) |
                   CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                   CP_LOAD_STATE4_0_STATE_BLOCK(tu5_stage2shadersb(v->type)) |
                   CP_LOAD_STATE4_0_NUM_UNIT(sizedwords / 4));
   OUT_RING(cs, CP_LOAD_STATE4_1_EXT_SRC_ADDR(0) |
                   CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
   OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   tu_cs_emit_array(cs, dwords, sizedwords);
}

static void
tu5_emit_immediates(struct tu_cs *cs, const struct ir3_shader_variant *v)
{
   const struct ir3_const_state *const_state = ir3_const_state(v);
   uint32_t base = const_state->offsets.immediate;
   int size = DIV_ROUND_UP(const_state->immediates_count, 4);

   /* truncate size to avoid writing constants that shader
    * does not use:
    */
   size = MIN2(size + base, v->constlen) - base;

   /* convert out of vec4: */
   base *= 4;
   size *= 4;

   if (size > 0)
      tu5_emit_const(cs, v, base, size, const_state->immediates);
}

/* The draw parameters the vertex shader reads as constants */
static void
tu5_emit_vs_params(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                   const struct ir3_shader_variant *v)
{
   const struct ir3_const_state *const_state = ir3_const_state(v);
   uint32_t offset = const_state->offsets.driver_param;

   if (offset >= v->constlen)
      return;

   const struct tu_vs_params *p = &cmd->state.last_vs_params;
   uint32_t params[4] = {};

   params[IR3_DP_VS(draw_id)] = p->draw_id;
   params[IR3_DP_VS(vtxid_base)] = p->vertex_offset;
   params[IR3_DP_VS(instid_base)] = p->first_instance;

   tu5_emit_const(cs, v, offset * 4, 4, params);
}

void
tu5_emit_flushes(struct tu_cmd_buffer *cmd, struct tu_cs *cs, uint32_t flushes)
{
   if (flushes & (TU_CMD_FLAG_CCU_CLEAN_COLOR | TU_CMD_FLAG_CCU_INVALIDATE_COLOR))
      tu5_event_write(cmd, cs, PC_CCU_FLUSH_COLOR_TS, true);
   if (flushes & (TU_CMD_FLAG_CCU_CLEAN_DEPTH | TU_CMD_FLAG_CCU_INVALIDATE_DEPTH))
      tu5_event_write(cmd, cs, PC_CCU_FLUSH_DEPTH_TS, true);
   if (flushes & TU_CMD_FLAG_CCU_INVALIDATE_COLOR)
      tu5_event_write(cmd, cs, PC_CCU_INVALIDATE_COLOR, false);
   if (flushes & TU_CMD_FLAG_CCU_INVALIDATE_DEPTH)
      tu5_event_write(cmd, cs, PC_CCU_INVALIDATE_DEPTH, false);
   if (flushes & (TU_CMD_FLAG_CACHE_CLEAN | TU_CMD_FLAG_CACHE_INVALIDATE))
      tu5_cache_flush(cs);
   else if (flushes & TU_CMD_FLAG_WAIT_FOR_IDLE)
      tu_cs_emit_wfi(cs);
   if (flushes & TU_CMD_FLAG_WAIT_FOR_ME)
      tu_cs_emit_pkt7(cs, CP_WAIT_FOR_ME, 0);
}

struct tu5_program {
   const struct ir3_shader_variant *vs, *fs;
   const struct tu_shader *vs_shader, *fs_shader;
   bool binning_pass;
   bool rasterflat;
   unsigned max_loc;
};

struct stage {
   const struct ir3_shader_variant *v;
   const struct ir3_info *i;
   /* const sizes are in units of 4 * vec4 */
   uint8_t constoff;
   uint8_t constlen;
   /* instr sizes are in units of 16 instructions */
   uint8_t instroff;
   uint8_t instrlen;
};

enum { VS = 0, FS = 1, HS = 2, DS = 3, GS = 4, MAX_STAGES };

static void
setup_stages(const struct tu5_program *emit, struct stage *s)
{
   unsigned i;

   s[VS].v = emit->vs;
   s[FS].v = emit->fs;

   s[HS].v = s[DS].v = s[GS].v = NULL; /* for now */

   for (i = 0; i < MAX_STAGES; i++) {
      if (s[i].v) {
         s[i].i = &s[i].v->info;
         /* constlen is in units of 4 * vec4: */
         assert(s[i].v->constlen % 4 == 0);
         s[i].constlen = s[i].v->constlen / 4;
         /* instrlen is already in units of 16 instr.. although
          * probably we should ditch that and not make the compiler
          * care about instruction group size of a3xx vs a5xx
          */
         s[i].instrlen = s[i].v->instrlen;
      } else {
         s[i].i = NULL;
         s[i].constlen = 0;
         s[i].instrlen = 0;
      }
   }

   /* NOTE: at least for gles2, blob partitions VS at bottom of const
    * space and FS taking entire remaining space.  We probably don't
    * need to do that the same way, but for now mimic what the blob
    * does to make it easier to diff against register values from blob
    *
    * NOTE: if VS.instrlen + FS.instrlen > 64, then one or both shaders
    * is run from external memory.
    */
   if ((s[VS].instrlen + s[FS].instrlen) > 64) {
      /* prioritize FS for internal memory: */
      if (s[FS].instrlen < 64) {
         /* if FS can fit, kick VS out to external memory: */
         s[VS].instrlen = 0;
      } else if (s[VS].instrlen < 64) {
         /* otherwise if VS can fit, kick out FS: */
         s[FS].instrlen = 0;
      } else {
         /* neither can fit, run both from external memory: */
         s[VS].instrlen = 0;
         s[FS].instrlen = 0;
      }
   }

   unsigned constoff = 0;
   for (i = 0; i < MAX_STAGES; i++) {
      s[i].constoff = constoff;
      constoff += s[i].constlen;
   }

   s[VS].instroff = 0;
   s[FS].instroff = 64 - s[FS].instrlen;
   s[HS].instroff = s[DS].instroff = s[GS].instroff = s[FS].instroff;
}

static inline uint32_t
next_regid(uint32_t reg, uint32_t increment)
{
   if (VALIDREG(reg))
      return reg + increment;
   else
      return regid(63, 0);
}

static void
tu5_program_emit(struct tu_cs *ring, struct tu5_program *emit)
{
   struct stage s[MAX_STAGES];
   uint32_t pos_regid, psize_regid, color_regid[8];
   uint32_t face_regid, coord_regid, zwcoord_regid, samp_id_regid,
      samp_mask_regid;
   uint32_t ij_regid[IJ_COUNT], vertex_regid, instance_regid, clip0_regid,
      clip1_regid;
   enum a3xx_threadsize fssz;
   uint8_t psize_loc = ~0;
   int i, j;

   setup_stages(emit, s);

   bool do_streamout = (s[VS].v->stream_output.num_outputs > 0);
   uint8_t clip_mask = s[VS].v->clip_mask,
           cull_mask = s[VS].v->cull_mask;
   uint8_t clip_cull_mask = clip_mask | cull_mask;

   /* Unlike a6xx, we don't factor the rasterizer's clip enables in here.  It's
    * already handled by the frontend by storing 0.0 to the clipdist in the
    * shader variant (using either nir_lower_clip_disable for clip distances
    * from the source shader, or nir_lower_clip_vs for user clip planes).
    * Masking the disabled clipdists off causes GPU hangs in tests like
    * spec@glsl-1.20@execution@clipping@vs-clip-vertex-enables.
    */

   fssz = (s[FS].i->double_threadsize) ? FOUR_QUADS : TWO_QUADS;

   pos_regid = ir3_find_output_regid(s[VS].v, VARYING_SLOT_POS);
   psize_regid = ir3_find_output_regid(s[VS].v, VARYING_SLOT_PSIZ);
   clip0_regid = ir3_find_output_regid(s[VS].v, VARYING_SLOT_CLIP_DIST0);
   clip1_regid = ir3_find_output_regid(s[VS].v, VARYING_SLOT_CLIP_DIST1);
   vertex_regid =
      ir3_find_sysval_regid(s[VS].v, SYSTEM_VALUE_VERTEX_ID_ZERO_BASE);
   instance_regid = ir3_find_sysval_regid(s[VS].v, SYSTEM_VALUE_INSTANCE_ID);

   if (s[FS].v->color0_mrt) {
      color_regid[0] = color_regid[1] = color_regid[2] = color_regid[3] =
         color_regid[4] = color_regid[5] = color_regid[6] = color_regid[7] =
            ir3_find_output_regid(s[FS].v, FRAG_RESULT_COLOR);
   } else {
      color_regid[0] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA0);
      color_regid[1] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA1);
      color_regid[2] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA2);
      color_regid[3] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA3);
      color_regid[4] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA4);
      color_regid[5] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA5);
      color_regid[6] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA6);
      color_regid[7] = ir3_find_output_regid(s[FS].v, FRAG_RESULT_DATA7);
   }

   samp_id_regid = ir3_find_sysval_regid(s[FS].v, SYSTEM_VALUE_SAMPLE_ID);
   samp_mask_regid =
      ir3_find_sysval_regid(s[FS].v, SYSTEM_VALUE_SAMPLE_MASK_IN);
   face_regid = ir3_find_sysval_regid(s[FS].v, SYSTEM_VALUE_FRONT_FACE);
   coord_regid = ir3_find_sysval_regid(s[FS].v, SYSTEM_VALUE_FRAG_COORD);
   zwcoord_regid = next_regid(coord_regid, 2);
   for (unsigned i = 0; i < ARRAY_SIZE(ij_regid); i++)
      ij_regid[i] = ir3_find_sysval_regid(
         s[FS].v, SYSTEM_VALUE_BARYCENTRIC_PERSP_PIXEL + i);

   /* we could probably divide this up into things that need to be
    * emitted if frag-prog is dirty vs if vert-prog is dirty..
    */

   OUT_PKT4(ring, REG_A5XX_HLSQ_VS_CONFIG, 5);
   OUT_RING(ring, A5XX_HLSQ_VS_CONFIG_CONSTOBJECTOFFSET(s[VS].constoff) |
                     A5XX_HLSQ_VS_CONFIG_SHADEROBJOFFSET(s[VS].instroff) |
                     COND(s[VS].v, A5XX_HLSQ_VS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_HLSQ_FS_CONFIG_CONSTOBJECTOFFSET(s[FS].constoff) |
                     A5XX_HLSQ_FS_CONFIG_SHADEROBJOFFSET(s[FS].instroff) |
                     COND(s[FS].v, A5XX_HLSQ_FS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_HLSQ_HS_CONFIG_CONSTOBJECTOFFSET(s[HS].constoff) |
                     A5XX_HLSQ_HS_CONFIG_SHADEROBJOFFSET(s[HS].instroff) |
                     COND(s[HS].v, A5XX_HLSQ_HS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_HLSQ_DS_CONFIG_CONSTOBJECTOFFSET(s[DS].constoff) |
                     A5XX_HLSQ_DS_CONFIG_SHADEROBJOFFSET(s[DS].instroff) |
                     COND(s[DS].v, A5XX_HLSQ_DS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_HLSQ_GS_CONFIG_CONSTOBJECTOFFSET(s[GS].constoff) |
                     A5XX_HLSQ_GS_CONFIG_SHADEROBJOFFSET(s[GS].instroff) |
                     COND(s[GS].v, A5XX_HLSQ_GS_CONFIG_ENABLED));

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONFIG, 1);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_HLSQ_VS_CNTL, 5);
   OUT_RING(ring, A5XX_HLSQ_VS_CNTL_INSTRLEN(s[VS].instrlen) |
                     COND(s[VS].v && s[VS].v->has_ssbo,
                          A5XX_HLSQ_VS_CNTL_SSBO_ENABLE));
   OUT_RING(ring, A5XX_HLSQ_FS_CNTL_INSTRLEN(s[FS].instrlen) |
                     COND(s[FS].v && s[FS].v->has_ssbo,
                          A5XX_HLSQ_FS_CNTL_SSBO_ENABLE));
   OUT_RING(ring, A5XX_HLSQ_HS_CNTL_INSTRLEN(s[HS].instrlen) |
                     COND(s[HS].v && s[HS].v->has_ssbo,
                          A5XX_HLSQ_HS_CNTL_SSBO_ENABLE));
   OUT_RING(ring, A5XX_HLSQ_DS_CNTL_INSTRLEN(s[DS].instrlen) |
                     COND(s[DS].v && s[DS].v->has_ssbo,
                          A5XX_HLSQ_DS_CNTL_SSBO_ENABLE));
   OUT_RING(ring, A5XX_HLSQ_GS_CNTL_INSTRLEN(s[GS].instrlen) |
                     COND(s[GS].v && s[GS].v->has_ssbo,
                          A5XX_HLSQ_GS_CNTL_SSBO_ENABLE));

   OUT_PKT4(ring, REG_A5XX_SP_VS_CONFIG, 5);
   OUT_RING(ring, A5XX_SP_VS_CONFIG_CONSTOBJECTOFFSET(s[VS].constoff) |
                     A5XX_SP_VS_CONFIG_SHADEROBJOFFSET(s[VS].instroff) |
                     COND(s[VS].v, A5XX_SP_VS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_SP_FS_CONFIG_CONSTOBJECTOFFSET(s[FS].constoff) |
                     A5XX_SP_FS_CONFIG_SHADEROBJOFFSET(s[FS].instroff) |
                     COND(s[FS].v, A5XX_SP_FS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_SP_HS_CONFIG_CONSTOBJECTOFFSET(s[HS].constoff) |
                     A5XX_SP_HS_CONFIG_SHADEROBJOFFSET(s[HS].instroff) |
                     COND(s[HS].v, A5XX_SP_HS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_SP_DS_CONFIG_CONSTOBJECTOFFSET(s[DS].constoff) |
                     A5XX_SP_DS_CONFIG_SHADEROBJOFFSET(s[DS].instroff) |
                     COND(s[DS].v, A5XX_SP_DS_CONFIG_ENABLED));
   OUT_RING(ring, A5XX_SP_GS_CONFIG_CONSTOBJECTOFFSET(s[GS].constoff) |
                     A5XX_SP_GS_CONFIG_SHADEROBJOFFSET(s[GS].instroff) |
                     COND(s[GS].v, A5XX_SP_GS_CONFIG_ENABLED));

   OUT_PKT4(ring, REG_A5XX_SP_CS_CONFIG, 1);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_HLSQ_VS_CONSTLEN, 2);
   OUT_RING(ring, s[VS].constlen); /* HLSQ_VS_CONSTLEN */
   OUT_RING(ring, s[VS].instrlen); /* HLSQ_VS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_HLSQ_FS_CONSTLEN, 2);
   OUT_RING(ring, s[FS].constlen); /* HLSQ_FS_CONSTLEN */
   OUT_RING(ring, s[FS].instrlen); /* HLSQ_FS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_HLSQ_HS_CONSTLEN, 2);
   OUT_RING(ring, s[HS].constlen); /* HLSQ_HS_CONSTLEN */
   OUT_RING(ring, s[HS].instrlen); /* HLSQ_HS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_HLSQ_DS_CONSTLEN, 2);
   OUT_RING(ring, s[DS].constlen); /* HLSQ_DS_CONSTLEN */
   OUT_RING(ring, s[DS].instrlen); /* HLSQ_DS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_HLSQ_GS_CONSTLEN, 2);
   OUT_RING(ring, s[GS].constlen); /* HLSQ_GS_CONSTLEN */
   OUT_RING(ring, s[GS].instrlen); /* HLSQ_GS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_HLSQ_CS_CONSTLEN, 2);
   OUT_RING(ring, 0x00000000); /* HLSQ_CS_CONSTLEN */
   OUT_RING(ring, 0x00000000); /* HLSQ_CS_INSTRLEN */

   OUT_PKT4(ring, REG_A5XX_SP_VS_CTRL_REG0, 1);
   OUT_RING(
      ring,
      A5XX_SP_VS_CTRL_REG0_HALFREGFOOTPRINT(s[VS].i->max_half_reg + 1) |
         A5XX_SP_VS_CTRL_REG0_FULLREGFOOTPRINT(s[VS].i->max_reg + 1) |
         COND(s[VS].instrlen != 0, A5XX_SP_VS_CTRL_REG0_BUFFER) |
         /* XXX: 0x2 is only unset in
          * dEQP-GLES3.functional.ubo.single_nested_struct_array.single_buffer.packed_instance_array_vertex
          * on a collection of blob traces.  That shader is 1091 instrs, 0
          * half, 3 full, 108 constlen.  Other >1091 instr non-VS shaders don't
          * unset it, so that's not the trick.
          */
         0x2 |
         A5XX_SP_VS_CTRL_REG0_BRANCHSTACK(ir3_shader_branchstack_hw(s[VS].v)) |
         COND(s[VS].v->need_pixlod, A5XX_SP_VS_CTRL_REG0_PIXLODENABLE));

   /* If we have streamout, link against the real FS in the binning program,
    * rather than the dummy FS used for binning pass state, to ensure the
    * OUTLOC's match.  Depending on whether we end up doing sysmem or gmem, the
    * actual streamout could happen with either the binning pass or draw pass
    * program, but the same streamout stateobj is used in either case:
    */
   const struct ir3_shader_variant *link_fs = s[FS].v;
   if (do_streamout && emit->binning_pass)
      link_fs = emit->fs;
   struct ir3_shader_linkage l = {0};
   ir3_link_shaders(&l, s[VS].v, link_fs, true);

   uint8_t clip0_loc = l.clip0_loc;
   uint8_t clip1_loc = l.clip1_loc;

   OUT_PKT4(ring, REG_A5XX_VPC_VAR_DISABLE(0), 4);
   OUT_RING(ring, ~l.varmask[0]); /* VPC_VAR[0].DISABLE */
   OUT_RING(ring, ~l.varmask[1]); /* VPC_VAR[1].DISABLE */
   OUT_RING(ring, ~l.varmask[2]); /* VPC_VAR[2].DISABLE */
   OUT_RING(ring, ~l.varmask[3]); /* VPC_VAR[3].DISABLE */

   /* Add stream out outputs after computing the VPC_VAR_DISABLE bitmask. */
   ir3_link_stream_out(&l, s[VS].v);

   /* a5xx appends pos/psize to end of the linkage map: */
   if (VALIDREG(pos_regid))
      ir3_link_add(&l, VARYING_SLOT_POS, pos_regid, 0xf, l.max_loc);

   if (VALIDREG(psize_regid)) {
      psize_loc = l.max_loc;
      ir3_link_add(&l, VARYING_SLOT_PSIZ, psize_regid, 0x1, l.max_loc);
   }

   /* Handle the case where clip/cull distances aren't read by the FS. Make
    * sure to avoid adding an output with an empty writemask if the user
    * disables all the clip distances in the API so that the slot is unused.
    */
   if (clip0_loc == 0xff && VALIDREG(clip0_regid) &&
       (clip_cull_mask & 0xf) != 0) {
      clip0_loc = l.max_loc;
      ir3_link_add(&l, VARYING_SLOT_CLIP_DIST0, clip0_regid,
                   clip_cull_mask & 0xf, l.max_loc);
   }

   if (clip1_loc == 0xff && VALIDREG(clip1_regid) &&
       (clip_cull_mask >> 4) != 0) {
      clip1_loc = l.max_loc;
      ir3_link_add(&l, VARYING_SLOT_CLIP_DIST1, clip1_regid,
                   clip_cull_mask >> 4, l.max_loc);
   }

   /* If we have stream-out, we use the full shader for binning
    * pass, rather than the optimized binning pass one, so that we
    * have all the varying outputs available for xfb.  So streamout
    * state should always be derived from the non-binning pass
    * program:
    */
   /* TODO: transform feedback */
   assert(!do_streamout);

   for (i = 0, j = 0; (i < 16) && (j < l.cnt); i++) {
      uint32_t reg = 0;

      OUT_PKT4(ring, REG_A5XX_SP_VS_OUT_REG(i), 1);

      reg |= A5XX_SP_VS_OUT_REG_A_REGID(l.var[j].regid);
      reg |= A5XX_SP_VS_OUT_REG_A_COMPMASK(l.var[j].compmask);
      j++;

      reg |= A5XX_SP_VS_OUT_REG_B_REGID(l.var[j].regid);
      reg |= A5XX_SP_VS_OUT_REG_B_COMPMASK(l.var[j].compmask);
      j++;

      OUT_RING(ring, reg);
   }

   for (i = 0, j = 0; (i < 8) && (j < l.cnt); i++) {
      uint32_t reg = 0;

      OUT_PKT4(ring, REG_A5XX_SP_VS_VPC_DST_REG(i), 1);

      reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC0(l.var[j++].loc);
      reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC1(l.var[j++].loc);
      reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC2(l.var[j++].loc);
      reg |= A5XX_SP_VS_VPC_DST_REG_OUTLOC3(l.var[j++].loc);

      OUT_RING(ring, reg);
   }

   tu5_emit_shader_obj(ring, emit->vs_shader, REG_A5XX_SP_VS_OBJ_START_LO);

   if (s[VS].instrlen)
      tu5_emit_shader(ring, s[VS].v, emit->vs_shader->binary_iova);

   // TODO depending on other bits in this reg (if any) set somewhere else?
   OUT_PKT4(ring, REG_A5XX_PC_PRIM_VTX_CNTL, 1);
   OUT_RING(ring, COND(s[VS].v->writes_psize, A5XX_PC_PRIM_VTX_CNTL_PSIZE));

   OUT_PKT4(ring, REG_A5XX_SP_PRIMITIVE_CNTL, 1);
   OUT_RING(ring, A5XX_SP_PRIMITIVE_CNTL_VSOUT(l.cnt));

   OUT_PKT4(ring, REG_A5XX_VPC_CNTL_0, 1);
   OUT_RING(ring, A5XX_VPC_CNTL_0_STRIDE_IN_VPC(l.max_loc) |
                     COND(s[FS].v->total_in > 0, A5XX_VPC_CNTL_0_VARYING) |
                     0x10000); // XXX

   emit->max_loc = l.max_loc;

   if (emit->binning_pass) {
      OUT_PKT4(ring, REG_A5XX_SP_FS_OBJ_START_LO, 2);
      OUT_RING(ring, 0x00000000); /* SP_FS_OBJ_START_LO */
      OUT_RING(ring, 0x00000000); /* SP_FS_OBJ_START_HI */
   } else {
      tu5_emit_shader_obj(ring, emit->fs_shader, REG_A5XX_SP_FS_OBJ_START_LO);
   }

   OUT_PKT4(ring, REG_A5XX_HLSQ_CONTROL_0_REG, 5);
   OUT_RING(ring, A5XX_HLSQ_CONTROL_0_REG_FSTHREADSIZE(fssz) |
                     A5XX_HLSQ_CONTROL_0_REG_CSTHREADSIZE(TWO_QUADS) |
                     0x00000880); /* XXX HLSQ_CONTROL_0 */
   OUT_RING(ring, A5XX_HLSQ_CONTROL_1_REG_PRIMALLOCTHRESHOLD(63));
   OUT_RING(ring, A5XX_HLSQ_CONTROL_2_REG_FACEREGID(face_regid) |
                     A5XX_HLSQ_CONTROL_2_REG_SAMPLEID(samp_id_regid) |
                     A5XX_HLSQ_CONTROL_2_REG_SAMPLEMASK(samp_mask_regid) |
                     A5XX_HLSQ_CONTROL_2_REG_CENTERRHW(ij_regid[IJ_PERSP_CENTER_RHW]));
   OUT_RING(
      ring,
      A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_PIXEL(ij_regid[IJ_PERSP_PIXEL]) |
         A5XX_HLSQ_CONTROL_3_REG_IJ_LINEAR_PIXEL(ij_regid[IJ_LINEAR_PIXEL]) |
         A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_CENTROID(
            ij_regid[IJ_PERSP_CENTROID]) |
         A5XX_HLSQ_CONTROL_3_REG_IJ_PERSP_CENTROID(
            ij_regid[IJ_LINEAR_CENTROID]));
   OUT_RING(
      ring,
      A5XX_HLSQ_CONTROL_4_REG_XYCOORDREGID(coord_regid) |
         A5XX_HLSQ_CONTROL_4_REG_ZWCOORDREGID(zwcoord_regid) |
         A5XX_HLSQ_CONTROL_4_REG_IJ_PERSP_SAMPLE(ij_regid[IJ_PERSP_SAMPLE]) |
         A5XX_HLSQ_CONTROL_4_REG_IJ_LINEAR_SAMPLE(ij_regid[IJ_LINEAR_SAMPLE]));

   OUT_PKT4(ring, REG_A5XX_SP_FS_CTRL_REG0, 1);
   OUT_RING(
      ring,
      COND(s[FS].v->total_in > 0, A5XX_SP_FS_CTRL_REG0_VARYING) |
         0x40002 | /* XXX set pretty much everywhere */
         COND(s[FS].instrlen != 0, A5XX_SP_FS_CTRL_REG0_BUFFER) |
         A5XX_SP_FS_CTRL_REG0_THREADSIZE(fssz) |
         A5XX_SP_FS_CTRL_REG0_HALFREGFOOTPRINT(s[FS].i->max_half_reg + 1) |
         A5XX_SP_FS_CTRL_REG0_FULLREGFOOTPRINT(s[FS].i->max_reg + 1) |
         A5XX_SP_FS_CTRL_REG0_BRANCHSTACK(ir3_shader_branchstack_hw(s[FS].v)) |
         COND(s[FS].v->need_pixlod, A5XX_SP_FS_CTRL_REG0_PIXLODENABLE));

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0x020fffff); /* XXX */

   OUT_PKT4(ring, REG_A5XX_VPC_GS_SIV_CNTL, 1);
   OUT_RING(ring, 0x0000ffff); /* XXX */

   OUT_PKT4(ring, REG_A5XX_SP_SP_CNTL, 1);
   OUT_RING(ring, 0x00000010); /* XXX */

   OUT_PKT4(ring, REG_A5XX_GRAS_CNTL, 1);
   OUT_RING(ring,
            CONDREG(ij_regid[IJ_PERSP_PIXEL], A5XX_GRAS_CNTL_IJ_PERSP_PIXEL) |
               CONDREG(ij_regid[IJ_PERSP_CENTROID],
                       A5XX_GRAS_CNTL_IJ_PERSP_CENTROID) |
               CONDREG(ij_regid[IJ_PERSP_SAMPLE],
                       A5XX_GRAS_CNTL_IJ_PERSP_SAMPLE) |
               CONDREG(ij_regid[IJ_LINEAR_PIXEL], A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL) |
               CONDREG(ij_regid[IJ_LINEAR_CENTROID],
                       A5XX_GRAS_CNTL_IJ_LINEAR_CENTROID) |
               CONDREG(ij_regid[IJ_LINEAR_SAMPLE],
                       A5XX_GRAS_CNTL_IJ_LINEAR_SAMPLE) |
               COND(s[FS].v->fragcoord_compmask != 0,
                    A5XX_GRAS_CNTL_COORD_MASK(s[FS].v->fragcoord_compmask) |
                       A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL) |
               COND(s[FS].v->frag_face, A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL) |
               CONDREG(ij_regid[IJ_LINEAR_PIXEL], A5XX_GRAS_CNTL_IJ_LINEAR_PIXEL));

   OUT_PKT4(ring, REG_A5XX_RB_RENDER_CONTROL0, 2);
   OUT_RING(
      ring,
      CONDREG(ij_regid[IJ_PERSP_PIXEL],
              A5XX_RB_RENDER_CONTROL0_IJ_PERSP_PIXEL) |
         CONDREG(ij_regid[IJ_PERSP_CENTROID],
                 A5XX_RB_RENDER_CONTROL0_IJ_PERSP_CENTROID) |
         CONDREG(ij_regid[IJ_PERSP_SAMPLE],
                 A5XX_RB_RENDER_CONTROL0_IJ_PERSP_SAMPLE) |
         CONDREG(ij_regid[IJ_LINEAR_PIXEL],
              A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL) |
         CONDREG(ij_regid[IJ_LINEAR_CENTROID],
                 A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_CENTROID) |
         CONDREG(ij_regid[IJ_LINEAR_SAMPLE],
                 A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_SAMPLE) |
         COND(s[FS].v->fragcoord_compmask != 0,
              A5XX_RB_RENDER_CONTROL0_COORD_MASK(s[FS].v->fragcoord_compmask) |
                 A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL) |
         COND(s[FS].v->frag_face, A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL) |
         CONDREG(ij_regid[IJ_LINEAR_PIXEL], A5XX_RB_RENDER_CONTROL0_IJ_LINEAR_PIXEL));
   OUT_RING(ring,
            CONDREG(samp_mask_regid, A5XX_RB_RENDER_CONTROL1_SAMPLEMASK) |
               COND(s[FS].v->frag_face, A5XX_RB_RENDER_CONTROL1_FACENESS) |
               CONDREG(samp_id_regid, A5XX_RB_RENDER_CONTROL1_SAMPLEID));

   OUT_PKT4(ring, REG_A5XX_SP_FS_OUTPUT_REG(0), 8);
   for (i = 0; i < 8; i++) {
      OUT_RING(ring, A5XX_SP_FS_OUTPUT_REG_REGID(color_regid[i]) |
                        COND(color_regid[i] & HALF_REG_ID,
                             A5XX_SP_FS_OUTPUT_REG_HALF_PRECISION));
   }

   OUT_PKT4(ring, REG_A5XX_VPC_PACK, 1);
   OUT_RING(ring, A5XX_VPC_PACK_NUMNONPOSVAR(s[FS].v->total_in) |
                     A5XX_VPC_PACK_PSIZELOC(psize_loc));

   if (!emit->binning_pass) {
      uint32_t vinterp[8], vpsrepl[8];

      memset(vinterp, 0, sizeof(vinterp));
      memset(vpsrepl, 0, sizeof(vpsrepl));

      /* looks like we need to do int varyings in the frag
       * shader on a5xx (no flatshad reg?  or a420.0 bug?):
       *
       *    (sy)(ss)nop
       *    (sy)ldlv.u32 r0.x,l[r0.x], 1
       *    ldlv.u32 r0.y,l[r0.x+1], 1
       *    (ss)bary.f (ei)r63.x, 0, r0.x
       *    (ss)(rpt1)cov.s32f16 hr0.x, (r)r0.x
       *    (rpt5)nop
       *    sam (f16)(xyzw)hr0.x, hr0.x, s#0, t#0
       *
       * Possibly on later a5xx variants we'll be able to use
       * something like the code below instead of workaround
       * in the shader:
       */
      /* figure out VARYING_INTERP / VARYING_PS_REPL register values: */
      for (j = -1;
           (j = ir3_next_varying(s[FS].v, j)) < (int)s[FS].v->inputs_count;) {
         /* NOTE: varyings are packed, so if compmask is 0xb
          * then first, third, and fourth component occupy
          * three consecutive varying slots:
          */
         unsigned compmask = s[FS].v->inputs[j].compmask;

         uint32_t inloc = s[FS].v->inputs[j].inloc;

         if (s[FS].v->inputs[j].flat ||
             (s[FS].v->inputs[j].rasterflat && emit->rasterflat)) {
            uint32_t loc = inloc;

            for (i = 0; i < 4; i++) {
               if (compmask & (1 << i)) {
                  vinterp[loc / 16] |= 1 << ((loc % 16) * 2);
                  // flatshade[loc / 32] |= 1 << (loc % 32);
                  loc++;
               }
            }
         }

         /* gl_PointCoord is the one point sprite varying in Vulkan */
         const bool coord_mode = false;
         if (s[FS].v->inputs[j].slot == VARYING_SLOT_PNTC) {
            /* mask is two 2-bit fields, where:
             *   '01' -> S
             *   '10' -> T
             *   '11' -> 1 - T  (flip mode)
             */
            unsigned mask = coord_mode ? 0b1101 : 0b1001;
            uint32_t loc = inloc;
            if (compmask & 0x1) {
               vpsrepl[loc / 16] |= ((mask >> 0) & 0x3) << ((loc % 16) * 2);
               loc++;
            }
            if (compmask & 0x2) {
               vpsrepl[loc / 16] |= ((mask >> 2) & 0x3) << ((loc % 16) * 2);
               loc++;
            }
            if (compmask & 0x4) {
               /* .z <- 0.0f */
               vinterp[loc / 16] |= 0b10 << ((loc % 16) * 2);
               loc++;
            }
            if (compmask & 0x8) {
               /* .w <- 1.0f */
               vinterp[loc / 16] |= 0b11 << ((loc % 16) * 2);
               loc++;
            }
         }
      }

      OUT_PKT4(ring, REG_A5XX_VPC_VARYING_INTERP_MODE(0), 8);
      for (i = 0; i < 8; i++)
         OUT_RING(ring, vinterp[i]); /* VPC_VARYING_INTERP[i].MODE */

      OUT_PKT4(ring, REG_A5XX_VPC_VARYING_PS_REPL_MODE(0), 8);
      for (i = 0; i < 8; i++)
         OUT_RING(ring, vpsrepl[i]); /* VPC_VARYING_PS_REPL[i] */
   }

   OUT_PKT4(ring, REG_A5XX_GRAS_VS_CL_CNTL, 1);
   OUT_RING(ring, A5XX_GRAS_VS_CL_CNTL_CLIP_MASK(clip_mask) |
                     A5XX_GRAS_VS_CL_CNTL_CULL_MASK(cull_mask));

   OUT_PKT4(ring, REG_A5XX_VPC_CLIP_CNTL, 1);
   OUT_RING(ring, A5XX_VPC_CLIP_CNTL_CLIP_MASK(clip_cull_mask) |
                     A5XX_VPC_CLIP_CNTL_CLIP_DIST_03_LOC(clip0_loc) |
                     A5XX_VPC_CLIP_CNTL_CLIP_DIST_47_LOC(clip1_loc));

   OUT_PKT4(ring, REG_A5XX_PC_CLIP_CNTL, 1);
   OUT_RING(ring, A5XX_PC_CLIP_CNTL_CLIP_MASK(clip_mask));

   if (!emit->binning_pass)
      if (s[FS].instrlen)
         tu5_emit_shader(ring, s[FS].v, emit->fs_shader->binary_iova);

   OUT_PKT4(ring, REG_A5XX_VFD_CONTROL_1, 5);
   OUT_RING(ring, A5XX_VFD_CONTROL_1_REGID4VTX(vertex_regid) |
                     A5XX_VFD_CONTROL_1_REGID4INST(instance_regid) | 0xfc0000);
   OUT_RING(ring, 0x0000fcfc); /* VFD_CONTROL_2 */
   OUT_RING(ring, 0x0000fcfc); /* VFD_CONTROL_3 */
   OUT_RING(ring, 0x000000fc); /* VFD_CONTROL_4 */
   OUT_RING(ring, 0x00000000); /* VFD_CONTROL_5 */
}

void
tu5_init_hw(struct tu_cmd_buffer *cmd, struct tu_cs *ring)
{
   tu5_set_render_mode(ring, BYPASS);
   tu5_cache_flush(ring);

   OUT_PKT4(ring, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(ring, 0xfffff);

   /*
   t7              opcode: CP_PERFCOUNTER_ACTION (50) (4 dwords)
   0000000500024048:               70d08003 00000000 001c5000 00000005
   t7              opcode: CP_PERFCOUNTER_ACTION (50) (4 dwords)
   0000000500024058:               70d08003 00000010 001c7000 00000005

   t7              opcode: CP_WAIT_FOR_IDLE (26) (1 dwords)
   0000000500024068:               70268000
   */

   OUT_PKT4(ring, REG_A5XX_PC_RESTART_INDEX, 1);
   OUT_RING(ring, 0xffffffff);

   OUT_PKT4(ring, REG_A5XX_PC_RASTER_CNTL, 1);
   OUT_RING(ring, 0x00000012);

   OUT_PKT4(ring, REG_A5XX_GRAS_SU_POINT_MINMAX, 2);
   OUT_RING(ring, A5XX_GRAS_SU_POINT_MINMAX_MIN(1.0f) |
                     A5XX_GRAS_SU_POINT_MINMAX_MAX(4092.0f));
   OUT_RING(ring, A5XX_GRAS_SU_POINT_SIZE(0.5f));

   OUT_PKT4(ring, REG_A5XX_GRAS_SU_CONSERVATIVE_RAS_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* GRAS_SU_CONSERVATIVE_RAS_CNTL */

   OUT_PKT4(ring, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* GRAS_SC_SCREEN_SCISSOR_CNTL */

   OUT_PKT4(ring, REG_A5XX_SP_VS_CONFIG_MAX_CONST, 1);
   OUT_RING(ring, 0); /* SP_VS_CONFIG_MAX_CONST */

   OUT_PKT4(ring, REG_A5XX_SP_FS_CONFIG_MAX_CONST, 1);
   OUT_RING(ring, 0); /* SP_FS_CONFIG_MAX_CONST */

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E292, 2);
   OUT_RING(ring, 0x00000000); /* UNKNOWN_E292 */
   OUT_RING(ring, 0x00000000); /* UNKNOWN_E293 */

   OUT_PKT4(ring, REG_A5XX_RB_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000044); /* RB_MODE_CNTL */

   OUT_PKT4(ring, REG_A5XX_RB_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00100000); /* RB_DBG_ECO_CNTL */

   OUT_PKT4(ring, REG_A5XX_VFD_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* VFD_MODE_CNTL */

   OUT_PKT4(ring, REG_A5XX_PC_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001f); /* PC_MODE_CNTL */

   OUT_PKT4(ring, REG_A5XX_SP_MODE_CNTL, 1);
   OUT_RING(ring, 0x0000001e); /* SP_MODE_CNTL */

   if (cmd->device->physical_device->dev_id.gpu_id == 540) {
      OUT_PKT4(ring, REG_A5XX_SP_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x800); /* SP_DBG_ECO_CNTL */

      OUT_PKT4(ring, REG_A5XX_HLSQ_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x0);

      OUT_PKT4(ring, REG_A5XX_VPC_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x800400);
   } else {
      OUT_PKT4(ring, REG_A5XX_SP_DBG_ECO_CNTL, 1);
      OUT_RING(ring, 0x40000800); /* SP_DBG_ECO_CNTL */
   }

   OUT_PKT4(ring, REG_A5XX_TPL1_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000544); /* TPL1_MODE_CNTL */

   OUT_PKT4(ring, REG_A5XX_HLSQ_TIMEOUT_THRESHOLD_0, 2);
   OUT_RING(ring, 0x00000080); /* HLSQ_TIMEOUT_THRESHOLD_0 */
   OUT_RING(ring, 0x00000000); /* HLSQ_TIMEOUT_THRESHOLD_1 */

   OUT_PKT4(ring, REG_A5XX_VPC_DBG_ECO_CNTL, 1);
   OUT_RING(ring, 0x00000400); /* VPC_DBG_ECO_CNTL */

   OUT_PKT4(ring, REG_A5XX_HLSQ_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000001); /* HLSQ_MODE_CNTL */

   OUT_PKT4(ring, REG_A5XX_VPC_MODE_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* VPC_MODE_CNTL */

   /* we don't use this yet.. probably best to disable.. */
   OUT_PKT7(ring, CP_SET_DRAW_STATE, 3);
   OUT_RING(ring, CP_SET_DRAW_STATE__0_COUNT(0) |
                     CP_SET_DRAW_STATE__0_DISABLE_ALL_GROUPS |
                     CP_SET_DRAW_STATE__0_GROUP_ID(0));
   OUT_RING(ring, CP_SET_DRAW_STATE__1_ADDR_LO(0));
   OUT_RING(ring, CP_SET_DRAW_STATE__2_ADDR_HI(0));

   OUT_PKT4(ring, REG_A5XX_GRAS_SU_CONSERVATIVE_RAS_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* GRAS_SU_CONSERVATIVE_RAS_CNTL */

   OUT_PKT4(ring, REG_A5XX_GRAS_SC_BIN_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* GRAS_SC_BIN_CNTL */

   OUT_PKT4(ring, REG_A5XX_GRAS_SC_BIN_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* GRAS_SC_BIN_CNTL */

   OUT_PKT4(ring, REG_A5XX_VPC_FS_PRIMITIVEID_CNTL, 1);
   OUT_RING(ring, 0x000000ff); /* VPC_FS_PRIMITIVEID_CNTL */

   OUT_PKT4(ring, REG_A5XX_VPC_SO_OVERRIDE, 1);
   OUT_RING(ring, A5XX_VPC_SO_OVERRIDE_SO_DISABLE);

   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_BASE_LO(0), 3);
   OUT_RING(ring, 0x00000000); /* VPC_SO_BUFFER_BASE_LO_0 */
   OUT_RING(ring, 0x00000000); /* VPC_SO_BUFFER_BASE_HI_0 */
   OUT_RING(ring, 0x00000000); /* VPC_SO_BUFFER_SIZE_0 */

   OUT_PKT4(ring, REG_A5XX_VPC_SO_FLUSH_BASE_LO(0), 2);
   OUT_RING(ring, 0x00000000); /* VPC_SO_FLUSH_BASE_LO_0 */
   OUT_RING(ring, 0x00000000); /* VPC_SO_FLUSH_BASE_HI_0 */

   OUT_PKT4(ring, REG_A5XX_PC_GS_PARAM, 1);
   OUT_RING(ring, 0x00000000); /* PC_GS_PARAM */

   OUT_PKT4(ring, REG_A5XX_PC_HS_PARAM, 1);
   OUT_RING(ring, 0x00000000); /* PC_HS_PARAM */

   OUT_PKT4(ring, REG_A5XX_TPL1_TP_FS_ROTATION_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* TPL1_TP_FS_ROTATION_CNTL */

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E004, 1);
   OUT_RING(ring, 0x00000000); /* UNKNOWN_E004 */

   OUT_PKT4(ring, REG_A5XX_GRAS_SU_LAYERED, 1);
   OUT_RING(ring, 0x00000000); /* GRAS_SU_LAYERED */

   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUF_CNTL, 1);
   OUT_RING(ring, 0x00000000); /* VPC_SO_BUF_CNTL */

   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(0), 1);
   OUT_RING(ring, 0x00000000); /* UNKNOWN_E2AB */

   OUT_PKT4(ring, REG_A5XX_PC_GS_LAYERED, 1);
   OUT_RING(ring, 0x00000000); /* PC_GS_LAYERED */

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E5AB, 1);
   OUT_RING(ring, 0x00000000); /* UNKNOWN_E5AB */

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E5C2, 1);
   OUT_RING(ring, 0x00000000); /* UNKNOWN_E5C2 */

   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_BASE_LO(1), 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(1), 6);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(2), 6);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_VPC_SO_BUFFER_OFFSET(3), 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E5DB, 1);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_SP_HS_CTRL_REG0, 1);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_SP_GS_CTRL_REG0, 1);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_TPL1_VS_TEX_COUNT, 4);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_TPL1_FS_TEX_COUNT, 2);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E7C0, 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E7C5, 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E7CA, 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E7CF, 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E7D4, 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_UNKNOWN_E7D9, 3);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);
   OUT_RING(ring, 0x00000000);

   OUT_PKT4(ring, REG_A5XX_RB_CLEAR_CNTL, 1);
   OUT_RING(ring, 0x00000000);
}

static void
tu5_emit_mrt(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   const struct tu_subpass *subpass = cmd->state.subpass;
   uint32_t components = 0;

   for (unsigned i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
      enum a5xx_color_fmt format = (enum a5xx_color_fmt)0;
      enum a5xx_tile_mode tile_mode = TILE5_LINEAR;
      enum a3xx_color_swap swap = WZYX;
      bool srgb = false, sint = false, uint = false;
      uint32_t stride = 0;
      uint32_t size = 0;
      uint64_t base = 0;

      if (i < subpass->color_count &&
          subpass->color_attachments[i].attachment != VK_ATTACHMENT_UNUSED) {
         const struct tu_image_view *iview =
            cmd->state.attachments[subpass->color_attachments[i].attachment];
         const struct tu_image *image = iview->image;
         const struct fdl_layout *layout = &image->layout[0];
         enum pipe_format pformat = vk_format_to_pipe_format(iview->vk.format);
         uint32_t level = iview->vk.base_mip_level;

         format = fd5_pipe2color(pformat);
         swap = fd5_pipe2swap(pformat);
         srgb = util_format_is_srgb(pformat);
         sint = util_format_is_pure_sint(pformat);
         uint = util_format_is_pure_uint(pformat);

         stride = fdl_pitch(layout, level);
         size = fdl_layer_stride(layout, level);
         tile_mode = (enum a5xx_tile_mode)fdl_tile_mode(layout, level);
         base = image->iova +
                fdl_surface_offset(layout, level, iview->vk.base_array_layer);

         components |= 0xf << (i * 4);
      }

      OUT_PKT4(cs, REG_A5XX_RB_MRT_BUF_INFO(i), 5);
      OUT_RING(cs, A5XX_RB_MRT_BUF_INFO_COLOR_FORMAT(format) |
                      A5XX_RB_MRT_BUF_INFO_COLOR_TILE_MODE(tile_mode) |
                      A5XX_RB_MRT_BUF_INFO_COLOR_SWAP(swap) |
                      COND(srgb, A5XX_RB_MRT_BUF_INFO_COLOR_SRGB));
      OUT_RING(cs, A5XX_RB_MRT_PITCH(stride));
      OUT_RING(cs, A5XX_RB_MRT_ARRAY_PITCH(size));
      tu_cs_emit_qw(cs, base); /* RB_MRT[i].BASE_LO/HI */

      OUT_PKT4(cs, REG_A5XX_SP_FS_MRT_REG(i), 1);
      OUT_RING(cs, A5XX_SP_FS_MRT_REG_COLOR_FORMAT(format) |
                      COND(sint, A5XX_SP_FS_MRT_REG_COLOR_SINT) |
                      COND(uint, A5XX_SP_FS_MRT_REG_COLOR_UINT) |
                      COND(srgb, A5XX_SP_FS_MRT_REG_COLOR_SRGB));

      /* no UBWC */
      OUT_PKT4(cs, REG_A5XX_RB_MRT_FLAG_BUFFER(i), 4);
      OUT_RING(cs, 0x00000000); /* RB_MRT_FLAG_BUFFER[i].ADDR_LO */
      OUT_RING(cs, 0x00000000); /* RB_MRT_FLAG_BUFFER[i].ADDR_HI */
      OUT_RING(cs, A5XX_RB_MRT_FLAG_BUFFER_PITCH(0));
      OUT_RING(cs, A5XX_RB_MRT_FLAG_BUFFER_ARRAY_PITCH(0));
   }

   OUT_PKT4(cs, REG_A5XX_RB_RENDER_COMPONENTS, 1);
   OUT_RING(cs, components);
}

static void
tu5_emit_zs(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   const struct tu_subpass *subpass = cmd->state.subpass;
   uint32_t a = subpass->depth_stencil_attachment.attachment;

   if (a != VK_ATTACHMENT_UNUSED) {
      const struct tu_image_view *iview = cmd->state.attachments[a];
      const struct tu_image *image = iview->image;
      const struct fdl_layout *layout = &image->layout[0];
      uint32_t level = iview->vk.base_mip_level;
      uint32_t layer = iview->vk.base_array_layer;
      enum a5xx_depth_format fmt =
         fd5_pipe2depth(vk_format_to_pipe_format(image->vk.format));

      OUT_PKT4(cs, REG_A5XX_RB_DEPTH_BUFFER_INFO, 5);
      OUT_RING(cs, A5XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT(fmt));
      tu_cs_emit_qw(cs, image->iova + fdl_surface_offset(layout, level, layer));
      OUT_RING(cs, A5XX_RB_DEPTH_BUFFER_PITCH(fdl_pitch(layout, level)));
      OUT_RING(cs,
               A5XX_RB_DEPTH_BUFFER_ARRAY_PITCH(fdl_layer_stride(layout, level)));

      OUT_PKT4(cs, REG_A5XX_GRAS_SU_DEPTH_BUFFER_INFO, 1);
      OUT_RING(cs, A5XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT(fmt));

      OUT_PKT4(cs, REG_A5XX_RB_DEPTH_FLAG_BUFFER_BASE_LO, 3);
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_FLAG_BUFFER_BASE_LO */
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_FLAG_BUFFER_BASE_HI */
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_FLAG_BUFFER_PITCH */

      /* no LRZ */
      OUT_PKT4(cs, REG_A5XX_GRAS_LRZ_BUFFER_BASE_LO, 3);
      OUT_RING(cs, 0x00000000);
      OUT_RING(cs, 0x00000000);
      OUT_RING(cs, 0x00000000); /* GRAS_LRZ_BUFFER_PITCH */

      OUT_PKT4(cs, REG_A5XX_GRAS_LRZ_FAST_CLEAR_BUFFER_BASE_LO, 2);
      OUT_RING(cs, 0x00000000);
      OUT_RING(cs, 0x00000000);

      if (image->vk.format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
         const struct fdl_layout *slayout = &image->layout[1];

         OUT_PKT4(cs, REG_A5XX_RB_STENCIL_INFO, 5);
         OUT_RING(cs, A5XX_RB_STENCIL_INFO_SEPARATE_STENCIL);
         tu_cs_emit_qw(cs, image->iova +
                              fdl_surface_offset(slayout, level, layer));
         OUT_RING(cs, A5XX_RB_STENCIL_PITCH(fdl_pitch(slayout, level)));
         OUT_RING(cs, A5XX_RB_STENCIL_ARRAY_PITCH(
                         fdl_layer_stride(slayout, level)));
      } else {
         OUT_PKT4(cs, REG_A5XX_RB_STENCIL_INFO, 1);
         OUT_RING(cs, 0x00000000); /* RB_STENCIL_INFO */
      }
   } else {
      OUT_PKT4(cs, REG_A5XX_RB_DEPTH_BUFFER_INFO, 5);
      OUT_RING(cs, A5XX_RB_DEPTH_BUFFER_INFO_DEPTH_FORMAT(DEPTH5_NONE));
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_BUFFER_BASE_LO */
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_BUFFER_BASE_HI */
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_BUFFER_PITCH */
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_BUFFER_ARRAY_PITCH */

      OUT_PKT4(cs, REG_A5XX_GRAS_SU_DEPTH_BUFFER_INFO, 1);
      OUT_RING(cs, A5XX_GRAS_SU_DEPTH_BUFFER_INFO_DEPTH_FORMAT(DEPTH5_NONE));

      OUT_PKT4(cs, REG_A5XX_RB_DEPTH_FLAG_BUFFER_BASE_LO, 3);
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_FLAG_BUFFER_BASE_LO */
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_FLAG_BUFFER_BASE_HI */
      OUT_RING(cs, 0x00000000); /* RB_DEPTH_FLAG_BUFFER_PITCH */

      OUT_PKT4(cs, REG_A5XX_RB_STENCIL_INFO, 1);
      OUT_RING(cs, 0x00000000); /* RB_STENCIL_INFO */
   }
}

static void
tu5_emit_msaa(struct tu_cs *cs, VkSampleCountFlagBits vk_samples)
{
   enum a3xx_msaa_samples samples = tu_msaa_samples(vk_samples);

   OUT_PKT4(cs, REG_A5XX_TPL1_TP_RAS_MSAA_CNTL, 2);
   OUT_RING(cs, A5XX_TPL1_TP_RAS_MSAA_CNTL_SAMPLES(samples));
   OUT_RING(cs, A5XX_TPL1_TP_DEST_MSAA_CNTL_SAMPLES(samples) |
                   COND(samples == MSAA_ONE,
                        A5XX_TPL1_TP_DEST_MSAA_CNTL_MSAA_DISABLE));

   OUT_PKT4(cs, REG_A5XX_RB_RAS_MSAA_CNTL, 2);
   OUT_RING(cs, A5XX_RB_RAS_MSAA_CNTL_SAMPLES(samples));
   OUT_RING(cs, A5XX_RB_DEST_MSAA_CNTL_SAMPLES(samples) |
                   COND(samples == MSAA_ONE,
                        A5XX_RB_DEST_MSAA_CNTL_MSAA_DISABLE));

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_RAS_MSAA_CNTL, 2);
   OUT_RING(cs, A5XX_GRAS_SC_RAS_MSAA_CNTL_SAMPLES(samples));
   OUT_RING(cs, A5XX_GRAS_SC_DEST_MSAA_CNTL_SAMPLES(samples) |
                   COND(samples == MSAA_ONE,
                        A5XX_GRAS_SC_DEST_MSAA_CNTL_MSAA_DISABLE));
}

void
tu5_emit_subpass_begin(struct tu_cmd_buffer *cmd)
{
   struct tu_cs *cs = &cmd->draw_cs;

   tu5_emit_zs(cmd, cs);
   tu5_emit_mrt(cmd, cs);
   tu5_emit_msaa(cs, cmd->state.subpass->samples);

   OUT_PKT4(cs, REG_A5XX_RB_RENDER_CNTL, 1);
   OUT_RING(cs, 0x00000008);

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_CNTL, 1);
   OUT_RING(cs, 0x00000008);
}

/* TODO: tiled rendering through GMEM. Everything is drawn straight into the
 * attachments for now.
 */
void
tu5_cmd_render(struct tu_cmd_buffer *cmd)
{
   const struct fd_dev_info *info = cmd->device->physical_device->info;
   const struct tu_framebuffer *fb = cmd->state.framebuffer;
   struct tu_cs *cs = &cmd->cs;

   tu5_set_render_mode(cs, BYPASS);
   tu5_cache_flush(cs);

   tu5_emit_lrz_flush(cmd, cs);

   OUT_PKT7(cs, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
   OUT_RING(cs, 0x0);

   tu5_event_write(cmd, cs, PC_CCU_INVALIDATE_COLOR, false);

   OUT_PKT4(cs, REG_A5XX_PC_POWER_CNTL, 1);
   OUT_RING(cs, info->num_sp_cores - 1); /* PC_POWER_CNTL */

   OUT_PKT4(cs, REG_A5XX_VFD_POWER_CNTL, 1);
   OUT_RING(cs, info->num_sp_cores - 1); /* VFD_POWER_CNTL */

   /* 0x10000000 for BYPASS.. 0x7c13c080 for GMEM: */
   tu_cs_emit_wfi(cs);
   OUT_PKT4(cs, REG_A5XX_RB_CCU_CNTL, 1);
   OUT_RING(cs, 0x10000000); /* RB_CCU_CNTL */

   OUT_PKT4(cs, REG_A5XX_RB_CNTL, 1);
   OUT_RING(cs, A5XX_RB_CNTL_WIDTH(0) | A5XX_RB_CNTL_HEIGHT(0) |
                   A5XX_RB_CNTL_BYPASS);

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_WINDOW_SCISSOR_TL, 2);
   OUT_RING(cs, A5XX_GRAS_SC_WINDOW_SCISSOR_TL_X(0) |
                   A5XX_GRAS_SC_WINDOW_SCISSOR_TL_Y(0));
   OUT_RING(cs, A5XX_GRAS_SC_WINDOW_SCISSOR_BR_X(fb->width - 1) |
                   A5XX_GRAS_SC_WINDOW_SCISSOR_BR_Y(fb->height - 1));

   OUT_PKT4(cs, REG_A5XX_RB_RESOLVE_CNTL_1, 2);
   OUT_RING(cs, A5XX_RB_RESOLVE_CNTL_1_X(0) | A5XX_RB_RESOLVE_CNTL_1_Y(0));
   OUT_RING(cs, A5XX_RB_RESOLVE_CNTL_2_X(fb->width - 1) |
                   A5XX_RB_RESOLVE_CNTL_2_Y(fb->height - 1));

   OUT_PKT4(cs, REG_A5XX_RB_WINDOW_OFFSET, 1);
   OUT_RING(cs, A5XX_RB_WINDOW_OFFSET_X(0) | A5XX_RB_WINDOW_OFFSET_Y(0));

   OUT_PKT7(cs, CP_SET_VISIBILITY_OVERRIDE, 1);
   OUT_RING(cs, 0x1);

   tu_cs_emit_call(cs, &cmd->draw_cs);

   OUT_PKT7(cs, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
   OUT_RING(cs, 0x0);

   tu5_emit_lrz_flush(cmd, cs);

   tu5_event_write(cmd, cs, PC_CCU_FLUSH_COLOR_TS, true);
   tu5_event_write(cmd, cs, PC_CCU_FLUSH_DEPTH_TS, true);
}

static void
tu5_emit_vertex_bufs(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                     const struct ir3_shader_variant *vp)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyn->vi;
   unsigned j = 0;

   for (unsigned i = 0; i < vp->inputs_count; i++) {
      if (vp->inputs[i].sysval || !vp->inputs[i].compmask)
         continue;

      unsigned loc = vp->inputs[i].slot - VERT_ATTRIB_GENERIC0;
      if (!(vi->attributes_valid & BITFIELD_BIT(loc)))
         continue;

      const struct vk_vertex_attribute_state *attr = &vi->attributes[loc];
      const struct vk_vertex_binding_state *binding =
         &vi->bindings[attr->binding];
      enum pipe_format pfmt = vk_format_to_pipe_format(attr->format);
      enum a5xx_vtx_fmt fmt = fd5_pipe2vtx(pfmt);
      bool isint = util_format_is_pure_integer(pfmt);
      bool instanced = binding->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
      uint64_t base = cmd->state.vb[attr->binding].base + attr->offset;
      uint32_t size = cmd->state.vb[attr->binding].size > attr->offset
                         ? cmd->state.vb[attr->binding].size - attr->offset
                         : 0;

      assert(fmt != VFMT5_NONE);

      OUT_PKT4(cs, REG_A5XX_VFD_FETCH(j), 4);
      tu_cs_emit_qw(cs, base);
      OUT_RING(cs, size); /* VFD_FETCH[j].SIZE */
      OUT_RING(cs, dyn->vi_binding_strides[attr->binding]);

      OUT_PKT4(cs, REG_A5XX_VFD_DECODE(j), 2);
      OUT_RING(cs, A5XX_VFD_DECODE_INSTR_IDX(j) |
                      A5XX_VFD_DECODE_INSTR_FORMAT(fmt) |
                      COND(instanced, A5XX_VFD_DECODE_INSTR_INSTANCED) |
                      A5XX_VFD_DECODE_INSTR_SWAP(fd5_pipe2swap(pfmt)) |
                      A5XX_VFD_DECODE_INSTR_UNK30 |
                      COND(!isint, A5XX_VFD_DECODE_INSTR_FLOAT));
      OUT_RING(cs, instanced ? MAX2(1, binding->divisor) : 1);

      OUT_PKT4(cs, REG_A5XX_VFD_DEST_CNTL(j), 1);
      OUT_RING(cs, A5XX_VFD_DEST_CNTL_INSTR_WRITEMASK(vp->inputs[i].compmask) |
                      A5XX_VFD_DEST_CNTL_INSTR_REGID(vp->inputs[i].regid));

      j++;
   }

   OUT_PKT4(cs, REG_A5XX_VFD_CONTROL_0, 1);
   OUT_RING(cs, A5XX_VFD_CONTROL_0_VTXCNT(j));
}

static void
tu5_emit_viewport_scissor(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const VkViewport *vp = &dyn->vp.viewports[0];
   const VkRect2D *sc = &dyn->vp.scissors[0];
   uint32_t minx = sc->offset.x, miny = sc->offset.y;
   uint32_t maxx = minx + sc->extent.width, maxy = miny + sc->extent.height;

   /* An empty scissor: top left past bottom right */
   if (maxx == minx || maxy == miny) {
      minx = miny = 1;
      maxx = maxy = 1;
   }
   maxx = MIN2(maxx, 16384);
   maxy = MIN2(maxy, 16384);

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0, 2);
   OUT_RING(cs, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(minx) |
                   A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(miny));
   OUT_RING(cs, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(maxx - 1) |
                   A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(maxy - 1));

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0, 2);
   OUT_RING(cs, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(minx) |
                   A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(miny));
   OUT_RING(cs, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(maxx - 1) |
                   A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(maxy - 1));

   float zscale = vp->maxDepth - vp->minDepth, zoffset = vp->minDepth;
   if (dyn->vp.depth_clip_negative_one_to_one) {
      zscale *= 0.5f;
      zoffset = (vp->minDepth + vp->maxDepth) * 0.5f;
   }

   tu_cs_emit_wfi(cs);
   OUT_PKT4(cs, REG_A5XX_GRAS_CL_VPORT_XOFFSET_0, 6);
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_XOFFSET_0(vp->x + vp->width / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_XSCALE_0(vp->width / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_YOFFSET_0(vp->y + vp->height / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_YSCALE_0(vp->height / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_ZOFFSET_0(zoffset));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_ZSCALE_0(zscale));
}

static void
tu5_emit_rasterizer(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                    const struct ir3_shader_variant *vs, unsigned max_loc,
                    bool indexed)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   bool msaa = cmd->state.subpass->samples > VK_SAMPLE_COUNT_1_BIT;
   enum adreno_pa_su_sc_draw polymode =
      dyn->rs.polygon_mode == VK_POLYGON_MODE_POINT  ? PC_DRAW_POINTS :
      dyn->rs.polygon_mode == VK_POLYGON_MODE_LINE   ? PC_DRAW_LINES :
                                                       PC_DRAW_TRIANGLES;
   float psize_min = 1.0f, psize_max = 1.0f;

   if (vs->writes_psize) {
      psize_min = 1.0f;
      psize_max = 4092.0f;
   }

   OUT_PKT4(cs, REG_A5XX_GRAS_SU_CNTL, 1);
   OUT_RING(cs,
            A5XX_GRAS_SU_CNTL_LINEHALFWIDTH(dyn->rs.line.width / 2.0f) |
               COND(dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT,
                    A5XX_GRAS_SU_CNTL_CULL_FRONT) |
               COND(dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT,
                    A5XX_GRAS_SU_CNTL_CULL_BACK) |
               COND(dyn->rs.front_face == VK_FRONT_FACE_CLOCKWISE,
                    A5XX_GRAS_SU_CNTL_FRONT_CW) |
               COND(dyn->rs.depth_bias.enable, A5XX_GRAS_SU_CNTL_POLY_OFFSET) |
               A5XX_GRAS_SU_CNTL_LINE_MODE(msaa ? RECTANGULAR : BRESENHAM));

   OUT_PKT4(cs, REG_A5XX_GRAS_SU_POINT_MINMAX, 2);
   OUT_RING(cs, A5XX_GRAS_SU_POINT_MINMAX_MIN(psize_min) |
                   A5XX_GRAS_SU_POINT_MINMAX_MAX(psize_max));
   OUT_RING(cs, A5XX_GRAS_SU_POINT_SIZE(1.0f));

   OUT_PKT4(cs, REG_A5XX_GRAS_SU_POLY_OFFSET_SCALE, 3);
   OUT_RING(cs, A5XX_GRAS_SU_POLY_OFFSET_SCALE(dyn->rs.depth_bias.slope));
   OUT_RING(cs,
            A5XX_GRAS_SU_POLY_OFFSET_OFFSET(dyn->rs.depth_bias.constant));
   OUT_RING(cs, A5XX_GRAS_SU_POLY_OFFSET_OFFSET_CLAMP(dyn->rs.depth_bias.clamp));

   OUT_PKT4(cs, REG_A5XX_PC_RASTER_CNTL, 1);
   OUT_RING(cs, A5XX_PC_RASTER_CNTL_POLYMODE_FRONT_PTYPE(polymode) |
                   A5XX_PC_RASTER_CNTL_POLYMODE_BACK_PTYPE(polymode) |
                   COND(polymode != PC_DRAW_TRIANGLES,
                        A5XX_PC_RASTER_CNTL_POLYMODE_ENABLE));

   /* The clip space z of Vulkan is [0, w] */
   OUT_PKT4(cs, REG_A5XX_GRAS_CL_CNTL, 1);
   OUT_RING(cs, COND(!dyn->vp.depth_clip_negative_one_to_one,
                     A5XX_GRAS_CL_CNTL_ZERO_GB_SCALE_Z));

   OUT_PKT4(cs, REG_A5XX_PC_PRIMITIVE_CNTL, 1);
   OUT_RING(cs,
            COND(dyn->rs.provoking_vertex ==
                    VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT,
                 A5XX_PC_PRIMITIVE_CNTL_PROVOKING_VTX_LAST) |
               A5XX_PC_PRIMITIVE_CNTL_STRIDE_IN_VPC(max_loc) |
               COND(dyn->ia.primitive_restart_enable && indexed,
                    A5XX_PC_PRIMITIVE_CNTL_PRIMITIVE_RESTART));
}

static uint32_t
tu5_stencil_control(const struct vk_stencil_test_face_state *front,
                    const struct vk_stencil_test_face_state *back)
{
   return A5XX_RB_STENCIL_CONTROL_STENCIL_READ |
          A5XX_RB_STENCIL_CONTROL_STENCIL_ENABLE |
          A5XX_RB_STENCIL_CONTROL_STENCIL_ENABLE_BF |
          A5XX_RB_STENCIL_CONTROL_FUNC(
             tu6_compare_func((VkCompareOp)front->op.compare)) |
          A5XX_RB_STENCIL_CONTROL_FAIL(
             tu6_stencil_op((VkStencilOp)front->op.fail)) |
          A5XX_RB_STENCIL_CONTROL_ZPASS(
             tu6_stencil_op((VkStencilOp)front->op.pass)) |
          A5XX_RB_STENCIL_CONTROL_ZFAIL(
             tu6_stencil_op((VkStencilOp)front->op.depth_fail)) |
          A5XX_RB_STENCIL_CONTROL_FUNC_BF(
             tu6_compare_func((VkCompareOp)back->op.compare)) |
          A5XX_RB_STENCIL_CONTROL_FAIL_BF(
             tu6_stencil_op((VkStencilOp)back->op.fail)) |
          A5XX_RB_STENCIL_CONTROL_ZPASS_BF(
             tu6_stencil_op((VkStencilOp)back->op.pass)) |
          A5XX_RB_STENCIL_CONTROL_ZFAIL_BF(
             tu6_stencil_op((VkStencilOp)back->op.depth_fail));
}

static void
tu5_emit_depth_stencil(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                       const struct ir3_shader_variant *fp)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   bool has_zs = cmd->state.subpass->depth_stencil_attachment.attachment !=
                 VK_ATTACHMENT_UNUSED;
   bool depth_test = has_zs && dyn->ds.depth.test_enable;
   bool stencil_test = has_zs && dyn->ds.stencil.test_enable;
   bool fragz = fp->no_earlyz || fp->has_kill || fp->writes_pos ||
                dyn->ms.alpha_to_coverage_enable;

   OUT_PKT4(cs, REG_A5XX_RB_ALPHA_CONTROL, 1);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_RB_STENCIL_CONTROL, 1);
   OUT_RING(cs, stencil_test ? tu5_stencil_control(&dyn->ds.stencil.front,
                                                  &dyn->ds.stencil.back)
                             : 0);

   OUT_PKT4(cs, REG_A5XX_GRAS_LRZ_CNTL, 1);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_RB_STENCILREFMASK, 2);
   OUT_RING(cs, A5XX_RB_STENCILREFMASK_STENCILWRITEMASK(
                   dyn->ds.stencil.front.write_mask) |
                   A5XX_RB_STENCILREFMASK_STENCILMASK(
                      dyn->ds.stencil.front.compare_mask) |
                   A5XX_RB_STENCILREFMASK_STENCILREF(
                      dyn->ds.stencil.front.reference));
   OUT_RING(cs, A5XX_RB_STENCILREFMASK_BF_STENCILWRITEMASK(
                   dyn->ds.stencil.back.write_mask) |
                   A5XX_RB_STENCILREFMASK_BF_STENCILMASK(
                      dyn->ds.stencil.back.compare_mask) |
                   A5XX_RB_STENCILREFMASK_BF_STENCILREF(
                      dyn->ds.stencil.back.reference));

   OUT_PKT4(cs, REG_A5XX_RB_DEPTH_CNTL, 1);
   OUT_RING(cs, A5XX_RB_DEPTH_CNTL_ZFUNC(tu6_compare_func(
                   (VkCompareOp)dyn->ds.depth.compare_op)) |
                   COND(depth_test, A5XX_RB_DEPTH_CNTL_Z_TEST_ENABLE |
                                       A5XX_RB_DEPTH_CNTL_Z_READ_ENABLE) |
                   COND(depth_test && dyn->ds.depth.write_enable,
                        A5XX_RB_DEPTH_CNTL_Z_WRITE_ENABLE));

   OUT_PKT4(cs, REG_A5XX_RB_DEPTH_PLANE_CNTL, 1);
   OUT_RING(cs, COND(fragz, A5XX_RB_DEPTH_PLANE_CNTL_FRAG_WRITES_Z) |
                   COND(fragz && fp->fragcoord_compmask != 0,
                        A5XX_RB_DEPTH_PLANE_CNTL_UNK1));

   OUT_PKT4(cs, REG_A5XX_GRAS_SU_DEPTH_PLANE_CNTL, 1);
   OUT_RING(cs, COND(fragz, A5XX_GRAS_SU_DEPTH_PLANE_CNTL_FRAG_WRITES_Z) |
                   COND(fragz && fp->fragcoord_compmask != 0,
                        A5XX_GRAS_SU_DEPTH_PLANE_CNTL_UNK1));
}

static void
tu5_emit_blend(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
               const struct ir3_shader_variant *fp)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const struct tu_subpass *subpass = cmd->state.subpass;
   enum a3xx_rop_code rop = ROP_COPY;
   uint32_t mrt_blend = 0;
   unsigned nr = dyn->rs.rasterizer_discard_enable ? 0 : subpass->color_count;

   if (dyn->cb.logic_op_enable)
      rop = tu6_rop((VkLogicOp)dyn->cb.logic_op);

   for (unsigned i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
      const struct vk_color_blend_attachment_state *att =
         &dyn->cb.attachments[i];
      uint32_t control = 0, blend_control = 0;

      if (i < subpass->color_count &&
          subpass->color_attachments[i].attachment != VK_ATTACHMENT_UNUSED) {
         const struct tu_image_view *iview =
            cmd->state.attachments[subpass->color_attachments[i].attachment];
         enum pipe_format format = vk_format_to_pipe_format(iview->vk.format);
         bool is_int = util_format_is_pure_integer(format);
         bool has_alpha = util_format_has_alpha(format);
         uint32_t write_mask =
            (dyn->cb.color_write_enables & BITFIELD_BIT(i)) ? att->write_mask
                                                           : 0;

         blend_control =
            A5XX_RB_MRT_BLEND_CONTROL_RGB_SRC_FACTOR(tu6_blend_factor(
               (VkBlendFactor)att->src_color_blend_factor)) |
            A5XX_RB_MRT_BLEND_CONTROL_RGB_BLEND_OPCODE(
               tu6_blend_op((VkBlendOp)att->color_blend_op)) |
            A5XX_RB_MRT_BLEND_CONTROL_RGB_DEST_FACTOR(tu6_blend_factor(
               (VkBlendFactor)att->dst_color_blend_factor)) |
            A5XX_RB_MRT_BLEND_CONTROL_ALPHA_SRC_FACTOR(tu6_blend_factor(
               (VkBlendFactor)att->src_alpha_blend_factor)) |
            A5XX_RB_MRT_BLEND_CONTROL_ALPHA_BLEND_OPCODE(
               tu6_blend_op((VkBlendOp)att->alpha_blend_op)) |
            A5XX_RB_MRT_BLEND_CONTROL_ALPHA_DEST_FACTOR(tu6_blend_factor(
               (VkBlendFactor)att->dst_alpha_blend_factor));

         control = A5XX_RB_MRT_CONTROL_COMPONENT_ENABLE(write_mask);
         if (is_int) {
            control |= A5XX_RB_MRT_CONTROL_ROP_CODE(ROP_COPY);
         } else {
            control |= A5XX_RB_MRT_CONTROL_ROP_CODE(rop) |
                       COND(dyn->cb.logic_op_enable,
                            A5XX_RB_MRT_CONTROL_ROP_ENABLE);
            if (att->blend_enable && !dyn->cb.logic_op_enable) {
               control |= A5XX_RB_MRT_CONTROL_BLEND |
                          COND(has_alpha, A5XX_RB_MRT_CONTROL_BLEND2);
               mrt_blend |= 1 << i;
            }
            if (dyn->cb.logic_op_enable &&
                dyn->cb.logic_op != VK_LOGIC_OP_CLEAR &&
                dyn->cb.logic_op != VK_LOGIC_OP_COPY &&
                dyn->cb.logic_op != VK_LOGIC_OP_COPY_INVERTED &&
                dyn->cb.logic_op != VK_LOGIC_OP_SET)
               mrt_blend |= 1 << i;
         }
      }

      OUT_PKT4(cs, REG_A5XX_RB_MRT_CONTROL(i), 1);
      OUT_RING(cs, control);

      OUT_PKT4(cs, REG_A5XX_RB_MRT_BLEND_CONTROL(i), 1);
      OUT_RING(cs, blend_control);
   }

   OUT_PKT4(cs, REG_A5XX_SP_BLEND_CNTL, 1);
   OUT_RING(cs, A5XX_SP_BLEND_CNTL_ENABLE_BLEND(mrt_blend) |
                   A5XX_SP_BLEND_CNTL_UNK8 |
                   COND(dyn->ms.alpha_to_coverage_enable,
                        A5XX_SP_BLEND_CNTL_ALPHA_TO_COVERAGE));

   OUT_PKT4(cs, REG_A5XX_RB_BLEND_CNTL, 1);
   OUT_RING(cs, A5XX_RB_BLEND_CNTL_ENABLE_BLEND(mrt_blend) |
                   COND(dyn->ms.alpha_to_coverage_enable,
                        A5XX_RB_BLEND_CNTL_ALPHA_TO_COVERAGE) |
                   A5XX_RB_BLEND_CNTL_INDEPENDENT_BLEND |
                   A5XX_RB_BLEND_CNTL_SAMPLE_MASK(dyn->ms.sample_mask & 0xffff));

   const float *bc = dyn->cb.blend_constants;
   OUT_PKT4(cs, REG_A5XX_RB_BLEND_RED, 8);
   OUT_RING(cs, A5XX_RB_BLEND_RED_FLOAT(bc[0]) |
                   A5XX_RB_BLEND_RED_UINT(CLAMP(bc[0], 0.f, 1.f) * 0xff) |
                   A5XX_RB_BLEND_RED_SINT(CLAMP(bc[0], -1.f, 1.f) * 0x7f));
   OUT_RING(cs, A5XX_RB_BLEND_RED_F32(bc[0]));
   OUT_RING(cs, A5XX_RB_BLEND_GREEN_FLOAT(bc[1]) |
                   A5XX_RB_BLEND_GREEN_UINT(CLAMP(bc[1], 0.f, 1.f) * 0xff) |
                   A5XX_RB_BLEND_GREEN_SINT(CLAMP(bc[1], -1.f, 1.f) * 0x7f));
   OUT_RING(cs, A5XX_RB_BLEND_RED_F32(bc[1]));
   OUT_RING(cs, A5XX_RB_BLEND_BLUE_FLOAT(bc[2]) |
                   A5XX_RB_BLEND_BLUE_UINT(CLAMP(bc[2], 0.f, 1.f) * 0xff) |
                   A5XX_RB_BLEND_BLUE_SINT(CLAMP(bc[2], -1.f, 1.f) * 0x7f));
   OUT_RING(cs, A5XX_RB_BLEND_BLUE_F32(bc[2]));
   OUT_RING(cs, A5XX_RB_BLEND_ALPHA_FLOAT(bc[3]) |
                   A5XX_RB_BLEND_ALPHA_UINT(CLAMP(bc[3], 0.f, 1.f) * 0xff) |
                   A5XX_RB_BLEND_ALPHA_SINT(CLAMP(bc[3], -1.f, 1.f) * 0x7f));
   OUT_RING(cs, A5XX_RB_BLEND_ALPHA_F32(bc[3]));

   uint32_t posz_regid = ir3_find_output_regid(fp, FRAG_RESULT_DEPTH);

   OUT_PKT4(cs, REG_A5XX_RB_FS_OUTPUT_CNTL, 1);
   OUT_RING(cs, A5XX_RB_FS_OUTPUT_CNTL_MRT(nr) |
                   COND(fp->writes_pos, A5XX_RB_FS_OUTPUT_CNTL_FRAG_WRITES_Z));

   OUT_PKT4(cs, REG_A5XX_SP_FS_OUTPUT_CNTL, 1);
   OUT_RING(cs, A5XX_SP_FS_OUTPUT_CNTL_MRT(nr) |
                   A5XX_SP_FS_OUTPUT_CNTL_DEPTH_REGID(posz_regid) |
                   A5XX_SP_FS_OUTPUT_CNTL_SAMPLEMASK_REGID(regid(63, 0)));
}

/* TODO: only what changed. Everything is written out for each draw for now,
 * as the draw stream is not built from draw state groups here.
 */
VkResult
tu5_draw_state(struct tu_cmd_buffer *cmd, struct tu_cs *cs, bool indexed)
{
   const struct tu_shader *vs_shader = cmd->state.shaders[MESA_SHADER_VERTEX];
   const struct tu_shader *fs_shader = cmd->state.shaders[MESA_SHADER_FRAGMENT];
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   struct tu5_program prog = {
      .vs = vs_shader->variant,
      .fs = fs_shader->variant,
      .vs_shader = vs_shader,
      .fs_shader = fs_shader,
   };

   tu5_emit_depth_stencil(cmd, cs, prog.fs);
   tu5_emit_viewport_scissor(cmd, cs);
   tu5_program_emit(cs, &prog);
   tu5_emit_rasterizer(cmd, cs, prog.vs, prog.max_loc, indexed);
   tu5_emit_blend(cmd, cs, prog.fs);
   tu5_emit_vertex_bufs(cmd, cs, prog.vs);

   tu5_emit_immediates(cs, prog.vs);
   tu5_emit_vs_params(cmd, cs, prog.vs);
   tu5_emit_immediates(cs, prog.fs);

   OUT_PKT4(cs, REG_A5XX_VFD_INDEX_OFFSET, 2);
   OUT_RING(cs, cmd->state.last_vs_params.vertex_offset);
   OUT_RING(cs, cmd->state.last_vs_params.first_instance);

   /* TODO: textures, and with them the border colors */
   OUT_PKT4(cs, REG_A5XX_TPL1_VS_TEX_COUNT, 1);
   OUT_RING(cs, 0);
   OUT_PKT4(cs, REG_A5XX_TPL1_FS_TEX_COUNT, 1);
   OUT_RING(cs, 0);
   OUT_PKT4(cs, REG_A5XX_TPL1_CS_TEX_COUNT, 1);
   OUT_RING(cs, 0);

   cmd->state.dirty = (enum tu_cmd_dirty_bits)0;
   vk_dynamic_graphics_state_clear_dirty(&cmd->vk.dynamic_graphics_state);
   (void)dyn;

   return VK_SUCCESS;
}
