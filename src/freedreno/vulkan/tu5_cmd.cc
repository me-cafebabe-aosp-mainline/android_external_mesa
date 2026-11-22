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

#include "tu_buffer.h"
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

/* A shader as the hardware gets it: where its instructions are, and its
 * private memory if it has any.
 */
struct tu5_shader {
   const struct ir3_shader_variant *v;
   uint64_t iova;
   uint64_t pvtmem_iova;
   uint32_t pvtmem_per_fiber_size;
   uint32_t pvtmem_per_sp_size;
};

static struct tu5_shader
tu5_shader_from(const struct tu_shader *shader)
{
   return (struct tu5_shader) {
      .v = shader->variant,
      .iova = shader->binary_iova,
      .pvtmem_iova = shader->pvtmem.iova,
      .pvtmem_per_fiber_size = shader->pvtmem.per_fiber_size,
      .pvtmem_per_sp_size = shader->pvtmem.per_sp_size,
   };
}

static struct tu5_shader
tu5_global_shader(const struct tu_device *dev, enum global_shader id)
{
   return (struct tu5_shader) {
      .v = dev->global_shader_variants[id],
      .iova = dev->global_shader_va[id],
   };
}

static void
tu5_emit_shader_obj(struct tu_cs *cs, const struct tu5_shader *shader,
                    uint32_t shader_obj_reg)
{
   OUT_PKT4(cs, shader_obj_reg, 6);
   tu_cs_emit_qw(cs, shader->iova); /* SP_VS_OBJ_START_LO/HI */
   OUT_RING(cs, A5XX_SP_VS_PVT_MEM_PARAM_MEMSIZEPERITEM(
                   shader->pvtmem_per_fiber_size) |
                   A5XX_SP_VS_PVT_MEM_PARAM_HWSTACKOFFSET(
                      shader->pvtmem_per_sp_size));
   tu_cs_emit_qw(cs, shader->pvtmem_iova); /* SP_xS_PVT_MEM_ADDR */
   OUT_RING(cs,
            A5XX_SP_VS_PVT_MEM_SIZE_TOTALPVTMEMSIZE(shader->pvtmem_per_sp_size));
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

/* Where a descriptor a shader uses is, or NULL with no set bound */
static const uint32_t *
tu5_descriptor(struct tu_cmd_buffer *cmd, const struct tu_shader *shader,
               const struct tu5_desc_ref *ref)
{
   struct tu_descriptor_state *descriptors = tu_get_descriptors_state(
      cmd, shader->variant->type == MESA_SHADER_COMPUTE
              ? VK_PIPELINE_BIND_POINT_COMPUTE
              : VK_PIPELINE_BIND_POINT_GRAPHICS);

   if (ref->dynamic)
      return &descriptors->dynamic_descriptors[ref->offset];

   const struct tu_descriptor_set *set = descriptors->sets[ref->set];

   return set ? set->mapped_ptr + ref->offset : NULL;
}

static uint64_t
tu5_ubo_address(struct tu_cmd_buffer *cmd, const struct tu_shader *shader,
                const struct tu5_desc_ref *ref)
{
   const uint32_t *desc = tu5_descriptor(cmd, shader, ref);

   if (!desc)
      return 0;

   if (ref->storage) {
      /* see write_buffer_descriptor_addr() */
      uint64_t base = desc[4] | ((uint64_t)(desc[5] & 0x1ffff) << 32);
      uint32_t offset = ((desc[2] & A6XX_TEX_CONST_2_STARTOFFSETTEXELS__MASK) >>
                         A6XX_TEX_CONST_2_STARTOFFSETTEXELS__SHIFT) * 4;

      return base ? base + offset : 0;
   }

   /* see write_ubo_descriptor_addr() */
   return desc[0] | ((uint64_t)(desc[1] & 0x1ffff) << 32);
}

/* The textures and the samplers of a shader */
static void
tu5_emit_textures(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                  const struct tu_shader *shader)
{
   const struct tu5_desc_map *map = &shader->const_state.a5xx;
   gl_shader_stage stage = shader->variant->type;
   enum a4xx_state_block sb = stage == MESA_SHADER_VERTEX ? SB4_VS_TEX :
                              stage == MESA_SHADER_FRAGMENT ? SB4_FS_TEX :
                                                              SB4_CS_TEX;
   unsigned num_tex = map->count[TU5_DESC_TEXTURE];
   unsigned num_samp = map->count[TU5_DESC_SAMPLER];

   /* What is fetched without a sampler names the first one */
   if (num_tex && !num_samp)
      num_samp = 1;

   if (num_samp) {
      OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 4 * num_samp);
      OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(0) |
                      CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                      CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                      CP_LOAD_STATE4_0_NUM_UNIT(num_samp));
      OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_SHADER) |
                      CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
      OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
      for (unsigned i = 0; i < num_samp; i++) {
         const uint32_t *desc = i < map->count[TU5_DESC_SAMPLER]
                                   ? tu5_descriptor(cmd, shader, &map->samp[i]) : NULL;

         /* The same as those of Adreno 6xx, of which the third one has the
          * border colour alone here: its number among those of the device.
          */
         OUT_RING(cs, desc ? desc[0] : 0);
         OUT_RING(cs, desc ? desc[1] : 0);
         OUT_RING(cs, desc ? desc[2] & A5XX_TEX_SAMP_2_BCOLOR_OFFSET__MASK : 0);
         OUT_RING(cs, 0x00000000);
      }
   }

   if (num_tex) {
      OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 12 * num_tex);
      OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(0) |
                      CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                      CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                      CP_LOAD_STATE4_0_NUM_UNIT(num_tex));
      OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                      CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
      OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
      for (unsigned i = 0; i < num_tex; i++) {
         const uint32_t *desc = tu5_descriptor(cmd, shader, &map->tex[i]);

         /* see tu5_image_view_descriptor(): the first six, what follows
          * them in the descriptor being for the storage images
          */
         for (unsigned j = 0; j < 6; j++)
            OUT_RING(cs, desc ? desc[j] : 0);
         for (unsigned j = 6; j < 12; j++)
            OUT_RING(cs, 0x00000000);
      }
   }

   OUT_PKT4(cs, stage == MESA_SHADER_VERTEX ? REG_A5XX_TPL1_VS_TEX_COUNT :
                stage == MESA_SHADER_FRAGMENT ? REG_A5XX_TPL1_FS_TEX_COUNT :
                                                REG_A5XX_TPL1_CS_TEX_COUNT, 1);
   OUT_RING(cs, num_tex);
}

/* The storage buffers of a fragment or of a compute shader */
static void
tu5_emit_ssbos(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
               const struct tu_shader *shader)
{
   const struct tu5_desc_map *map = &shader->const_state.a5xx;
   unsigned count = map->count[TU5_DESC_SSBO];
   enum a4xx_state_block sb;

   if (!count)
      return;

   switch (shader->variant->type) {
   case MESA_SHADER_FRAGMENT:
      sb = SB4_SSBO;
      break;
   case MESA_SHADER_COMPUTE:
      sb = SB4_CS_SSBO;
      break;
   default:
      return;
   }

   uint64_t va[TU5_MAX_SSBOS];
   uint32_t size[TU5_MAX_SSBOS];

   for (unsigned i = 0; i < count; i++) {
      const uint32_t *desc = tu5_descriptor(cmd, shader, &map->ssbo[i]);

      va[i] = 0;
      size[i] = 0;
      if (!desc)
         continue;

      /* see write_buffer_descriptor_addr() */
      uint64_t base = desc[4] | ((uint64_t)(desc[5] & 0x1ffff) << 32);
      uint32_t offset = ((desc[2] & A6XX_TEX_CONST_2_STARTOFFSETTEXELS__MASK) >>
                         A6XX_TEX_CONST_2_STARTOFFSETTEXELS__SHIFT) * 4;

      if (!base)
         continue;
      va[i] = base + offset;
      size[i] = desc[1] * 4;
   }

   /* in bytes */
   OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 2 * count);
   OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(0) |
                   CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                   CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                   CP_LOAD_STATE4_0_NUM_UNIT(count));
   OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                   CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
   OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   for (unsigned i = 0; i < count; i++) {
      OUT_RING(cs, A5XX_SSBO_1_0_WIDTH(size[i] & 0xffff));
      OUT_RING(cs, A5XX_SSBO_1_1_HEIGHT(size[i] >> 16));
   }

   OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 2 * count);
   OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(0) |
                   CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                   CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                   CP_LOAD_STATE4_0_NUM_UNIT(count));
   OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_UBO) |
                   CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
   OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   for (unsigned i = 0; i < count; i++)
      tu_cs_emit_qw(cs, va[i]);
}

/* The storage images of a fragment or of a compute shader: these are written
 * as storage buffers are, after them, and read as textures, after the
 * textures. Their sizes are constants of the shader.
 */
static void
tu5_emit_images(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                const struct tu_shader *shader)
{
   const struct ir3_shader_variant *v = shader->variant;
   const struct ir3_const_state *const_state = ir3_const_state(v);
   const struct tu5_desc_map *map = &shader->const_state.a5xx;
   const struct ir3_ibo_mapping *m = &v->image_mapping;
   unsigned count = map->count[TU5_DESC_IMAGE];
   enum a4xx_state_block sb, texsb;
   uint32_t texcount_reg;

   if (!count)
      return;

   switch (v->type) {
   case MESA_SHADER_FRAGMENT:
      sb = SB4_SSBO;
      texsb = SB4_FS_TEX;
      texcount_reg = REG_A5XX_TPL1_FS_TEX_COUNT;
      break;
   case MESA_SHADER_COMPUTE:
      sb = SB4_CS_SSBO;
      texsb = SB4_CS_TEX;
      texcount_reg = REG_A5XX_TPL1_CS_TEX_COUNT;
      break;
   default:
      return;
   }

   uint32_t dims[align(MAX2(const_state->image_dims.count, 1), 4)];
   uint32_t num_tex = map->count[TU5_DESC_TEXTURE];

   memset(dims, 0, sizeof(dims));

   for (unsigned i = 0; i < count; i++) {
      static const uint32_t none[A6XX_TEX_CONST_DWORDS] = {};
      const uint32_t *desc = tu5_descriptor(cmd, shader, &map->image[i]);

      /* see tu5_image_view_storage_descriptor() */
      if (!desc)
         desc = none;

      if (m->image_to_tex[i] != IBO_INVALID) {
         unsigned slot = m->tex_base + m->image_to_tex[i];

         OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 12);
         OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(slot) |
                         CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                         CP_LOAD_STATE4_0_STATE_BLOCK(texsb) |
                         CP_LOAD_STATE4_0_NUM_UNIT(1));
         OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                         CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
         OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
         for (unsigned j = 0; j < 6; j++)
            OUT_RING(cs, desc[j]);
         for (unsigned j = 6; j < 12; j++)
            OUT_RING(cs, 0x00000000);

         num_tex = MAX2(num_tex, slot + 1);
      }

      OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 2);
      OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(v->num_ssbos + i) |
                      CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                      CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                      CP_LOAD_STATE4_0_NUM_UNIT(1));
      OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                      CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
      OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
      OUT_RING(cs, desc[6]); /* SSBO_1_0 */
      OUT_RING(cs, desc[7]); /* SSBO_1_1 */

      OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 2);
      OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(v->num_ssbos + i) |
                      CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                      CP_LOAD_STATE4_0_STATE_BLOCK(sb) |
                      CP_LOAD_STATE4_0_NUM_UNIT(1));
      OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_UBO) |
                      CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
      OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
      OUT_RING(cs, desc[4]);
      OUT_RING(cs, desc[5] & A5XX_TEX_CONST_5_BASE_HI__MASK);

      if (const_state->image_dims.mask & (1u << i)) {
         unsigned off = const_state->image_dims.off[i];

         dims[off + 0] = desc[8];  /* bytes a texel */
         dims[off + 1] = desc[9];  /* bytes a row */
         dims[off + 2] = desc[10]; /* bytes a layer */
      }
   }

   /* after tu5_emit_textures() */
   OUT_PKT4(cs, texcount_reg, 1);
   OUT_RING(cs, num_tex);

   uint32_t offset = const_state->offsets.image_dims;

   if (const_state->image_dims.count && offset < v->constlen) {
      uint32_t dwords = MIN2((uint32_t)ARRAY_SIZE(dims),
                             (v->constlen - offset) * 4);

      tu5_emit_const(cs, v, offset * 4, dwords, dims);
   }
}

/* The constants of a shader that come from the descriptors and from the
 * push constants.
 */
static void
tu5_emit_user_consts(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                     const struct tu_shader *shader)
{
   const struct ir3_shader_variant *v = shader->variant;
   const struct ir3_const_state *const_state = ir3_const_state(v);
   const struct tu5_desc_map *map = &shader->const_state.a5xx;
   const struct tu_push_constant_range *push = &shader->const_state.push_consts;

   if (push->type == IR3_PUSH_CONSTS_PER_STAGE && push->dwords &&
       push->lo < v->constlen * 4) {
      uint32_t dwords = MIN2(push->dwords, v->constlen * 4 - push->lo);

      tu5_emit_const(cs, v, push->lo, dwords, &cmd->push_constants[push->lo]);
   }

   /* The ranges of the uniform buffers read as constants */
   const struct ir3_ubo_analysis_state *state = &const_state->ubo_state;

   for (unsigned i = 0; i < state->num_enabled; i++) {
      const struct ir3_ubo_range *range = &state->range[i];
      uint32_t offset = range->offset;
      uint32_t size = range->end - range->start;

      if (offset >= v->constlen * 16)
         continue;
      size = MIN2(size, v->constlen * 16 - offset);

      if (range->ubo.bindless)
         continue;

      uint64_t va;

      if ((int)range->ubo.block == const_state->consts_ubo.idx) {
         /* the constant data of the shader, which follows its instructions */
         va = shader->binary_iova + v->info.constant_data_offset;
      } else if (range->ubo.block < map->count[TU5_DESC_UBO]) {
         va = tu5_ubo_address(cmd, shader, &map->ubo[range->ubo.block]);
      } else {
         continue;
      }

      if (!va)
         continue;
      va += range->start;

      OUT_PKT7(cs, CP_LOAD_STATE4, 3);
      OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(offset / 16) |
                      CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) |
                      CP_LOAD_STATE4_0_STATE_BLOCK(tu5_stage2shadersb(v->type)) |
                      CP_LOAD_STATE4_0_NUM_UNIT(size / 16));
      OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) | (uint32_t)va);
      OUT_RING(cs, (uint32_t)(va >> 32));
   }

   /* The addresses of the uniform buffers, and of the constant data of the
    * shader, which is read the same way.
    */
   uint32_t ptrs = const_state->offsets.ubo;

   if (const_state->num_ubos && ptrs < v->constlen) {
      uint32_t num = const_state->num_ubos;
      uint32_t anum = align(num, 2);

      OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 2 * anum);
      OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(ptrs) |
                      CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                      CP_LOAD_STATE4_0_STATE_BLOCK(tu5_stage2shadersb(v->type)) |
                      CP_LOAD_STATE4_0_NUM_UNIT(anum / 2));
      OUT_RING(cs, CP_LOAD_STATE4_1_EXT_SRC_ADDR(0) |
                      CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS));
      OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
      for (uint32_t i = 0; i < anum; i++) {
         uint64_t va = 0;

         if (i < num && (int)i == const_state->consts_ubo.idx)
            va = shader->binary_iova + v->info.constant_data_offset;
         else if (i < map->count[TU5_DESC_UBO])
            va = tu5_ubo_address(cmd, shader, &map->ubo[i]);

         tu_cs_emit_qw(cs, va);
      }
   }
}

/* A border colour as Adreno 5xx takes it. The components are in the order
 * the format has them, which is that of the colour for all but the formats
 * with blue first: the colours of Vulkan being the same either way, these
 * are packed the one way.
 *
 * TODO: custom border colours, which would be for a format
 */
struct PACKED tu5_bcolor_entry {
   uint32_t fp32[4];
   uint16_t ui16[4];
   int16_t si16[4];
   uint16_t fp16[4];
   uint16_t rgb565;
   uint16_t rgb5a1;
   uint16_t rgba4;
   uint8_t __pad0[2];
   uint8_t ui8[4];
   int8_t si8[4];
   uint32_t rgb10a2;
   uint32_t z24; /* also s8? */
   uint16_t srgb[4]; /* as fp16[], clamped */
   uint8_t __pad1[56];
};
static_assert(sizeof(struct tu5_bcolor_entry) == 128, "");

void
tu5_pack_border_color(void *entry, const VkClearColorValue *val, bool is_int)
{
   struct tu5_bcolor_entry *e = (struct tu5_bcolor_entry *)entry;

   memset(e, 0, sizeof(*e));

   for (unsigned c = 0; c < 4; c++) {
      if (is_int) {
         e->fp32[c] = val->uint32[c];
         e->fp16[c] = (uint16_t)MIN2(val->uint32[c], 0xffffu);
         continue;
      }

      float f = val->float32[c];
      float f_u = CLAMP(f, 0, 1);
      float f_s = CLAMP(f, -1, 1);

      e->fp32[c] = fui(f);
      e->fp16[c] = _mesa_float_to_half(f);
      e->srgb[c] = _mesa_float_to_half(f_u);
      e->ui16[c] = f_u * 0xffff;
      e->si16[c] = f_s * 0x7fff;
      e->ui8[c] = f_u * 0xff;
      e->si8[c] = f_s * 0x7f;
      if (c == 1)
         e->rgb565 |= (int)(f_u * 0x3f) << 5;
      else if (c < 3)
         e->rgb565 |= (int)(f_u * 0x1f) << (c ? 11 : 0);
      if (c == 3)
         e->rgb5a1 |= (f_u > 0.5f) ? 0x8000 : 0;
      else
         e->rgb5a1 |= (int)(f_u * 0x1f) << (c * 5);
      if (c == 3)
         e->rgb10a2 |= (int)(f_u * 0x3) << 30;
      else
         e->rgb10a2 |= (int)(f_u * 0x3ff) << (c * 10);
      e->rgba4 |= (int)(f_u * 0xf) << (c * 4);
      if (c == 0)
         e->z24 = f_u * 0xffffff;
   }
}

/* For the occlusion queries: where the count of the samples that passed is
 * to be copied by the event that follows.
 */
void
tu5_emit_sample_count_addr(struct tu_cs *cs, uint64_t iova)
{
   OUT_PKT4(cs, REG_A5XX_RB_SAMPLE_COUNT_CONTROL, 1);
   OUT_RING(cs, A5XX_RB_SAMPLE_COUNT_CONTROL_COPY);

   OUT_PKT4(cs, REG_A5XX_RB_SAMPLE_COUNT_ADDR_LO, 2);
   tu_cs_emit_qw(cs, iova);
}

/* For the timestamp queries: the low half of the 64 bit counter */
uint32_t
tu5_always_on_counter_reg(void)
{
   return REG_A5XX_RBBM_ALWAYSON_COUNTER_LO;
}

static void tu5_emit_mrt(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                         uint32_t layer);
static void tu5_emit_zs(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                        uint32_t layer);

/* Multiview: the views are drawn one after the other. The views of the
 * subpass, or the one view there is without multiview.
 */
uint32_t
tu5_views(const struct tu_cmd_buffer *cmd)
{
   uint32_t mask = cmd->state.subpass->multiview_mask;

   return mask ? mask : 1;
}

/* Before the draw of a view: the targets are its layer of the attachments,
 * and the shaders are given its index.
 */
void
tu5_bind_view(struct tu_cmd_buffer *cmd, struct tu_cs *cs, uint32_t view)
{
   if (!cmd->state.subpass->multiview_mask)
      return;

   if (cmd->state.attachments) {
      tu5_emit_zs(cmd, cs, view);
      tu5_emit_mrt(cmd, cs, view);
   } else {
      /* A secondary command buffer that does not know the framebuffer:
       * the primary one wrote where the views draw, see
       * tu5_write_view_bases().
       */
      const struct tu_subpass *subpass = cmd->state.subpass;
      uint64_t bases = global_iova_arr(cmd, a5xx_view_bases, view * 10);

      tu_cs_emit_wfi(cs);
      for (unsigned i = 0; i < subpass->color_count; i++) {
         if (subpass->color_attachments[i].attachment == VK_ATTACHMENT_UNUSED)
            continue;
         OUT_PKT7(cs, CP_MEM_TO_REG, 3);
         OUT_RING(cs, CP_MEM_TO_REG_0_REG(REG_A5XX_RB_MRT_BASE_LO(i)) |
                         CP_MEM_TO_REG_0_CNT(2));
         tu_cs_emit_qw(cs, bases + i * sizeof(uint64_t));
      }

      uint32_t a = subpass->depth_stencil_attachment.attachment;

      if (a != VK_ATTACHMENT_UNUSED) {
         OUT_PKT7(cs, CP_MEM_TO_REG, 3);
         OUT_RING(cs, CP_MEM_TO_REG_0_REG(REG_A5XX_RB_DEPTH_BUFFER_BASE_LO) |
                         CP_MEM_TO_REG_0_CNT(2));
         tu_cs_emit_qw(cs, bases + 8 * sizeof(uint64_t));

         if (cmd->state.pass->attachments[a].format ==
             VK_FORMAT_D32_SFLOAT_S8_UINT) {
            OUT_PKT7(cs, CP_MEM_TO_REG, 3);
            OUT_RING(cs, CP_MEM_TO_REG_0_REG(REG_A5XX_RB_STENCIL_BASE_LO) |
                            CP_MEM_TO_REG_0_CNT(2));
            tu_cs_emit_qw(cs, bases + 9 * sizeof(uint64_t));
         }
      }
   }

   for (unsigned stage = MESA_SHADER_VERTEX; stage <= MESA_SHADER_FRAGMENT;
        stage += MESA_SHADER_FRAGMENT - MESA_SHADER_VERTEX) {
      const struct tu_shader *shader = cmd->state.shaders[stage];
      uint32_t loc = shader->const_state.a5xx.view_loc;
      const uint32_t consts[4] = { view };

      if (loc != ~0u && loc < shader->variant->constlen * 4)
         tu5_emit_const(cs, shader->variant, loc, 4, consts);
   }
}

/* vkCmdDrawIndirect(), vkCmdDrawIndexedIndirect(): after the state of the
 * draw, a draw for each of the structures of the buffer. What the vertex
 * shader reads of them as constants is copied to where these are loaded from.
 */
void
tu5_draw_indirect(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                  uint32_t initiator, bool indexed, uint64_t iova,
                  uint32_t count, uint32_t stride)
{
   const struct ir3_shader_variant *vs =
      cmd->state.shaders[MESA_SHADER_VERTEX]->variant;
   uint32_t param = ir3_const_state(vs)->offsets.driver_param;
   uint64_t tmp = global_iova_arr(cmd, cs_indirect_xyz, 0);

   for (uint32_t i = 0; i < count; i++, iova += stride) {
      if (param < vs->constlen) {
         /* the first vertex, then the first instance */
         uint64_t first = iova + (indexed ? 3 : 2) * sizeof(uint32_t);
         uint32_t id[4] = {};

         id[IR3_DP_VS(draw_id)] = i;

         tu_cs_emit_wfi(cs);
         OUT_PKT7(cs, CP_MEM_WRITE, 2 + 4);
         tu_cs_emit_qw(cs, tmp);
         tu_cs_emit_array(cs, id, 4);

         OUT_PKT7(cs, CP_MEM_TO_MEM, 5);
         OUT_RING(cs, 0x00000000);
         tu_cs_emit_qw(cs, tmp + IR3_DP_VS(vtxid_base) * sizeof(uint32_t));
         tu_cs_emit_qw(cs, first);

         OUT_PKT7(cs, CP_MEM_TO_MEM, 5);
         OUT_RING(cs, 0x00000000);
         tu_cs_emit_qw(cs, tmp + IR3_DP_VS(instid_base) * sizeof(uint32_t));
         tu_cs_emit_qw(cs, first + sizeof(uint32_t));
         tu_cs_emit_wfi(cs);

         OUT_PKT7(cs, CP_LOAD_STATE4, 3);
         OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(param) |
                         CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) |
                         CP_LOAD_STATE4_0_STATE_BLOCK(SB4_VS_SHADER) |
                         CP_LOAD_STATE4_0_NUM_UNIT(1));
         OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                         (uint32_t)tmp);
         OUT_RING(cs, (uint32_t)(tmp >> 32));
      }

      if (indexed) {
         OUT_PKT7(cs, CP_DRAW_INDX_INDIRECT, 6);
         OUT_RING(cs, initiator);
         tu_cs_emit_qw(cs, cmd->state.index_va);
         OUT_RING(cs, A5XX_CP_DRAW_INDX_INDIRECT_3_MAX_INDICES(
                         cmd->state.max_index_count));
         tu_cs_emit_qw(cs, iova);
      } else {
         OUT_PKT7(cs, CP_DRAW_INDIRECT, 3);
         OUT_RING(cs, initiator);
         tu_cs_emit_qw(cs, iova);
      }
   }
}

/* vkCmdDispatch(): the compute shader and all its state, then the grid */
void
tu5_dispatch(struct tu_cmd_buffer *cmd, const uint32_t *blocks,
             const uint32_t *offsets, uint64_t indirect_iova)
{
   struct tu_cs *cs = &cmd->cs;
   const struct tu_shader *shader = cmd->state.shaders[MESA_SHADER_COMPUTE];
   const struct ir3_shader_variant *v = shader->variant;
   const struct ir3_const_state *const_state = ir3_const_state(v);
   const struct ir3_info *i = &v->info;
   struct tu5_shader so = tu5_shader_from(shader);
   enum a3xx_threadsize thrsz = i->double_threadsize ? FOUR_QUADS : TWO_QUADS;
   unsigned instrlen = v->instrlen;

   /* A shader of more than 32 * 16 instructions is not loaded ahead */
   if (instrlen > 32)
      instrlen = 0;

   OUT_PKT4(cs, REG_A5XX_SP_SP_CNTL, 1);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_HLSQ_CONTROL_0_REG, 1);
   OUT_RING(cs, A5XX_HLSQ_CONTROL_0_REG_FSTHREADSIZE(TWO_QUADS) |
                   A5XX_HLSQ_CONTROL_0_REG_CSTHREADSIZE(thrsz) |
                   0x00000880 /* XXX */);

   OUT_PKT4(cs, REG_A5XX_SP_CS_CTRL_REG0, 1);
   OUT_RING(cs, A5XX_SP_CS_CTRL_REG0_THREADSIZE(thrsz) |
                   A5XX_SP_CS_CTRL_REG0_HALFREGFOOTPRINT(i->max_half_reg + 1) |
                   A5XX_SP_CS_CTRL_REG0_FULLREGFOOTPRINT(i->max_reg + 1) |
                   A5XX_SP_CS_CTRL_REG0_BRANCHSTACK(
                      ir3_shader_branchstack_hw(v)) |
                   COND(instrlen != 0, A5XX_SP_CS_CTRL_REG0_BUFFER) |
                   0x2 /* XXX */);

   OUT_PKT4(cs, REG_A5XX_HLSQ_CS_CONFIG, 1);
   OUT_RING(cs, A5XX_HLSQ_CS_CONFIG_CONSTOBJECTOFFSET(0) |
                   A5XX_HLSQ_CS_CONFIG_SHADEROBJOFFSET(0) |
                   A5XX_HLSQ_CS_CONFIG_ENABLED);

   OUT_PKT4(cs, REG_A5XX_HLSQ_CS_CNTL, 1);
   OUT_RING(cs, A5XX_HLSQ_CS_CNTL_INSTRLEN(instrlen) |
                   COND(v->has_ssbo, A5XX_HLSQ_CS_CNTL_SSBO_ENABLE));

   OUT_PKT4(cs, REG_A5XX_SP_CS_CONFIG, 1);
   OUT_RING(cs, A5XX_SP_CS_CONFIG_CONSTOBJECTOFFSET(0) |
                   A5XX_SP_CS_CONFIG_SHADEROBJOFFSET(0) |
                   A5XX_SP_CS_CONFIG_ENABLED);

   assert(v->constlen % 4 == 0);
   OUT_PKT4(cs, REG_A5XX_HLSQ_CS_CONSTLEN, 2);
   OUT_RING(cs, v->constlen / 4); /* HLSQ_CS_CONSTLEN */
   OUT_RING(cs, instrlen);        /* HLSQ_CS_INSTRLEN */

   tu5_emit_shader_obj(cs, &so, REG_A5XX_SP_CS_OBJ_START_LO);

   OUT_PKT4(cs, REG_A5XX_HLSQ_UPDATE_CNTL, 1);
   OUT_RING(cs, 0x1f00000);

   OUT_PKT4(cs, REG_A5XX_HLSQ_CS_CNTL_0, 2);
   OUT_RING(cs, A5XX_HLSQ_CS_CNTL_0_WGIDCONSTID(ir3_find_sysval_regid(
                   v, SYSTEM_VALUE_WORKGROUP_ID)) |
                   A5XX_HLSQ_CS_CNTL_0_UNK0(regid(63, 0)) |
                   A5XX_HLSQ_CS_CNTL_0_UNK1(regid(63, 0)) |
                   A5XX_HLSQ_CS_CNTL_0_LOCALIDREGID(ir3_find_sysval_regid(
                      v, SYSTEM_VALUE_LOCAL_INVOCATION_ID)));
   OUT_RING(cs, 0x1); /* HLSQ_CS_CNTL_1 */

   if (instrlen > 0)
      tu5_emit_shader(cs, v, so.iova);

   OUT_PKT4(cs, REG_A5XX_TPL1_TP_BORDER_COLOR_BASE_ADDR_LO, 2);
   tu_cs_emit_qw(cs, global_iova(cmd, bcolor_builtin));

   tu5_emit_immediates(cs, v);
   tu5_emit_user_consts(cmd, cs, shader);
   tu5_emit_textures(cmd, cs, shader);
   tu5_emit_ssbos(cmd, cs, shader);
   tu5_emit_images(cmd, cs, shader);

   /* The parameters of the dispatch the shader reads as constants */
   uint32_t offset = const_state->offsets.driver_param;

   if (offset < v->constlen) {
      uint32_t params[dword_sizeof(struct ir3_driver_params_cs)] = {};
      uint32_t dwords = MIN2(align(const_state->num_driver_params, 4),
                             (v->constlen - offset) * 4);

      params[IR3_DP_CS(num_work_groups_x)] = blocks[0];
      params[IR3_DP_CS(num_work_groups_y)] = blocks[1];
      params[IR3_DP_CS(num_work_groups_z)] = blocks[2];
      params[IR3_DP_CS(work_dim)] = 3;
      params[IR3_DP_CS(base_group_x)] = offsets[0];
      params[IR3_DP_CS(base_group_y)] = offsets[1];
      params[IR3_DP_CS(base_group_z)] = offsets[2];
      params[IR3_DP_CS(subgroup_size)] = i->subgroup_size;
      params[IR3_DP_CS(local_group_size_x)] = v->local_size[0];
      params[IR3_DP_CS(local_group_size_y)] = v->local_size[1];
      params[IR3_DP_CS(local_group_size_z)] = v->local_size[2];
      params[IR3_DP_CS(subgroup_id_shift)] = util_logbase2(i->subgroup_size);

      dwords = MIN2(dwords, (uint32_t)ARRAY_SIZE(params));
      if (dwords)
         tu5_emit_const(cs, v, offset * 4, dwords, params);

      if (indirect_iova && dwords) {
         /* The numbers of the groups are in the buffer, at an address that
          * need not be a multiple of 16: copy them to where it is.
          */
         uint64_t tmp = global_iova_arr(cmd, cs_indirect_xyz, 0);

         tu_cs_emit_wfi(cs);
         for (uint32_t c = 0; c < 3; c++) {
            OUT_PKT7(cs, CP_MEM_TO_MEM, 5);
            OUT_RING(cs, 0x00000000);
            tu_cs_emit_qw(cs, tmp + c * sizeof(uint32_t));
            tu_cs_emit_qw(cs, indirect_iova + c * sizeof(uint32_t));
         }
         OUT_PKT7(cs, CP_MEM_WRITE, 3);
         tu_cs_emit_qw(cs, tmp + 3 * sizeof(uint32_t));
         OUT_RING(cs, 3); /* work_dim */
         tu_cs_emit_wfi(cs);

         OUT_PKT7(cs, CP_LOAD_STATE4, 3);
         OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(offset) |
                         CP_LOAD_STATE4_0_STATE_SRC(SS4_INDIRECT) |
                         CP_LOAD_STATE4_0_STATE_BLOCK(SB4_CS_SHADER) |
                         CP_LOAD_STATE4_0_NUM_UNIT(1));
         OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                         (uint32_t)tmp);
         OUT_RING(cs, (uint32_t)(tmp >> 32));
      }
   }

   const uint16_t *local_size = v->local_size;

   OUT_PKT4(cs, REG_A5XX_HLSQ_CS_NDRANGE_0, 7);
   OUT_RING(cs, A5XX_HLSQ_CS_NDRANGE_0_KERNELDIM(3) |
                   A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEX(local_size[0] - 1) |
                   A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEY(local_size[1] - 1) |
                   A5XX_HLSQ_CS_NDRANGE_0_LOCALSIZEZ(local_size[2] - 1));
   OUT_RING(cs, A5XX_HLSQ_CS_NDRANGE_1_GLOBALSIZE_X(local_size[0] * blocks[0]));
   OUT_RING(cs, 0); /* HLSQ_CS_NDRANGE_2_GLOBALOFF_X */
   OUT_RING(cs, A5XX_HLSQ_CS_NDRANGE_3_GLOBALSIZE_Y(local_size[1] * blocks[1]));
   OUT_RING(cs, 0); /* HLSQ_CS_NDRANGE_4_GLOBALOFF_Y */
   OUT_RING(cs, A5XX_HLSQ_CS_NDRANGE_5_GLOBALSIZE_Z(local_size[2] * blocks[2]));
   OUT_RING(cs, 0); /* HLSQ_CS_NDRANGE_6_GLOBALOFF_Z */

   OUT_PKT4(cs, REG_A5XX_HLSQ_CS_KERNEL_GROUP_X, 3);
   OUT_RING(cs, 1); /* HLSQ_CS_KERNEL_GROUP_X */
   OUT_RING(cs, 1); /* HLSQ_CS_KERNEL_GROUP_Y */
   OUT_RING(cs, 1); /* HLSQ_CS_KERNEL_GROUP_Z */

   if (indirect_iova) {
      OUT_PKT7(cs, CP_EXEC_CS_INDIRECT, 4);
      OUT_RING(cs, 0x00000000);
      tu_cs_emit_qw(cs, indirect_iova);
      OUT_RING(cs, A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEX(local_size[0] - 1) |
                      A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEY(local_size[1] - 1) |
                      A5XX_CP_EXEC_CS_INDIRECT_3_LOCALSIZEZ(local_size[2] - 1));
   } else {
      OUT_PKT7(cs, CP_EXEC_CS, 4);
      OUT_RING(cs, 0x00000000);
      OUT_RING(cs, CP_EXEC_CS_1_NGROUPS_X(blocks[0]));
      OUT_RING(cs, CP_EXEC_CS_2_NGROUPS_Y(blocks[1]));
      OUT_RING(cs, CP_EXEC_CS_3_NGROUPS_Z(blocks[2]));
   }

   /* What was written is to be seen by what comes after */
   tu5_cache_flush(cs);
}

/* The descriptor of an image view, for write_image_descriptor() */
void
tu5_image_view_descriptor(uint32_t *descriptor, const struct tu_image *image,
                          unsigned plane, enum pipe_format format,
                          const VkImageSubresourceRange *range,
                          VkImageViewType type, const unsigned char *swiz)
{
   const struct fdl_layout *layout = &image->layout[plane];
   uint32_t level = range->baseMipLevel;
   uint32_t levels = vk_image_subresource_level_count(&image->vk, range);
   uint32_t layers = vk_image_subresource_layer_count(&image->vk, range);
   uint64_t iova =
      image->iova + fdl_surface_offset(layout, level, range->baseArrayLayer);
   enum a5xx_tex_type tex_type;
   uint32_t depth = layers;

   switch (type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
      tex_type = A5XX_TEX_1D;
      break;
   case VK_IMAGE_VIEW_TYPE_CUBE:
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      tex_type = A5XX_TEX_CUBE;
      depth = layers / 6;
      break;
   case VK_IMAGE_VIEW_TYPE_3D:
      tex_type = A5XX_TEX_3D;
      depth = u_minify(layout->depth0, level);
      break;
   default:
      tex_type = A5XX_TEX_2D;
      break;
   }

   memset(descriptor, 0, A6XX_TEX_CONST_DWORDS * sizeof(uint32_t));

   uint32_t tex_swiz;

   if (util_format_is_depth_or_stencil(format)) {
      /* Depth and stencil read as (value, 0, 0, 1) */
      unsigned char zs_swiz[4];

      for (unsigned c = 0; c < 4; c++) {
         switch (swiz[c]) {
         case PIPE_SWIZZLE_X:
            zs_swiz[c] = PIPE_SWIZZLE_X;
            break;
         case PIPE_SWIZZLE_W:
         case PIPE_SWIZZLE_1:
            zs_swiz[c] = PIPE_SWIZZLE_1;
            break;
         default:
            zs_swiz[c] = PIPE_SWIZZLE_0;
            break;
         }
      }
      tex_swiz = fd5_tex_swiz(PIPE_FORMAT_R8G8B8A8_UNORM, zs_swiz[0],
                              zs_swiz[1], zs_swiz[2], zs_swiz[3]);
   } else {
      tex_swiz = fd5_tex_swiz(format, swiz[0], swiz[1], swiz[2], swiz[3]);
   }

   descriptor[0] =
      A5XX_TEX_CONST_0_TILE_MODE((enum a5xx_tile_mode)fdl_tile_mode(layout, level)) |
      A5XX_TEX_CONST_0_FMT(fd5_pipe2tex(format)) |
      A5XX_TEX_CONST_0_SAMPLES(tu_msaa_samples(layout->nr_samples)) |
      A5XX_TEX_CONST_0_MIPLVLS(levels - 1) |
      tex_swiz |
      COND(format == PIPE_FORMAT_X24S8_UINT, A5XX_TEX_CONST_0_SWAP(XYZW)) |
      COND(util_format_is_srgb(format), A5XX_TEX_CONST_0_SRGB);
   uint32_t width = u_minify(layout->width0, level);
   uint32_t height = u_minify(layout->height0, level);

   /* The blocks of a compressed image are the texels of a view of another
    * format.
    */
   if (!util_format_is_compressed(format)) {
      width = util_format_get_nblocksx(layout->format, width);
      height = util_format_get_nblocksy(layout->format, height);
   }

   descriptor[1] = A5XX_TEX_CONST_1_WIDTH(width) |
                   A5XX_TEX_CONST_1_HEIGHT(height);
   descriptor[2] = A5XX_TEX_CONST_2_PITCHALIGN(layout->pitchalign - 6) |
                   A5XX_TEX_CONST_2_PITCH(fdl_pitch(layout, level)) |
                   A5XX_TEX_CONST_2_TYPE(tex_type);
   if (type == VK_IMAGE_VIEW_TYPE_3D) {
      /* The size the slices stop being made smaller at, from level 2 on,
       * see fdl5_layout(); none if they never stop, as with two levels.
       */
      const unsigned last = layout->mip_levels - 1;
      const bool stops = last >= 2 && layout->slices[last - 1].size0 <= 0xf000;

      descriptor[3] =
         A5XX_TEX_CONST_3_MIN_LAYERSZ(stops ? layout->slices[last].size0 : 0) |
         A5XX_TEX_CONST_3_ARRAY_PITCH(layout->slices[level].size0);
   } else {
      descriptor[3] = A5XX_TEX_CONST_3_ARRAY_PITCH(layout->layer_size);
   }
   if (layout->tile_all)
      descriptor[3] |= A5XX_TEX_CONST_3_TILE_ALL;
   descriptor[4] = (uint32_t)iova;
   descriptor[5] = (uint32_t)(iova >> 32) | A5XX_TEX_CONST_5_DEPTH(depth);
}

/* The descriptor of a buffer view, both as a texture and as a storage image,
 * see tu5_image_view_storage_descriptor(). The number of texels is in the
 * width, and what does not fit there in the height.
 */
void
tu5_buffer_view_descriptor(uint32_t *descriptor, enum pipe_format format,
                           uint64_t iova, uint32_t range)
{
   uint32_t cpp = util_format_get_blocksize(format);
   uint32_t elements = range / cpp;

   memset(descriptor, 0, A6XX_TEX_CONST_DWORDS * sizeof(uint32_t));

   descriptor[0] = A5XX_TEX_CONST_0_FMT(fd5_pipe2tex(format)) |
                   fd5_tex_swiz(format, PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y,
                                PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W) |
                   COND(util_format_is_srgb(format), A5XX_TEX_CONST_0_SRGB);
   descriptor[1] = A5XX_TEX_CONST_1_WIDTH(elements & 0x7fff) |
                   A5XX_TEX_CONST_1_HEIGHT(elements >> 15);
   descriptor[2] = A5XX_TEX_CONST_2_BUFFER |
                   A5XX_TEX_CONST_2_TYPE(A5XX_TEX_BUFFER);
   descriptor[4] = (uint32_t)iova;
   descriptor[5] = (uint32_t)(iova >> 32);

   descriptor[6] = A5XX_SSBO_1_0_FMT(fd5_pipe2tex(format)) |
                   A5XX_SSBO_1_0_WIDTH(elements & 0x7fff);
   descriptor[7] = A5XX_SSBO_1_1_HEIGHT(elements >> 15);
   descriptor[8] = cpp;
   /* for the size, which the shader divides by the bytes of a texel */
   descriptor[9] = ffs(cpp) - 1;
}

/* The descriptor of an image view as a storage image: that of a texture,
 * then what tu5_emit_images() takes.
 */
void
tu5_image_view_storage_descriptor(uint32_t *descriptor,
                                  const struct tu_image *image, unsigned plane,
                                  enum pipe_format format,
                                  const VkImageSubresourceRange *range,
                                  VkImageViewType type)
{
   static const unsigned char swiz[4] = { PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y,
                                          PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W };
   const struct fdl_layout *layout = &image->layout[plane];
   uint32_t level = range->baseMipLevel;
   uint32_t layers = vk_image_subresource_layer_count(&image->vk, range);
   uint32_t width = u_minify(layout->width0, level);
   uint32_t height = u_minify(layout->height0, level);
   uint32_t depth = layers;
   uint32_t layer_size = layout->layer_size;

   if (!util_format_is_compressed(format)) {
      width = util_format_get_nblocksx(layout->format, width);
      height = util_format_get_nblocksy(layout->format, height);
   }

   /* The faces of a cube are the layers of an array */
   if (type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
      type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

   tu5_image_view_descriptor(descriptor, image, plane, format, range, type,
                             swiz);

   if (type == VK_IMAGE_VIEW_TYPE_3D) {
      depth = u_minify(layout->depth0, level);
      layer_size = layout->slices[level].size0;
   }

   descriptor[6] = A5XX_SSBO_1_0_FMT(fd5_pipe2tex(format)) |
                   A5XX_SSBO_1_0_WIDTH(width);
   descriptor[7] = A5XX_SSBO_1_1_HEIGHT(height) | A5XX_SSBO_1_1_DEPTH(depth);
   descriptor[8] = util_format_get_blocksize(format);
   descriptor[9] = fdl_pitch(layout, level);
   descriptor[10] = layer_size;

   /* The second coordinate of the 1D arrays is the layer: these are written
    * as images of that many rows.
    */
   if (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY) {
      descriptor[7] = A5XX_SSBO_1_1_HEIGHT(layers) | A5XX_SSBO_1_1_DEPTH(1);
      descriptor[9] = layer_size;
   }
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
   struct tu5_shader vs_shader, fs_shader;
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
   if (!VALIDREG(vertex_regid))
      vertex_regid = ir3_find_sysval_regid(s[VS].v, SYSTEM_VALUE_VERTEX_ID);
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

   tu5_emit_shader_obj(ring, &emit->vs_shader, REG_A5XX_SP_VS_OBJ_START_LO);

   if (s[VS].instrlen)
      tu5_emit_shader(ring, s[VS].v, emit->vs_shader.iova);

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
      tu5_emit_shader_obj(ring, &emit->fs_shader, REG_A5XX_SP_FS_OBJ_START_LO);
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
         COND(s[FS].v->need_pixlod, A5XX_SP_FS_CTRL_REG0_PIXLODENABLE) |
         /* found by trying: three bits below PIXLODENABLE, as LODPIXMASK is
          * on Adreno 6xx, the bit that has the quads run whole for the fine
          * derivatives
          */
         COND(s[FS].v->need_full_quad, 0x00020000));

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

         /* What the vertex shader does not write of these reads as zero */
         if ((s[FS].v->inputs[j].slot == VARYING_SLOT_LAYER ||
              s[FS].v->inputs[j].slot == VARYING_SLOT_VIEWPORT) &&
             ir3_find_output(s[VS].v, (gl_varying_slot)s[FS].v->inputs[j].slot) < 0) {
            if (compmask & 0x1)
               vinterp[inloc / 16] |= 0b10 << ((inloc % 16) * 2);
            continue;
         }

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
         tu5_emit_shader(ring, s[FS].v, emit->fs_shader.iova);

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

/* What is drawn into or sampled from: a level and a layer of an image, or
 * a part of a buffer taken as an image.
 */
struct tu5_surf {
   enum pipe_format format;
   uint64_t iova;
   uint32_t width, height;
   uint32_t pitch;      /* bytes */
   uint32_t layer_size; /* bytes */
   enum a5xx_tile_mode tile_mode;
   uint32_t samples;
   /* As a texture: the component each of its four reads, if not its own */
   const unsigned char *swiz;
   /* As a texture: the slices of a level of a 3D image, if not zero, and the
    * size of those of its last level
    */
   uint32_t depth;
   uint32_t min_layer_size;
};

/* A level of a 3D image as a texture */
static struct tu5_surf
tu5_image_surf_3d(const struct tu_image *image, enum pipe_format format,
                  uint32_t level)
{
   const struct fdl_layout *layout = &image->layout[0];
   struct tu5_surf surf = {
      .format = format,
      .iova = image->iova + fdl_surface_offset(layout, level, 0),
      .width = u_minify(layout->width0, level),
      .height = u_minify(layout->height0, level),
      .pitch = fdl_pitch(layout, level),
      .layer_size = layout->slices[level].size0,
      .tile_mode = (enum a5xx_tile_mode)fdl_tile_mode(layout, level),
      .samples = 1,
      .depth = u_minify(layout->depth0, level),
      .min_layer_size = layout->slices[layout->mip_levels - 1].size0,
   };

   return surf;
}

static struct tu5_surf
tu5_image_surf(const struct tu_image *image, unsigned plane,
               enum pipe_format format, uint32_t level, uint32_t layer)
{
   const struct fdl_layout *layout = &image->layout[plane];
   uint32_t width = u_minify(layout->width0, level);
   uint32_t height = u_minify(layout->height0, level);

   /* The blocks of a compressed image are the texels of what it is copied
    * as, see tu5_copy_format().
    */
   if (!util_format_is_compressed(format)) {
      width = util_format_get_nblocksx(layout->format, width);
      height = util_format_get_nblocksy(layout->format, height);
   }

   return (struct tu5_surf) {
      .format = format,
      .iova = image->iova + fdl_surface_offset(layout, level, layer),
      .width = width,
      .height = height,
      .pitch = fdl_pitch(layout, level),
      .layer_size = fdl_layer_stride(layout, level),
      .tile_mode = (enum a5xx_tile_mode)fdl_tile_mode(layout, level),
      .samples = layout->nr_samples,
   };
}

/* The address and the pitch must be multiples of 64 bytes */
static struct tu5_surf
tu5_buffer_surf(enum pipe_format format, uint64_t iova, uint32_t width,
                uint32_t height, uint32_t pitch)
{
   assert(!(iova & 63) && !(pitch & 63));

   return (struct tu5_surf) {
      .format = format,
      .iova = iova,
      .width = width,
      .height = height,
      .pitch = pitch,
      .layer_size = align(pitch * height, 64),
      .tile_mode = TILE5_LINEAR,
      .samples = 1,
   };
}

/* One colour target, or none */
static void
tu5_emit_mrt_surf(struct tu_cs *cs, unsigned i, const struct tu5_surf *surf)
{
   enum a5xx_color_fmt format = (enum a5xx_color_fmt)0;
   enum a5xx_tile_mode tile_mode = TILE5_LINEAR;
   enum a3xx_color_swap swap = WZYX;
   bool srgb = false, sint = false, uint = false;
   uint32_t stride = 0;
   uint32_t size = 0;
   uint64_t base = 0;

   if (surf) {
      format = fd5_pipe2color(surf->format);
      swap = fd5_pipe2swap(surf->format);
      srgb = util_format_is_srgb(surf->format);
      sint = util_format_is_pure_sint(surf->format);
      uint = util_format_is_pure_uint(surf->format);

      stride = surf->pitch;
      size = surf->layer_size;
      tile_mode = surf->tile_mode;
      base = surf->iova;
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

/* A surface as the one 2D texture of the blits, sampled by texel */
static void
tu5_emit_blit_src(struct tu_cs *cs, const struct tu5_surf *surf, bool linear)
{
   enum a5xx_tex_filter filter = linear ? A5XX_TEX_LINEAR : A5XX_TEX_NEAREST;

   OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 4);
   OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(0) |
                   CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                   CP_LOAD_STATE4_0_STATE_BLOCK(SB4_FS_TEX) |
                   CP_LOAD_STATE4_0_NUM_UNIT(1));
   OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_SHADER) |
                   CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
   OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   OUT_RING(cs, A5XX_TEX_SAMP_0_XY_MAG(filter) | A5XX_TEX_SAMP_0_XY_MIN(filter) |
                   A5XX_TEX_SAMP_0_WRAP_S(A5XX_TEX_CLAMP_TO_EDGE) |
                   A5XX_TEX_SAMP_0_WRAP_T(A5XX_TEX_CLAMP_TO_EDGE) |
                   A5XX_TEX_SAMP_0_WRAP_R(A5XX_TEX_CLAMP_TO_EDGE));
   OUT_RING(cs, A5XX_TEX_SAMP_1_UNNORM_COORDS |
                   A5XX_TEX_SAMP_1_CUBEMAPSEAMLESSFILTOFF |
                   A5XX_TEX_SAMP_1_MIN_LOD(0.0f) | A5XX_TEX_SAMP_1_MAX_LOD(0.125f));
   OUT_RING(cs, 0x00000000);
   OUT_RING(cs, 0x00000000);

   OUT_PKT7(cs, CP_LOAD_STATE4, 3 + 12);
   OUT_RING(cs, CP_LOAD_STATE4_0_DST_OFF(0) |
                   CP_LOAD_STATE4_0_STATE_SRC(SS4_DIRECT) |
                   CP_LOAD_STATE4_0_STATE_BLOCK(SB4_FS_TEX) |
                   CP_LOAD_STATE4_0_NUM_UNIT(1));
   OUT_RING(cs, CP_LOAD_STATE4_1_STATE_TYPE(ST4_CONSTANTS) |
                   CP_LOAD_STATE4_1_EXT_SRC_ADDR(0));
   OUT_RING(cs, CP_LOAD_STATE4_2_EXT_SRC_ADDR_HI(0));
   OUT_RING(cs, A5XX_TEX_CONST_0_TILE_MODE(surf->tile_mode) |
                   A5XX_TEX_CONST_0_FMT(fd5_pipe2tex(surf->format)) |
                   A5XX_TEX_CONST_0_SAMPLES(tu_msaa_samples(surf->samples)) |
                   (surf->swiz
                       ? fd5_tex_swiz(surf->format, surf->swiz[0], surf->swiz[1],
                                      surf->swiz[2], surf->swiz[3])
                       : fd5_tex_swiz(surf->format, PIPE_SWIZZLE_X,
                                      PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z,
                                      PIPE_SWIZZLE_W)) |
                   COND(util_format_is_srgb(surf->format),
                        A5XX_TEX_CONST_0_SRGB));
   OUT_RING(cs, A5XX_TEX_CONST_1_WIDTH(surf->width) |
                   A5XX_TEX_CONST_1_HEIGHT(surf->height));
   OUT_RING(cs, A5XX_TEX_CONST_2_PITCHALIGN(0) |
                   A5XX_TEX_CONST_2_PITCH(surf->pitch) |
                   A5XX_TEX_CONST_2_TYPE(surf->depth ? A5XX_TEX_3D
                                                     : A5XX_TEX_2D));
   /* a level that is tiled although small is so for being of an image that
    * is tiled at every level
    */
   OUT_RING(cs, A5XX_TEX_CONST_3_ARRAY_PITCH(surf->layer_size) |
                   COND(surf->depth,
                        A5XX_TEX_CONST_3_MIN_LAYERSZ(surf->min_layer_size)) |
                   COND(surf->tile_mode != TILE5_LINEAR,
                        A5XX_TEX_CONST_3_TILE_ALL));
   OUT_RING(cs, (uint32_t)surf->iova);
   OUT_RING(cs, (uint32_t)(surf->iova >> 32) |
                   A5XX_TEX_CONST_5_DEPTH(surf->depth ? surf->depth : 1));
   for (unsigned i = 6; i < 12; i++)
      OUT_RING(cs, 0x00000000);
}

/* The colour targets of the subpass */
static void
tu5_emit_mrt(struct tu_cmd_buffer *cmd, struct tu_cs *cs, uint32_t layer)
{
   const struct tu_subpass *subpass = cmd->state.subpass;
   uint32_t components = 0;

   for (unsigned i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
      if (i < subpass->color_count &&
          subpass->color_attachments[i].attachment != VK_ATTACHMENT_UNUSED) {
         const struct tu_image_view *iview =
            cmd->state.attachments[subpass->color_attachments[i].attachment];

         struct tu5_surf surf =
            tu5_image_surf(iview->image, 0,
                           vk_format_to_pipe_format(iview->vk.format),
                           iview->vk.base_mip_level,
                           iview->vk.base_array_layer + layer);

         tu5_emit_mrt_surf(cs, i, &surf);
         components |= 0xf << (i * 4);
      } else {
         tu5_emit_mrt_surf(cs, i, NULL);
      }
   }

   OUT_PKT4(cs, REG_A5XX_RB_RENDER_COMPONENTS, 1);
   OUT_RING(cs, components);
}

/* The depth and stencil buffer: a level and a layer of an image, or none */
static void
tu5_emit_zs_surface(struct tu_cs *cs, const struct tu_image *image,
                    uint32_t level, uint32_t layer)
{
   if (image) {
      const struct fdl_layout *layout = &image->layout[0];
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
tu5_emit_zs(struct tu_cmd_buffer *cmd, struct tu_cs *cs, uint32_t layer)
{
   uint32_t a = cmd->state.subpass->depth_stencil_attachment.attachment;

   if (a != VK_ATTACHMENT_UNUSED) {
      const struct tu_image_view *iview = cmd->state.attachments[a];

      tu5_emit_zs_surface(cs, iview->image, iview->vk.base_mip_level,
                          iview->vk.base_array_layer + layer);
   } else {
      tu5_emit_zs_surface(cs, NULL, 0, 0);
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

/* A rectangle drawn with the internal shaders, for the clears and the blits.
 * a5xx gets these as a triangle strip of four vertices through the usual
 * viewport transform: see build_rect5_vs_shader().
 */
struct tu5_rect_draw {
   enum global_shader fs;

   /* Size of the target, for the viewport */
   uint32_t width, height;
   /* The rectangle on the target, which is the scissor too */
   VkRect2D rect;

   /* The colour target written, or -1 */
   int rt;
   /* Depth written from z, stencil from stencil_ref */
   bool depth, stencil;
   float z;
   uint32_t stencil_ref;

   const uint32_t *fs_consts;
   uint32_t fs_consts_dwords;

   /* The texture of a blit, and whether it is filtered */
   const struct tu5_surf *src;
   bool src_linear;
   /* Texture coordinates of the corners of the rectangle, and a third one */
   float s0, t0, s1, t1, r;

   /* The components not to write, of the colour target */
   uint32_t keep_mask;

   /* The samples to write, all of them if none */
   uint32_t sample_mask;
};

static void
tu5_draw_rect(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
              const struct tu5_rect_draw *d)
{
   struct tu5_program prog = {};

   prog.vs_shader = tu5_global_shader(cmd->device, GLOBAL_SH_VS_RECT5);
   prog.fs_shader = tu5_global_shader(cmd->device, d->fs);
   prog.vs = prog.vs_shader.v;
   prog.fs = prog.fs_shader.v;

   /* depth and stencil */
   OUT_PKT4(cs, REG_A5XX_RB_ALPHA_CONTROL, 1);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_RB_STENCIL_CONTROL, 1);
   OUT_RING(cs, COND(d->stencil,
                     A5XX_RB_STENCIL_CONTROL_STENCIL_ENABLE |
                        A5XX_RB_STENCIL_CONTROL_STENCIL_ENABLE_BF |
                        A5XX_RB_STENCIL_CONTROL_FUNC(FUNC_ALWAYS) |
                        A5XX_RB_STENCIL_CONTROL_FAIL(STENCIL_REPLACE) |
                        A5XX_RB_STENCIL_CONTROL_ZPASS(STENCIL_REPLACE) |
                        A5XX_RB_STENCIL_CONTROL_ZFAIL(STENCIL_REPLACE) |
                        A5XX_RB_STENCIL_CONTROL_FUNC_BF(FUNC_ALWAYS) |
                        A5XX_RB_STENCIL_CONTROL_FAIL_BF(STENCIL_REPLACE) |
                        A5XX_RB_STENCIL_CONTROL_ZPASS_BF(STENCIL_REPLACE) |
                        A5XX_RB_STENCIL_CONTROL_ZFAIL_BF(STENCIL_REPLACE)));

   OUT_PKT4(cs, REG_A5XX_GRAS_LRZ_CNTL, 1);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_RB_STENCILREFMASK, 2);
   OUT_RING(cs, A5XX_RB_STENCILREFMASK_STENCILWRITEMASK(0xff) |
                   A5XX_RB_STENCILREFMASK_STENCILMASK(0xff) |
                   A5XX_RB_STENCILREFMASK_STENCILREF(d->stencil_ref));
   OUT_RING(cs, A5XX_RB_STENCILREFMASK_BF_STENCILWRITEMASK(0xff) |
                   A5XX_RB_STENCILREFMASK_BF_STENCILMASK(0xff) |
                   A5XX_RB_STENCILREFMASK_BF_STENCILREF(d->stencil_ref));

   OUT_PKT4(cs, REG_A5XX_RB_DEPTH_CNTL, 1);
   OUT_RING(cs, COND(d->depth, A5XX_RB_DEPTH_CNTL_ZFUNC(FUNC_ALWAYS) |
                                  A5XX_RB_DEPTH_CNTL_Z_TEST_ENABLE |
                                  A5XX_RB_DEPTH_CNTL_Z_WRITE_ENABLE));

   OUT_PKT4(cs, REG_A5XX_RB_DEPTH_PLANE_CNTL, 1);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_GRAS_SU_DEPTH_PLANE_CNTL, 1);
   OUT_RING(cs, 0x00000000);

   /* scissor and viewport */
   uint32_t minx = d->rect.offset.x, miny = d->rect.offset.y;
   uint32_t maxx = minx + d->rect.extent.width - 1;
   uint32_t maxy = miny + d->rect.extent.height - 1;

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0, 2);
   OUT_RING(cs, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(minx) |
                   A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(miny));
   OUT_RING(cs, A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_X(maxx) |
                   A5XX_GRAS_SC_SCREEN_SCISSOR_TL_0_Y(maxy));

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0, 2);
   OUT_RING(cs, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(minx) |
                   A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(miny));
   OUT_RING(cs, A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_X(maxx) |
                   A5XX_GRAS_SC_VIEWPORT_SCISSOR_TL_0_Y(maxy));

   tu_cs_emit_wfi(cs);
   OUT_PKT4(cs, REG_A5XX_GRAS_CL_VPORT_XOFFSET_0, 6);
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_XOFFSET_0(d->width / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_XSCALE_0(d->width / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_YOFFSET_0(d->height / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_YSCALE_0(d->height / 2.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_ZOFFSET_0(0.0f));
   OUT_RING(cs, A5XX_GRAS_CL_VPORT_ZSCALE_0(1.0f));

   tu5_program_emit(cs, &prog);

   /* rasterizer */
   OUT_PKT4(cs, REG_A5XX_GRAS_SU_CNTL, 1);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_GRAS_SU_POLY_OFFSET_SCALE, 3);
   OUT_RING(cs, 0x00000000);
   OUT_RING(cs, 0x00000000);
   OUT_RING(cs, 0x00000000);

   OUT_PKT4(cs, REG_A5XX_PC_RASTER_CNTL, 1);
   OUT_RING(cs, A5XX_PC_RASTER_CNTL_POLYMODE_FRONT_PTYPE(PC_DRAW_TRIANGLES) |
                   A5XX_PC_RASTER_CNTL_POLYMODE_BACK_PTYPE(PC_DRAW_TRIANGLES));

   OUT_PKT4(cs, REG_A5XX_GRAS_CL_CNTL, 1);
   OUT_RING(cs, A5XX_GRAS_CL_CNTL_ZERO_GB_SCALE_Z);

   OUT_PKT4(cs, REG_A5XX_PC_PRIMITIVE_CNTL, 1);
   OUT_RING(cs, A5XX_PC_PRIMITIVE_CNTL_STRIDE_IN_VPC(prog.max_loc));

   /* The colour target written, with the first output of the shader */
   for (unsigned i = 0; i < A5XX_MAX_RENDER_TARGETS; i++) {
      OUT_PKT4(cs, REG_A5XX_RB_MRT_CONTROL(i), 1);
      OUT_RING(cs, COND((int)i == d->rt,
                        A5XX_RB_MRT_CONTROL_ROP_CODE(ROP_COPY) |
                           A5XX_RB_MRT_CONTROL_COMPONENT_ENABLE(
                              0xf & ~d->keep_mask)));

      OUT_PKT4(cs, REG_A5XX_RB_MRT_BLEND_CONTROL(i), 1);
      OUT_RING(cs, 0x00000000);
   }

   if (d->rt > 0) {
      uint32_t regid = ir3_find_output_regid(prog.fs, FRAG_RESULT_DATA0);

      OUT_PKT4(cs, REG_A5XX_SP_FS_OUTPUT_REG(d->rt), 1);
      OUT_RING(cs, A5XX_SP_FS_OUTPUT_REG_REGID(regid) |
                      COND(regid & HALF_REG_ID,
                           A5XX_SP_FS_OUTPUT_REG_HALF_PRECISION));
   }

   OUT_PKT4(cs, REG_A5XX_SP_BLEND_CNTL, 1);
   OUT_RING(cs, A5XX_SP_BLEND_CNTL_UNK8);

   OUT_PKT4(cs, REG_A5XX_RB_BLEND_CNTL, 1);
   OUT_RING(cs, A5XX_RB_BLEND_CNTL_INDEPENDENT_BLEND |
                   A5XX_RB_BLEND_CNTL_SAMPLE_MASK(
                      d->sample_mask ? d->sample_mask : 0xffff));

   OUT_PKT4(cs, REG_A5XX_RB_FS_OUTPUT_CNTL, 1);
   OUT_RING(cs, A5XX_RB_FS_OUTPUT_CNTL_MRT(d->rt + 1));

   OUT_PKT4(cs, REG_A5XX_SP_FS_OUTPUT_CNTL, 1);
   OUT_RING(cs, A5XX_SP_FS_OUTPUT_CNTL_MRT(d->rt + 1) |
                   A5XX_SP_FS_OUTPUT_CNTL_DEPTH_REGID(regid(63, 0)) |
                   A5XX_SP_FS_OUTPUT_CNTL_SAMPLEMASK_REGID(regid(63, 0)));

   /* no vertex buffers: the vertices are made from the constants */
   OUT_PKT4(cs, REG_A5XX_VFD_CONTROL_0, 1);
   OUT_RING(cs, A5XX_VFD_CONTROL_0_VTXCNT(0));

   OUT_PKT4(cs, REG_A5XX_VFD_INDEX_OFFSET, 2);
   OUT_RING(cs, 0);
   OUT_RING(cs, 0);

   const float vs_consts[12] = {
      2.0f * d->rect.offset.x / d->width - 1.0f,
      2.0f * d->rect.offset.y / d->height - 1.0f,
      2.0f * (d->rect.offset.x + d->rect.extent.width) / d->width - 1.0f,
      2.0f * (d->rect.offset.y + d->rect.extent.height) / d->height - 1.0f,
      d->s0, d->t0, d->s1, d->t1,
      d->z, d->r, 0.0f, 0.0f,
   };
   tu5_emit_const(cs, prog.vs, 0, 12, (const uint32_t *)vs_consts);
   tu5_emit_immediates(cs, prog.vs);

   if (d->fs_consts_dwords)
      tu5_emit_const(cs, prog.fs, 0, d->fs_consts_dwords, d->fs_consts);
   tu5_emit_immediates(cs, prog.fs);

   if (d->src)
      tu5_emit_blit_src(cs, d->src, d->src_linear);

   OUT_PKT4(cs, REG_A5XX_TPL1_VS_TEX_COUNT, 1);
   OUT_RING(cs, 0);
   OUT_PKT4(cs, REG_A5XX_TPL1_FS_TEX_COUNT, 1);
   OUT_RING(cs, d->src ? 1 : 0);

   OUT_PKT7(cs, CP_DRAW_INDX_OFFSET, 3);
   OUT_RING(cs, CP_DRAW_INDX_OFFSET_0_PRIM_TYPE(DI_PT_TRISTRIP) |
                   CP_DRAW_INDX_OFFSET_0_SOURCE_SELECT(DI_SRC_SEL_AUTO_INDEX) |
                   CP_DRAW_INDX_OFFSET_0_INDEX_SIZE(INDEX4_SIZE_32_BIT) |
                   CP_DRAW_INDX_OFFSET_0_VIS_CULL(IGNORE_VISIBILITY));
   OUT_RING(cs, 1); /* NumInstances */
   OUT_RING(cs, 4); /* NumIndices */

   /* All of the draw state was written over */
   cmd->state.dirty |= TU_CMD_DIRTY_DRAW_STATE;
}

/* Clear a rectangle of the targets bound: a colour one, or depth and stencil */
static void
tu5_clear_rect(struct tu_cmd_buffer *cmd, struct tu_cs *cs, int rt,
               VkImageAspectFlags aspects, const VkClearValue *value,
               VkRect2D rect, uint32_t width, uint32_t height)
{
   struct tu5_rect_draw d = {
      .fs = rt >= 0 ? (enum global_shader)(GLOBAL_SH_FS_CLEAR0 + 1)
                    : GLOBAL_SH_FS_CLEAR0,
      .width = width,
      .height = height,
      .rect = rect,
      .rt = rt,
   };

   if (!rect.extent.width || !rect.extent.height)
      return;

   if (rt >= 0) {
      d.fs_consts = value->color.uint32;
      d.fs_consts_dwords = 4;
   } else {
      d.depth = aspects & VK_IMAGE_ASPECT_DEPTH_BIT;
      d.stencil = aspects & VK_IMAGE_ASPECT_STENCIL_BIT;
      d.z = value->depthStencil.depth;
      d.stencil_ref = value->depthStencil.stencil & 0xff;
   }

   tu5_draw_rect(cmd, cs, &d);
}

/* Bind one colour target, or the depth and stencil buffer of an image, for
 * the clears and the blits.
 */
static void
tu5_bind_surf(struct tu_cs *cs, const struct tu5_surf *color,
              const struct tu_image *zs, uint32_t level, uint32_t layer)
{
   tu5_emit_zs_surface(cs, zs, level, layer);
   tu5_emit_mrt_surf(cs, 0, color);
   for (unsigned i = 1; i < A5XX_MAX_RENDER_TARGETS; i++)
      tu5_emit_mrt_surf(cs, i, NULL);

   OUT_PKT4(cs, REG_A5XX_RB_RENDER_COMPONENTS, 1);
   OUT_RING(cs, color ? 0xf : 0x0);

   tu5_emit_msaa(cs, (VkSampleCountFlagBits)(color ? color->samples
                                                   : zs->vk.samples));

   OUT_PKT4(cs, REG_A5XX_RB_RENDER_CNTL, 1);
   OUT_RING(cs, 0x00000008);

   OUT_PKT4(cs, REG_A5XX_GRAS_SC_CNTL, 1);
   OUT_RING(cs, 0x00000008);
}

static void
tu5_bind_target(struct tu_cs *cs, const struct tu_image *image,
                enum pipe_format format, uint32_t level, uint32_t layer)
{
   if (util_format_is_depth_or_stencil(format)) {
      tu5_bind_surf(cs, NULL, image, level, layer);
   } else {
      struct tu5_surf surf = tu5_image_surf(image, 0, format, level, layer);

      tu5_bind_surf(cs, &surf, NULL, 0, 0);
   }
}

/* The clears of the attachments that the subpass that starts uses first */
static void
tu5_emit_load_clears(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   const struct tu_render_pass *pass = cmd->state.pass;
   const struct tu_framebuffer *fb = cmd->state.framebuffer;
   uint32_t subpass_idx = cmd->state.subpass - pass->subpasses;

   for (uint32_t a = 0; a < pass->attachment_count; a++) {
      const struct tu_render_pass_attachment *att = &pass->attachments[a];

      if (!att->clear_mask || att->first_subpass_idx != subpass_idx)
         continue;

      const struct tu_image_view *iview = cmd->state.attachments[a];
      enum pipe_format format = vk_format_to_pipe_format(iview->vk.format);
      bool zs = util_format_is_depth_or_stencil(format);
      VkImageAspectFlags aspects = att->clear_mask;

      /* The clear of what has depth alone, or stencil alone, is said to be
       * that of a colour, see attachment_set_ops().
       */
      if (zs && (aspects & VK_IMAGE_ASPECT_COLOR_BIT)) {
         aspects = format == PIPE_FORMAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                 : VK_IMAGE_ASPECT_DEPTH_BIT;
      }

      /* the layers of the framebuffer, or those of the views */
      uint32_t layers = pass->num_views ? att->clear_views
                                        : BITFIELD_MASK(fb->layers);

      u_foreach_bit (layer, layers) {
         tu5_bind_target(cs, iview->image, format, iview->vk.base_mip_level,
                         iview->vk.base_array_layer + layer);
         tu5_clear_rect(cmd, cs, zs ? -1 : 0, aspects,
                        &cmd->state.clear_values[a], cmd->state.render_area,
                        iview->vk.extent.width, iview->vk.extent.height);
      }
   }
}

/* What comes before and after drawing straight into the images */
static void
tu5_sysmem_begin(struct tu_cmd_buffer *cmd, struct tu_cs *cs, uint32_t width,
                 uint32_t height)
{
   const struct fd_dev_info *info = cmd->device->physical_device->info;

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
   OUT_RING(cs, A5XX_GRAS_SC_WINDOW_SCISSOR_BR_X(width - 1) |
                   A5XX_GRAS_SC_WINDOW_SCISSOR_BR_Y(height - 1));

   OUT_PKT4(cs, REG_A5XX_RB_RESOLVE_CNTL_1, 2);
   OUT_RING(cs, A5XX_RB_RESOLVE_CNTL_1_X(0) | A5XX_RB_RESOLVE_CNTL_1_Y(0));
   OUT_RING(cs, A5XX_RB_RESOLVE_CNTL_2_X(width - 1) |
                   A5XX_RB_RESOLVE_CNTL_2_Y(height - 1));

   OUT_PKT4(cs, REG_A5XX_RB_WINDOW_OFFSET, 1);
   OUT_RING(cs, A5XX_RB_WINDOW_OFFSET_X(0) | A5XX_RB_WINDOW_OFFSET_Y(0));

   OUT_PKT7(cs, CP_SET_VISIBILITY_OVERRIDE, 1);
   OUT_RING(cs, 0x1);
}

static void
tu5_sysmem_end(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   OUT_PKT7(cs, CP_SKIP_IB2_ENABLE_GLOBAL, 1);
   OUT_RING(cs, 0x0);

   tu5_emit_lrz_flush(cmd, cs);

   tu5_event_write(cmd, cs, PC_CCU_FLUSH_COLOR_TS, true);
   tu5_event_write(cmd, cs, PC_CCU_FLUSH_DEPTH_TS, true);
}

static enum pipe_format tu5_copy_format(enum pipe_format format);

void
tu5_clear_image(struct tu_cmd_buffer *cmd, struct tu_image *image,
                const VkClearValue *value, uint32_t range_count,
                const VkImageSubresourceRange *ranges)
{
   struct tu_cs *cs = &cmd->cs;
   enum pipe_format format = vk_format_to_pipe_format(image->vk.format);
   bool zs = util_format_is_depth_or_stencil(format);
   VkClearValue packed = {};

   /* Stencil alone is not an attachment: it is cleared as bytes */
   if (format == PIPE_FORMAT_S8_UINT) {
      packed.color.uint32[0] = value->depthStencil.stencil & 0xff;
      format = PIPE_FORMAT_R8_UINT;
      value = &packed;
      zs = false;
   }

   /* What cannot be drawn to is cleared as the unsigned integers of its
    * size, to the bits of the colour.
    */
   if (!zs && !tu5_format_color_supported(format) &&
       !util_format_is_compressed(format) &&
       tu5_copy_format(format) != PIPE_FORMAT_NONE) {
      enum pipe_format as = tu5_copy_format(format);
      union {
         uint8_t u8[16];
         uint16_t u16[8];
         uint32_t u32[4];
      } bits = {};

      util_format_pack_rgba(format, bits.u8, value->color.uint32, 1);

      switch (util_format_get_blocksize(format)) {
      case 1:
         packed.color.uint32[0] = bits.u8[0];
         break;
      case 2:
         packed.color.uint32[0] = bits.u16[0];
         break;
      default:
         memcpy(packed.color.uint32, bits.u32, sizeof(bits.u32));
         break;
      }

      format = as;
      value = &packed;
   }

   for (uint32_t i = 0; i < range_count; i++) {
      const VkImageSubresourceRange *range = &ranges[i];
      uint32_t level_count = vk_image_subresource_level_count(&image->vk, range);
      uint32_t layer_count = vk_image_subresource_layer_count(&image->vk, range);

      for (uint32_t l = 0; l < level_count; l++) {
         uint32_t level = range->baseMipLevel + l;
         uint32_t width = u_minify(image->vk.extent.width, level);
         uint32_t height = u_minify(image->vk.extent.height, level);
         uint32_t layers = image->vk.image_type == VK_IMAGE_TYPE_3D
                              ? u_minify(image->vk.extent.depth, level)
                              : layer_count;
         VkRect2D rect = { { 0, 0 }, { width, height } };

         tu5_sysmem_begin(cmd, cs, width, height);
         for (uint32_t layer = 0; layer < layers; layer++) {
            tu5_bind_target(cs, image, format, level,
                            range->baseArrayLayer + layer);
            tu5_clear_rect(cmd, cs, zs ? -1 : 0, range->aspectMask, value,
                           rect, width, height);
         }
         tu5_sysmem_end(cmd, cs);
      }
   }

   /* The depth and stencil are written through the depth cache, which the
    * barriers of the transfers, being those of Adreno 6xx where the clears
    * are blits, do not clean.
    */
   if (zs) {
      tu5_emit_flushes(cmd, cs,
                       TU_CMD_FLAG_CCU_CLEAN_DEPTH |
                       TU_CMD_FLAG_CACHE_INVALIDATE |
                       TU_CMD_FLAG_WAIT_FOR_IDLE);
   }
}

/* Copies and blits, outside of render passes: a rectangle sampling the source
 * drawn into the destination.
 */
static void
tu5_blit_rect(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
              const struct tu5_surf *dst, VkRect2D dst_rect,
              const struct tu5_surf *src, float s0, float t0, float s1,
              float t1, bool linear, uint32_t keep_mask,
              enum global_shader fs = GLOBAL_SH_FS_BLIT, int sample = -1,
              float r = 0.0f)
{
   /* A sample of the source into the same of the destination */
   const uint32_t consts[4] = { (uint32_t)MAX2(sample, 0) };
   struct tu5_rect_draw d = {
      .fs = fs,
      .width = dst->width,
      .height = dst->height,
      .rect = dst_rect,
      .rt = 0,
      .src = src,
      .src_linear = linear,
      .s0 = s0,
      .t0 = t0,
      .s1 = s1,
      .t1 = t1,
      .r = r,
      .keep_mask = keep_mask,
   };

   if (!dst_rect.extent.width || !dst_rect.extent.height)
      return;

   if (sample >= 0) {
      d.fs_consts = consts;
      d.fs_consts_dwords = 4;
      d.sample_mask = 1u << sample;
   }

   tu5_bind_surf(cs, dst, NULL, 0, 0);
   tu5_draw_rect(cmd, cs, &d);
}

/* The window of the blits is not that of a framebuffer: the largest */
static void
tu5_blit_begin(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   tu5_sysmem_begin(cmd, cs, 16384, 16384);
}

/* Texels are copied as unsigned integers of their size, for the bits to come
 * through as they are.
 */
static enum pipe_format
tu5_copy_format(enum pipe_format format)
{
   switch (util_format_get_blocksize(format)) {
   case 1:
      return PIPE_FORMAT_R8_UINT;
   case 2:
      return PIPE_FORMAT_R16_UINT;
   case 4:
      return PIPE_FORMAT_R32_UINT;
   case 8:
      return PIPE_FORMAT_R32G32_UINT;
   case 16:
      return PIPE_FORMAT_R32G32B32A32_UINT;
   default:
      return PIPE_FORMAT_NONE;
   }
}

/* The plane of an aspect of an image, the format to copy it as, and the
 * components of that format the aspect does not have.
 */
static enum pipe_format
tu5_aspect_copy_format(const struct tu_image *image, VkImageAspectFlags aspect,
                       unsigned *plane, uint32_t *keep_mask)
{
   *plane = tu6_plane_index(image->vk.format, aspect);
   *keep_mask = 0;

   enum pipe_format format = tu6_plane_format(image->vk.format, *plane);

   if (format == PIPE_FORMAT_Z24_UNORM_S8_UINT ||
       format == PIPE_FORMAT_Z24X8_UNORM) {
      /* depth in the low three bytes, stencil in the high one */
      if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT)
         *keep_mask = 0x8;
      else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT)
         *keep_mask = 0x7;
      return PIPE_FORMAT_R8G8B8A8_UINT;
   }

   return tu5_copy_format(format);
}

/* A rectangle of texels between an image surface and the rows of a buffer.
 * The targets and the textures take addresses and pitches that are multiples
 * of 64 bytes: a buffer that is not laid out so goes row by row, each from
 * the 64 bytes before it.
 */
static void
tu5_copy_buffer_rect(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                     const struct tu5_surf *image, uint32_t x, uint32_t y,
                     uint64_t va, uint32_t pitch, uint32_t w, uint32_t h,
                     bool to_buffer, uint32_t keep_mask,
                     enum pipe_format format, const unsigned char *swiz)
{
   uint32_t cpp = util_format_get_blocksize(format);
   struct tu5_surf image_src = *image;

   /* what is read, of the image or of the buffer */
   image_src.swiz = swiz;
   image = &image_src;

   if (!(va & 63) && !(pitch & 63)) {
      struct tu5_surf buf = tu5_buffer_surf(format, va, w, h, pitch);

      buf.swiz = swiz;
      if (to_buffer) {
         tu5_blit_rect(cmd, cs, &buf, (VkRect2D) { { 0, 0 }, { w, h } }, image,
                       x, y, x + w, y + h, false, 0);
      } else {
         tu5_blit_rect(cmd, cs, image, (VkRect2D) { { (int32_t)x, (int32_t)y }, { w, h } },
                       &buf, 0, 0, w, h, false, keep_mask);
      }
      return;
   }

   for (uint32_t row = 0; row < h; row++) {
      uint64_t row_va = va + (uint64_t)row * pitch;
      uint32_t bx = (row_va & 63) / cpp;
      struct tu5_surf buf =
         tu5_buffer_surf(format, row_va & ~63ull, bx + w, 1,
                         align((bx + w) * cpp, 64));

      buf.swiz = swiz;
      if (to_buffer) {
         tu5_blit_rect(cmd, cs, &buf, (VkRect2D) { { (int32_t)bx, 0 }, { w, 1 } },
                       image, x, y + row, x + w, y + row + 1, false, 0);
      } else {
         tu5_blit_rect(cmd, cs, image,
                       (VkRect2D) { { (int32_t)x, (int32_t)(y + row) }, { w, 1 } },
                       &buf, bx, 0, bx + w, 1, false, keep_mask);
      }
   }
}

static void
tu5_copy_buffer_image(struct tu_cmd_buffer *cmd, struct tu_buffer *buffer,
                      struct tu_image *image, const VkBufferImageCopy2 *info,
                      bool to_buffer)
{
   struct tu_cs *cs = &cmd->cs;
   const VkImageSubresourceLayers *sub = &info->imageSubresource;
   bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   uint32_t layers = MAX2(info->imageExtent.depth,
                          vk_image_subresource_layer_count(&image->vk, sub));
   unsigned plane;
   uint32_t keep_mask;
   enum pipe_format format =
      tu5_aspect_copy_format(image, sub->aspectMask, &plane, &keep_mask);

   if (format == PIPE_FORMAT_NONE) {
      mesa_logw("a5xx: copy between a buffer and %s not implemented",
                util_format_name(vk_format_to_pipe_format(image->vk.format)));
      return;
   }

   /* The stencil of packed depth and stencil is a byte a texel in the
    * buffer, and the last of the four of a texel of the image.
    */
   static const unsigned char all_x[4] = { PIPE_SWIZZLE_X, PIPE_SWIZZLE_X,
                                           PIPE_SWIZZLE_X, PIPE_SWIZZLE_X };
   static const unsigned char all_w[4] = { PIPE_SWIZZLE_W, PIPE_SWIZZLE_W,
                                           PIPE_SWIZZLE_W, PIPE_SWIZZLE_W };
   bool packed_stencil =
      keep_mask == 0x7 && sub->aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT;
   enum pipe_format buf_format = packed_stencil ? PIPE_FORMAT_R8_UINT : format;
   const unsigned char *swiz =
      !packed_stencil ? NULL : to_buffer ? all_w : all_x;

   /* in blocks, for the compressed formats */
   enum pipe_format image_format = image->layout[plane].format;
   uint32_t bw = util_format_get_blockwidth(image_format);
   uint32_t bh = util_format_get_blockheight(image_format);
   uint32_t cpp = util_format_get_blocksize(buf_format);
   uint32_t row_length = info->bufferRowLength ? info->bufferRowLength
                                               : info->imageExtent.width;
   uint32_t image_height = info->bufferImageHeight ? info->bufferImageHeight
                                                   : info->imageExtent.height;
   uint32_t pitch = DIV_ROUND_UP(row_length, bw) * cpp;
   uint64_t layer_size = (uint64_t)DIV_ROUND_UP(image_height, bh) * pitch;

   tu5_blit_begin(cmd, cs);
   for (uint32_t i = 0; i < layers; i++) {
      uint32_t layer = (is_3d ? info->imageOffset.z : sub->baseArrayLayer) + i;
      struct tu5_surf surf =
         tu5_image_surf(image, plane, format, sub->mipLevel, layer);

      tu5_copy_buffer_rect(cmd, cs, &surf, info->imageOffset.x / bw,
                           info->imageOffset.y / bh,
                           buffer->iova + info->bufferOffset + layer_size * i,
                           pitch, DIV_ROUND_UP(info->imageExtent.width, bw),
                           DIV_ROUND_UP(info->imageExtent.height, bh),
                           to_buffer, keep_mask, buf_format, swiz);
   }
   tu5_sysmem_end(cmd, cs);
}

void
tu5_copy_buffer_to_image(struct tu_cmd_buffer *cmd, struct tu_buffer *src,
                         struct tu_image *dst, const VkBufferImageCopy2 *info)
{
   tu5_copy_buffer_image(cmd, src, dst, info, false);
}

void
tu5_copy_image_to_buffer(struct tu_cmd_buffer *cmd, struct tu_image *src,
                         struct tu_buffer *dst, const VkBufferImageCopy2 *info)
{
   tu5_copy_buffer_image(cmd, dst, src, info, true);
}

void
tu5_copy_image(struct tu_cmd_buffer *cmd, struct tu_image *src_image,
               struct tu_image *dst_image, const VkImageCopy2 *info)
{
   struct tu_cs *cs = &cmd->cs;
   unsigned src_plane, dst_plane;
   uint32_t src_keep, dst_keep;
   enum pipe_format src_format =
      tu5_aspect_copy_format(src_image, info->srcSubresource.aspectMask,
                             &src_plane, &src_keep);
   enum pipe_format dst_format =
      tu5_aspect_copy_format(dst_image, info->dstSubresource.aspectMask,
                             &dst_plane, &dst_keep);
   bool src_3d = src_image->vk.image_type == VK_IMAGE_TYPE_3D;
   bool dst_3d = dst_image->vk.image_type == VK_IMAGE_TYPE_3D;
   uint32_t layers =
      MAX2(info->extent.depth,
           vk_image_subresource_layer_count(&src_image->vk,
                                            &info->srcSubresource));

   if (src_format == PIPE_FORMAT_NONE || src_format != dst_format) {
      mesa_logw("a5xx: copy from %s to %s not implemented",
                util_format_name(vk_format_to_pipe_format(src_image->vk.format)),
                util_format_name(vk_format_to_pipe_format(dst_image->vk.format)));
      return;
   }

   tu5_blit_begin(cmd, cs);
   for (uint32_t i = 0; i < layers; i++) {
      struct tu5_surf src = tu5_image_surf(
         src_image, src_plane, src_format, info->srcSubresource.mipLevel,
         (src_3d ? info->srcOffset.z : info->srcSubresource.baseArrayLayer) + i);
      struct tu5_surf dst = tu5_image_surf(
         dst_image, dst_plane, dst_format, info->dstSubresource.mipLevel,
         (dst_3d ? info->dstOffset.z : info->dstSubresource.baseArrayLayer) + i);
      /* in blocks, for the compressed formats: a block of one is a texel
       * of the other
       */
      enum pipe_format sf = src_image->layout[src_plane].format;
      enum pipe_format df = dst_image->layout[dst_plane].format;
      uint32_t sbw = util_format_get_blockwidth(sf);
      uint32_t sbh = util_format_get_blockheight(sf);
      uint32_t w = DIV_ROUND_UP(info->extent.width, sbw);
      uint32_t h = DIV_ROUND_UP(info->extent.height, sbh);
      float sx = info->srcOffset.x / sbw, sy = info->srcOffset.y / sbh;
      VkRect2D rect = {
         { info->dstOffset.x / (int32_t)util_format_get_blockwidth(df),
           info->dstOffset.y / (int32_t)util_format_get_blockheight(df) },
         { w, h }
      };

      if (src.samples > 1) {
         /* sample by sample */
         for (uint32_t s = 0; s < src.samples; s++) {
            tu5_blit_rect(cmd, cs, &dst, rect, &src, sx, sy, sx + w, sy + h,
                          false, dst_keep, GLOBAL_SH_FS_COPY_MS5, s);
         }
         continue;
      }

      tu5_blit_rect(cmd, cs, &dst, rect, &src, sx, sy, sx + w, sy + h, false,
                    dst_keep);
   }
   tu5_sysmem_end(cmd, cs);
}

/* TODO: depth and stencil, multisampled sources */
void
tu5_blit_image(struct tu_cmd_buffer *cmd, struct tu_image *src_image,
               struct tu_image *dst_image, const VkImageBlit2 *info,
               VkFilter filter)
{
   struct tu_cs *cs = &cmd->cs;
   enum pipe_format src_format = vk_format_to_pipe_format(src_image->vk.format);
   enum pipe_format dst_format = vk_format_to_pipe_format(dst_image->vk.format);
   uint32_t layers = vk_image_subresource_layer_count(&dst_image->vk,
                                                      &info->dstSubresource);

   if (util_format_is_depth_or_stencil(dst_format)) {
      mesa_logw("a5xx: blit from %s to %s not implemented",
                util_format_name(src_format), util_format_name(dst_format));
      return;
   }

   /* The rectangle is drawn from its top left: mirror the source instead */
   int32_t dx0 = info->dstOffsets[0].x, dx1 = info->dstOffsets[1].x;
   int32_t dy0 = info->dstOffsets[0].y, dy1 = info->dstOffsets[1].y;
   float s0 = info->srcOffsets[0].x, s1 = info->srcOffsets[1].x;
   float t0 = info->srcOffsets[0].y, t1 = info->srcOffsets[1].y;

   if (dx0 > dx1) {
      int32_t tmp = dx0; dx0 = dx1; dx1 = tmp;
      float ftmp = s0; s0 = s1; s1 = ftmp;
   }
   if (dy0 > dy1) {
      int32_t tmp = dy0; dy0 = dy1; dy1 = tmp;
      float ftmp = t0; t0 = t1; t1 = ftmp;
   }

   VkRect2D rect = { { dx0, dy0 },
                     { (uint32_t)(dx1 - dx0), (uint32_t)(dy1 - dy0) } };

   /* The slices of the 3D images, as tu6_blit_image() has them: when their
    * number is not that of the source, or when they are mirrored, each one
    * samples the source as a 3D texture.
    */
   bool src_3d = src_image->vk.image_type == VK_IMAGE_TYPE_3D;
   bool dst_3d = dst_image->vk.image_type == VK_IMAGE_TYPE_3D;
   bool z_scale = false;
   int32_t src0_z = info->srcOffsets[0].z;
   int32_t src1_z = info->srcOffsets[1].z;
   int32_t dst_z = MIN2(info->dstOffsets[0].z, info->dstOffsets[1].z);

   if (dst_3d || src_3d) {
      layers = abs(info->dstOffsets[1].z - info->dstOffsets[0].z);
      if ((info->srcOffsets[1].z - info->srcOffsets[0].z !=
           info->dstOffsets[1].z - info->dstOffsets[0].z) ||
          info->srcOffsets[1].z < info->srcOffsets[0].z)
         z_scale = src_3d;
      if (info->dstOffsets[1].z < info->dstOffsets[0].z) {
         src0_z = info->srcOffsets[1].z;
         src1_z = info->srcOffsets[0].z;
      }
      if (vk_image_subresource_layer_count(&dst_image->vk,
                                           &info->dstSubresource) > 1) {
         layers = vk_image_subresource_layer_count(&dst_image->vk,
                                                   &info->dstSubresource);
      }
   }

   tu5_blit_begin(cmd, cs);
   for (uint32_t i = 0; i < layers; i++) {
      struct tu5_surf dst =
         tu5_image_surf(dst_image, 0, dst_format, info->dstSubresource.mipLevel,
                        (dst_3d ? dst_z : info->dstSubresource.baseArrayLayer) +
                           i);

      if (z_scale) {
         struct tu5_surf src = tu5_image_surf_3d(src_image, src_format,
                                                 info->srcSubresource.mipLevel);
         float t = ((float)i + 0.5f) / (float)layers;

         tu5_blit_rect(cmd, cs, &dst, rect, &src, s0, t0, s1, t1,
                       filter == VK_FILTER_LINEAR, 0, GLOBAL_SH_FS_BLIT_ZSCALE,
                       -1, t * (src1_z - src0_z) + src0_z);
         continue;
      }

      struct tu5_surf src =
         tu5_image_surf(src_image, 0, src_format, info->srcSubresource.mipLevel,
                        (src_3d ? MIN2(src0_z, src1_z)
                                : info->srcSubresource.baseArrayLayer) + i);

      tu5_blit_rect(cmd, cs, &dst, rect, &src, s0, t0, s1, t1,
                    filter == VK_FILTER_LINEAR, 0);
   }
   tu5_sysmem_end(cmd, cs);
}

/* The resolves: what is not filtered, the integers, takes the first sample */
static enum global_shader
tu5_resolve_fs(enum pipe_format format, uint32_t samples)
{
   if (samples == 1 || util_format_is_pure_integer(format) ||
       util_format_is_depth_or_stencil(format))
      return GLOBAL_SH_FS_RESOLVE5_1;

   return samples == 2 ? GLOBAL_SH_FS_RESOLVE5_2 : GLOBAL_SH_FS_RESOLVE5_4;
}

/* vkCmdResolveImage() */
void
tu5_resolve_image(struct tu_cmd_buffer *cmd, struct tu_image *src_image,
                  struct tu_image *dst_image, const VkImageResolve2 *info)
{
   struct tu_cs *cs = &cmd->cs;
   enum pipe_format src_format = vk_format_to_pipe_format(src_image->vk.format);
   enum pipe_format dst_format = vk_format_to_pipe_format(dst_image->vk.format);
   uint32_t layers = MAX2(info->extent.depth,
                          vk_image_subresource_layer_count(&dst_image->vk,
                                                           &info->dstSubresource));
   VkRect2D rect = { { info->dstOffset.x, info->dstOffset.y },
                     { info->extent.width, info->extent.height } };

   tu5_blit_begin(cmd, cs);
   for (uint32_t i = 0; i < layers; i++) {
      struct tu5_surf src =
         tu5_image_surf(src_image, 0, src_format, info->srcSubresource.mipLevel,
                        info->srcSubresource.baseArrayLayer + i);
      struct tu5_surf dst =
         tu5_image_surf(dst_image, 0, dst_format, info->dstSubresource.mipLevel,
                        info->dstSubresource.baseArrayLayer + i);

      tu5_blit_rect(cmd, cs, &dst, rect, &src, info->srcOffset.x,
                    info->srcOffset.y, info->srcOffset.x + info->extent.width,
                    info->srcOffset.y + info->extent.height, false, 0,
                    tu5_resolve_fs(src_format, src.samples));
   }
   tu5_sysmem_end(cmd, cs);
}

/* The resolve of an attachment into another at the end of a subpass, in the
 * draw stream of the render pass.
 *
 * TODO: depth and stencil
 */
void
tu5_resolve_sysmem(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                   const struct tu_image_view *src_view,
                   const struct tu_image_view *dst_view, uint32_t layer_mask,
                   uint32_t layers, const VkRect2D *rect)
{
   /* the layers of the framebuffer, or those of the views */
   if (!layer_mask)
      layer_mask = BITFIELD_MASK(layers);

   enum pipe_format src_format = vk_format_to_pipe_format(src_view->vk.format);
   enum pipe_format dst_format = vk_format_to_pipe_format(dst_view->vk.format);

   if (util_format_is_depth_or_stencil(dst_format)) {
      /* The first sample, the one resolve mode there is of depth and
       * stencil: copied as the integers of their size, aspect by aspect, as
       * the copies have them.
       */
      VkImageAspectFlags aspects = 0;

      if (util_format_has_depth(util_format_description(dst_format)))
         aspects |= VK_IMAGE_ASPECT_DEPTH_BIT;
      if (util_format_has_stencil(util_format_description(dst_format)))
         aspects |= VK_IMAGE_ASPECT_STENCIL_BIT;

      u_foreach_bit (aspect_bit, aspects) {
         VkImageAspectFlags aspect = (VkImageAspectFlags)BITFIELD_BIT(aspect_bit);
         unsigned src_plane, dst_plane;
         uint32_t src_keep, dst_keep;
         enum pipe_format sf = tu5_aspect_copy_format(src_view->image, aspect,
                                                      &src_plane, &src_keep);
         enum pipe_format df = tu5_aspect_copy_format(dst_view->image, aspect,
                                                      &dst_plane, &dst_keep);

         if (sf == PIPE_FORMAT_NONE || sf != df)
            continue;

         u_foreach_bit (i, layer_mask) {
            struct tu5_surf src =
               tu5_image_surf(src_view->image, src_plane, sf,
                              src_view->vk.base_mip_level,
                              src_view->vk.base_array_layer + i);
            struct tu5_surf dst =
               tu5_image_surf(dst_view->image, dst_plane, df,
                              dst_view->vk.base_mip_level,
                              dst_view->vk.base_array_layer + i);

            tu5_blit_rect(cmd, cs, &dst, *rect, &src, rect->offset.x,
                          rect->offset.y, rect->offset.x + rect->extent.width,
                          rect->offset.y + rect->extent.height, false, dst_keep,
                          GLOBAL_SH_FS_RESOLVE5_1);
         }
      }

      return;
   }

   u_foreach_bit (i, layer_mask) {
      struct tu5_surf src =
         tu5_image_surf(src_view->image, 0, src_format,
                        src_view->vk.base_mip_level,
                        src_view->vk.base_array_layer + i);
      struct tu5_surf dst =
         tu5_image_surf(dst_view->image, 0, dst_format,
                        dst_view->vk.base_mip_level,
                        dst_view->vk.base_array_layer + i);

      tu5_blit_rect(cmd, cs, &dst, *rect, &src, rect->offset.x, rect->offset.y,
                    rect->offset.x + rect->extent.width,
                    rect->offset.y + rect->extent.height, false, 0,
                    tu5_resolve_fs(src_format, src.samples));
   }
}

/* Buffers are copied as rows of bytes, each from the 64 bytes before it */
void
tu5_copy_buffer(struct tu_cmd_buffer *cmd, uint64_t dst_va, uint64_t src_va,
                uint64_t size)
{
   struct tu_cs *cs = &cmd->cs;

   tu5_blit_begin(cmd, cs);
   while (size) {
      uint32_t src_x = src_va & 63, dst_x = dst_va & 63;
      uint32_t width = MIN2(size, 0x4000 - MAX2(src_x, dst_x));
      struct tu5_surf src =
         tu5_buffer_surf(PIPE_FORMAT_R8_UINT, src_va & ~63ull, src_x + width, 1,
                         align(src_x + width, 64));
      struct tu5_surf dst =
         tu5_buffer_surf(PIPE_FORMAT_R8_UINT, dst_va & ~63ull, dst_x + width, 1,
                         align(dst_x + width, 64));

      tu5_blit_rect(cmd, cs, &dst, (VkRect2D) { { (int32_t)dst_x, 0 }, { width, 1 } },
                    &src, src_x, 0, src_x + width, 1, false, 0);

      src_va += width;
      dst_va += width;
      size -= width;
   }
   tu5_sysmem_end(cmd, cs);
}

/* vkCmdFillBuffer(): the offset and the size are multiples of four */
void
tu5_fill_buffer(struct tu_cmd_buffer *cmd, uint64_t dst_va, uint64_t size,
                uint32_t data)
{
   struct tu_cs *cs = &cmd->cs;
   VkClearValue value = {};
   uint32_t dwords = size / 4;

   value.color.uint32[0] = data;

   tu5_blit_begin(cmd, cs);
   while (dwords) {
      uint32_t dst_x = (dst_va & 63) / 4;
      uint32_t width = MIN2(dwords, 0x4000 - dst_x);
      struct tu5_surf dst =
         tu5_buffer_surf(PIPE_FORMAT_R32_UINT, dst_va & ~63ull, dst_x + width,
                         1, align((dst_x + width) * 4, 64));

      tu5_bind_surf(cs, &dst, NULL, 0, 0);
      tu5_clear_rect(cmd, cs, 0, VK_IMAGE_ASPECT_COLOR_BIT, &value,
                     (VkRect2D) { { (int32_t)dst_x, 0 }, { width, 1 } },
                     dst.width, dst.height);

      dst_va += width * 4;
      dwords -= width;
   }
   tu5_sysmem_end(cmd, cs);
}

/* vkCmdClearAttachments(), in the draw stream of the render pass */
void
tu5_clear_attachments(struct tu_cmd_buffer *cmd, uint32_t attachment_count,
                      const VkClearAttachment *attachments,
                      uint32_t rect_count, const VkClearRect *rects)
{
   const struct tu_subpass *subpass = cmd->state.subpass;
   const struct tu_framebuffer *fb = cmd->state.framebuffer;
   struct tu_cs *cs = &cmd->draw_cs;

   for (uint32_t i = 0; i < attachment_count; i++) {
      const VkClearAttachment *att = &attachments[i];
      int rt = -1;

      if (att->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
         if (att->colorAttachment >= subpass->color_count ||
             subpass->color_attachments[att->colorAttachment].attachment ==
                VK_ATTACHMENT_UNUSED)
            continue;
         rt = att->colorAttachment;
      } else if (subpass->depth_stencil_attachment.attachment ==
                 VK_ATTACHMENT_UNUSED) {
         continue;
      }

      /* The layers past the first: the target is bound to each in turn,
       * then to the first again.
       */
      uint32_t a = rt >= 0 ? subpass->color_attachments[rt].attachment
                           : subpass->depth_stencil_attachment.attachment;
      /* A secondary command buffer need not know the framebuffer: there,
       * the first layer alone is cleared, and the window is the largest.
       *
       * TODO: the other layers, as the primary command buffer runs it
       */
      const struct tu_image_view *iview =
         cmd->state.attachments ? cmd->state.attachments[a] : NULL;
      enum pipe_format format = vk_format_to_pipe_format(
         cmd->state.pass->attachments[a].format);
      uint32_t width = fb ? fb->width : 16384;
      uint32_t height = fb ? fb->height : 16384;
      /* with multiview, the layer of the view drawn last is what is bound */
      bool rebound = subpass->multiview_mask != 0;

      for (uint32_t r = 0; r < rect_count; r++) {
         /* with multiview, the layers of the views */
         uint32_t layers = subpass->multiview_mask
                              ? subpass->multiview_mask
                              : BITFIELD_MASK(rects[r].layerCount)
                                   << rects[r].baseArrayLayer;

         u_foreach_bit (layer, layers) {

            if (!iview && layer)
               continue;

            if (layer || rebound) {
               if (rt >= 0) {
                  struct tu5_surf surf =
                     tu5_image_surf(iview->image, 0, format,
                                    iview->vk.base_mip_level,
                                    iview->vk.base_array_layer + layer);

                  tu5_emit_mrt_surf(cs, rt, &surf);
               } else {
                  tu5_emit_zs_surface(cs, iview->image,
                                      iview->vk.base_mip_level,
                                      iview->vk.base_array_layer + layer);
               }
               rebound = layer != 0;
            }

            tu5_clear_rect(cmd, cs, rt, att->aspectMask, &att->clearValue,
                           rects[r].rect, width, height);
         }
      }

      if (rebound) {
         if (rt >= 0) {
            struct tu5_surf surf =
               tu5_image_surf(iview->image, 0, format, iview->vk.base_mip_level,
                              iview->vk.base_array_layer);

            tu5_emit_mrt_surf(cs, rt, &surf);
         } else {
            tu5_emit_zs_surface(cs, iview->image, iview->vk.base_mip_level,
                                iview->vk.base_array_layer);
         }
      }
   }
}

void
tu5_emit_subpass_begin(struct tu_cmd_buffer *cmd)
{
   struct tu_cs *cs = &cmd->draw_cs;

   tu5_emit_load_clears(cmd, cs);

   tu5_emit_zs(cmd, cs, 0);
   tu5_emit_mrt(cmd, cs, 0);
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
   const struct tu_framebuffer *fb = cmd->state.framebuffer;
   struct tu_cs *cs = &cmd->cs;

   tu5_sysmem_begin(cmd, cs, fb->width, fb->height);
   tu_cs_emit_call(cs, &cmd->draw_cs);
   /* what the queries of the render pass end with */
   tu_cs_emit_call(cs, &cmd->draw_epilogue_cs);

   /* The resolves of the last subpass: those of the others are in the draw
    * stream, see tu_CmdNextSubpass2().
    */
   const struct tu_subpass *subpass = cmd->state.subpass;

   if (subpass->resolve_attachments) {
      /* what was drawn is to be read as textures */
      tu5_emit_flushes(cmd, cs,
                       TU_CMD_FLAG_CCU_CLEAN_COLOR |
                       TU_CMD_FLAG_CCU_CLEAN_DEPTH |
                       TU_CMD_FLAG_CACHE_INVALIDATE |
                       TU_CMD_FLAG_WAIT_FOR_IDLE);

      for (unsigned i = 0; i < subpass->resolve_count; i++) {
         uint32_t a = subpass->resolve_attachments[i].attachment;

         if (a == VK_ATTACHMENT_UNUSED)
            continue;

         uint32_t src = tu_subpass_get_attachment_to_resolve(subpass, i);

         tu5_resolve_sysmem(cmd, cs, cmd->state.attachments[src],
                            cmd->state.attachments[a],
                            subpass->multiview_mask, fb->layers,
                            &cmd->state.render_area);
      }
   }

   tu5_sysmem_end(cmd, cs);
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

   /* An empty scissor: top left past bottom right. And so when no sample is
    * to be written, which a sample mask of zero does not get.
    */
   const uint32_t samples = cmd->state.subpass->samples;
   const bool no_samples =
      !(dyn->ms.sample_mask & BITFIELD_MASK(MAX2(samples, 1)));

   if (maxx == minx || maxy == miny || no_samples) {
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
         /* of the render pass: a secondary command buffer need not know the
          * framebuffer
          */
         enum pipe_format format = vk_format_to_pipe_format(
            cmd->state.pass->attachments[subpass->color_attachments[i].attachment]
               .format);
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
                   COND(fp->writes_pos, A5XX_RB_FS_OUTPUT_CNTL_FRAG_WRITES_Z) |
                   /* found by trying: next to FRAG_WRITES_Z, as on Adreno
                    * 6xx, the bit for the sample mask the shader writes
                    */
                   COND(ir3_find_output_regid(fp, FRAG_RESULT_SAMPLE_MASK) !=
                           regid(63, 0),
                        0x40));

   OUT_PKT4(cs, REG_A5XX_SP_FS_OUTPUT_CNTL, 1);
   OUT_RING(cs, A5XX_SP_FS_OUTPUT_CNTL_MRT(nr) |
                   A5XX_SP_FS_OUTPUT_CNTL_DEPTH_REGID(posz_regid) |
                   A5XX_SP_FS_OUTPUT_CNTL_SAMPLEMASK_REGID(
                      ir3_find_output_regid(fp, FRAG_RESULT_SAMPLE_MASK)));
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
      .vs_shader = tu5_shader_from(vs_shader),
      .fs_shader = tu5_shader_from(fs_shader),
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

   tu5_emit_user_consts(cmd, cs, vs_shader);
   tu5_emit_user_consts(cmd, cs, fs_shader);
   tu5_emit_textures(cmd, cs, vs_shader);
   tu5_emit_textures(cmd, cs, fs_shader);
   tu5_emit_ssbos(cmd, cs, fs_shader);
   tu5_emit_images(cmd, cs, fs_shader);

   OUT_PKT4(cs, REG_A5XX_VFD_INDEX_OFFSET, 2);
   OUT_RING(cs, cmd->state.last_vs_params.vertex_offset);
   OUT_RING(cs, cmd->state.last_vs_params.first_instance);

   /* see tu5_pack_border_color() */
   OUT_PKT4(cs, REG_A5XX_TPL1_TP_BORDER_COLOR_BASE_ADDR_LO, 2);
   tu_cs_emit_qw(cs, global_iova(cmd, bcolor_builtin));

   OUT_PKT4(cs, REG_A5XX_TPL1_CS_TEX_COUNT, 1);
   OUT_RING(cs, 0);

   cmd->state.dirty = (enum tu_cmd_dirty_bits)0;
   vk_dynamic_graphics_state_clear_dirty(&cmd->vk.dynamic_graphics_state);
   (void)dyn;

   return VK_SUCCESS;
}

/* Before the commands of a secondary command buffer, with multiview: where
 * each view draws, for tu5_bind_view().
 */
void
tu5_write_view_bases(struct tu_cmd_buffer *cmd, struct tu_cs *cs)
{
   const struct tu_subpass *subpass = cmd->state.subpass;

   if (!subpass || !subpass->multiview_mask || !cmd->state.attachments)
      return;

   u_foreach_bit (view, subpass->multiview_mask) {
      uint64_t bases[10] = {};

      for (unsigned i = 0; i < subpass->color_count; i++) {
         uint32_t a = subpass->color_attachments[i].attachment;

         if (a == VK_ATTACHMENT_UNUSED)
            continue;

         const struct tu_image_view *iview = cmd->state.attachments[a];

         bases[i] = tu5_image_surf(iview->image, 0,
                                   vk_format_to_pipe_format(iview->vk.format),
                                   iview->vk.base_mip_level,
                                   iview->vk.base_array_layer + view).iova;
      }

      uint32_t a = subpass->depth_stencil_attachment.attachment;

      if (a != VK_ATTACHMENT_UNUSED) {
         const struct tu_image_view *iview = cmd->state.attachments[a];
         const struct tu_image *image = iview->image;
         uint32_t level = iview->vk.base_mip_level;
         uint32_t layer = iview->vk.base_array_layer + view;

         bases[8] = image->iova +
                    fdl_surface_offset(&image->layout[0], level, layer);
         if (image->vk.format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
            bases[9] = image->iova +
                       fdl_surface_offset(&image->layout[1], level, layer);
         }
      }

      OUT_PKT7(cs, CP_MEM_WRITE, 2 + 20);
      tu_cs_emit_qw(cs, global_iova_arr(cmd, a5xx_view_bases, view * 10));
      for (unsigned i = 0; i < 10; i++)
         tu_cs_emit_qw(cs, bases[i]);
   }

   OUT_PKT7(cs, CP_WAIT_MEM_WRITES, 0);
}
