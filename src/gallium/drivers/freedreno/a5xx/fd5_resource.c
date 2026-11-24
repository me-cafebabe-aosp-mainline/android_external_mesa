/*
 * Copyright © 2018 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "drm-uapi/drm_fourcc.h"

#include "fd5_resource.h"
#include "fd5_blitter.h"
#include "fd5_hw.h"

static void
setup_lrz(struct fd_resource *rsc)
{
   struct fd_screen *screen = fd_screen(rsc->b.b.screen);
   fdl5_lrz_layout_init(&rsc->lrz_layout, rsc->b.b.width0, rsc->b.b.height0,
                        rsc->b.b.nr_samples);
   rsc->lrz = fd_bo_new(screen->dev, rsc->lrz_layout.lrz_total_size,
                        FD_BO_NOMAP, "lrz");
}

uint32_t
fd5_layout_resource(struct fd_resource *rsc, enum fd_layout_type type)
{
   struct pipe_resource *prsc = &rsc->b.b;
   bool ubwc = false;
   unsigned tile_mode = 0;

   if (FD_DBG(LRZ) && has_depth(prsc->format) && !is_z32(prsc->format))
      setup_lrz(rsc);

   if (type >= FD_LAYOUT_TILED)
      tile_mode = fd5_tile_mode(prsc);
   if (type == FD_LAYOUT_UBWC)
      ubwc = true;

   struct fdl_image_params params = fd_image_params(prsc, ubwc, tile_mode, 0);

   fdl5_layout_image(&rsc->layout, &params);

   return rsc->layout.size;
}

/* UBWC, of the buffers gralloc (minigbm) makes for the window and the
 * composition: of the colour formats of 32 bits a texel, the flags first
 * (see fdl5_layout()), as the Vulkan driver has it
 */
static bool
ok_ubwc_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_R8G8B8X8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
   case PIPE_FORMAT_B8G8R8X8_SRGB:
      return true;
   default:
      return false;
   }
}

static int
fd5_layout_resource_for_modifier(struct fd_resource *rsc, uint64_t modifier)
{
   struct pipe_resource *prsc = &rsc->b.b;

   switch (modifier) {
   case DRM_FORMAT_MOD_QCOM_COMPRESSED: {
      uint32_t pitch = rsc->layout.pitch0;

      if (!ok_ubwc_format(prsc->format) || prsc->target != PIPE_TEXTURE_2D ||
          prsc->last_level || prsc->array_size > 1 || prsc->nr_samples > 1)
         return -1;

      rsc->layout.ubwc = true;
      rsc->layout.tile_mode = TILE5_3;
      rsc->layout.tile_all = true;
      rsc->layout.size = 0;
      fdl5_layout(&rsc->layout, prsc->format, 1, prsc->width0, prsc->height0,
                  1, 1, 1, false);

      if (fdl_pitch(&rsc->layout, 0) != pitch ||
          rsc->layout.size > fd_bo_size(rsc->bo))
         return -1;
      return 0;
   }
   case DRM_FORMAT_MOD_LINEAR:
   case DRM_FORMAT_MOD_INVALID:
      return 0;
   default:
      return -1;
   }
}

static bool
fd5_is_format_supported(struct pipe_screen *pscreen, enum pipe_format format,
                        uint64_t modifier)
{
   switch (modifier) {
   case DRM_FORMAT_MOD_LINEAR:
      return true;
   case DRM_FORMAT_MOD_QCOM_COMPRESSED:
      return ok_ubwc_format(format);
   default:
      return false;
   }
}

void
fd5_resource_screen_init(struct pipe_screen *pscreen)
{
   struct fd_screen *screen = fd_screen(pscreen);

   screen->layout_resource_for_modifier = fd5_layout_resource_for_modifier;
   screen->is_format_supported = fd5_is_format_supported;
}
