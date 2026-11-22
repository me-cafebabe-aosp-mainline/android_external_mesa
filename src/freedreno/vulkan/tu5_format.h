/*
 * Copyright © 2016 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * The a5xx format table of the gallium driver, for the Vulkan one.
 */

#ifndef TU5_FORMAT_H
#define TU5_FORMAT_H

#include <stdbool.h>
#include <stdint.h>

#include "util/format/u_format.h"
#include "util/half_float.h"
#include "util/u_math.h"

#include "fd5_hw.h"

#ifdef __cplusplus
extern "C" {
#endif

bool tu5_format_vtx_supported(enum pipe_format format);
bool tu5_format_texture_supported(enum pipe_format format);
bool tu5_format_color_supported(enum pipe_format format);
bool tu5_format_depth_supported(enum pipe_format format);
bool tu5_format_tileable(enum pipe_format format);

enum a5xx_vtx_fmt fd5_pipe2vtx(enum pipe_format format);
enum a5xx_tex_fmt fd5_pipe2tex(enum pipe_format format);
enum a5xx_color_fmt fd5_pipe2color(enum pipe_format format);
enum a3xx_color_swap fd5_pipe2swap(enum pipe_format format);
enum a5xx_depth_format fd5_pipe2depth(enum pipe_format format);
uint32_t fd5_tex_swiz(enum pipe_format format, unsigned swizzle_r,
                      unsigned swizzle_g, unsigned swizzle_b,
                      unsigned swizzle_a);

#ifdef __cplusplus
}
#endif

#endif /* TU5_FORMAT_H */
