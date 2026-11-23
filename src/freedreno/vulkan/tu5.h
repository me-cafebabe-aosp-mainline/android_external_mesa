/*
 * Copyright © 2026 hiru
 * SPDX-License-Identifier: MIT
 *
 * Adreno 5xx. These Adrenos have no bindless descriptors, so the state the
 * shaders use is gathered from the descriptor sets and written out for every
 * draw, and the register layout is the one the gallium driver programs
 * (a5xx/fd5_*.c).
 */

#ifndef TU5_H
#define TU5_H

#include "tu_common.h"

/* The dword of the texture descriptors with flags of Adreno 5xx, which
 * reads the first six only
 */
/* The layers of the table of the targets the primary command buffers write
 * for the secondary ones (tu6_global::a5xx_view_bases)
 */
#define TU5_MAX_TABLE_LAYERS 16

#define TU5_TEX_FLAGS 15
/* A 4444 format with alpha in the low bits: the built-in border colors, of
 * which the packed 16-bit ones are decoded as texels of the format, are
 * those of device->a5xx_bcolor_alpha_low.
 */
#define TU5_TEX_FLAG_ALPHA_LOW (1u << 0)

/* The descriptors of a combined image sampler with a conversion of a
 * multi-planar format, in slots of 16 dwords: the first plane, the sampler,
 * then the other planes.
 */
#define TU5_YCBCR_SLOTS 4
#define TU5_YCBCR_PLANE_SLOT(p) ((p) ? (p) + 1 : 0)

/* The cache flushes asked for, tu_cmd_flush_bits */
void tu5_emit_flushes(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                      uint32_t flushes);

/* The static registers, at the start of a command buffer */
void tu5_init_hw(struct tu_cmd_buffer *cmd, struct tu_cs *cs);

/* The attachments of the subpass that starts, in the draw stream */
void tu5_emit_subpass_begin(struct tu_cmd_buffer *cmd);

/* The render pass: its draw stream, in system memory */
void tu5_cmd_render(struct tu_cmd_buffer *cmd);

/* vkCmdClearColorImage() and vkCmdClearDepthStencilImage() */
void tu5_clear_image(struct tu_cmd_buffer *cmd, struct tu_image *image,
                     const VkClearValue *value, uint32_t range_count,
                     const VkImageSubresourceRange *ranges);

/* The formats: tu5_format.c */
#ifdef __cplusplus
extern "C" {
#endif
bool tu5_format_vtx_supported(enum pipe_format format);
bool tu5_format_texture_supported(enum pipe_format format);
bool tu5_format_color_supported(enum pipe_format format);
bool tu5_format_depth_supported(enum pipe_format format);
bool tu5_format_tileable(enum pipe_format format);
enum pipe_format tu5_storage_format(enum pipe_format format);
#ifdef __cplusplus
}
#endif

/* The border colours */
void tu5_pack_border_color(void *entry, const VkClearColorValue *val,
                           bool is_int);
void tu5_pack_border_color_format(void *entry, const VkClearColorValue *val,
                                  bool is_int, VkFormat format);

/* GMEM */
bool tu5_use_gmem(struct tu_cmd_buffer *cmd);

/* Binning: the visibility streams of the pipes the tiles are grouped in */
#define TU5_VSC_PIPES 16
#define TU5_VSC_PIPE_SIZE 0x20000
bool tu5_use_binning(struct tu_cmd_buffer *cmd);

/* Multiview */
uint32_t tu5_views(const struct tu_cmd_buffer *cmd);
void tu5_bind_view(struct tu_cmd_buffer *cmd, struct tu_cs *cs, uint32_t view);
void tu5_write_view_bases(struct tu_cmd_buffer *cmd, struct tu_cs *cs);

/* The primitive restart index of the index type bound */
void tu5_emit_restart_index(struct tu_cs *cs, uint32_t restart_index);

/* Queries */
void tu5_emit_sample_count_addr(struct tu_cs *cs, uint64_t iova);
uint32_t tu5_always_on_counter_reg(void);

/* vkCmdDrawIndirect(), vkCmdDrawIndexedIndirect() */
void tu5_draw_indirect(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                       uint32_t initiator, bool indexed, uint64_t iova,
                       uint32_t count, uint32_t stride);

/* vkCmdDispatch(), vkCmdDispatchIndirect(): indirect_iova is 0 for the first */
void tu5_dispatch(struct tu_cmd_buffer *cmd, const uint32_t *blocks,
                  const uint32_t *offsets, uint64_t indirect_iova);

/* The descriptor of an image view */
void tu5_image_view_descriptor(uint32_t *descriptor,
                               const struct tu_image *image, unsigned plane,
                               enum pipe_format format,
                               const VkImageSubresourceRange *range,
                               VkImageViewType type, const unsigned char *swiz);

void tu5_buffer_view_descriptor(uint32_t *descriptor, enum pipe_format format,
                                uint64_t iova, uint32_t range);
void tu5_image_view_storage_descriptor(uint32_t *descriptor,
                                       const struct tu_image *image,
                                       unsigned plane, enum pipe_format format,
                                       const VkImageSubresourceRange *range,
                                       VkImageViewType type);

/* Copies and blits */
void tu5_copy_buffer_to_image(struct tu_cmd_buffer *cmd, struct tu_buffer *src,
                              struct tu_image *dst,
                              const VkBufferImageCopy2 *info);
void tu5_copy_image_to_buffer(struct tu_cmd_buffer *cmd, struct tu_image *src,
                              struct tu_buffer *dst,
                              const VkBufferImageCopy2 *info);
void tu5_copy_image(struct tu_cmd_buffer *cmd, struct tu_image *src,
                    struct tu_image *dst, const VkImageCopy2 *info);
void tu5_blit_image(struct tu_cmd_buffer *cmd, struct tu_image *src,
                    struct tu_image *dst, const VkImageBlit2 *info,
                    VkFilter filter);
void tu5_resolve_image(struct tu_cmd_buffer *cmd, struct tu_image *src,
                       struct tu_image *dst, const VkImageResolve2 *info);
void tu5_resolve_sysmem(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                        const struct tu_image_view *src,
                        const struct tu_image_view *dst, uint32_t layer_mask,
                        uint32_t layers, const VkRect2D *rect);
void tu5_copy_buffer(struct tu_cmd_buffer *cmd, uint64_t dst_va,
                     uint64_t src_va, uint64_t size);
void tu5_fill_buffer(struct tu_cmd_buffer *cmd, uint64_t dst_va, uint64_t size,
                     uint32_t data);

/* vkCmdClearAttachments() */
void tu5_clear_attachments(struct tu_cmd_buffer *cmd,
                           uint32_t attachment_count,
                           const VkClearAttachment *attachments,
                           uint32_t rect_count, const VkClearRect *rects);

/* All the state of a draw */
VkResult tu5_draw_state(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                        bool indexed);

#endif /* TU5_H */
