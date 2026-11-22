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

/* The descriptor of an image view */
void tu5_image_view_descriptor(uint32_t *descriptor,
                               const struct tu_image *image, unsigned plane,
                               enum pipe_format format,
                               const VkImageSubresourceRange *range,
                               VkImageViewType type, const unsigned char *swiz);

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
