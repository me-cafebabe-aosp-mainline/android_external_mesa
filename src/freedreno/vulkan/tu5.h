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

/* vkCmdClearAttachments() */
void tu5_clear_attachments(struct tu_cmd_buffer *cmd,
                           uint32_t attachment_count,
                           const VkClearAttachment *attachments,
                           uint32_t rect_count, const VkClearRect *rects);

/* All the state of a draw */
VkResult tu5_draw_state(struct tu_cmd_buffer *cmd, struct tu_cs *cs,
                        bool indexed);

#endif /* TU5_H */
