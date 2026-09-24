/**
 * Harness request-history conversion into pi-ai's Context vocabulary.
 *
 * @module dsh-opencode-go/conversion/context
 */

import { brandString } from '@deepseek-ai/dsh-brand'
import { LlmError, offloadedImageText, requestImageHandleText } from '@deepseek-ai/dsh-llm'
import type { ContentBlock, GenerateOptions, ImageAttachmentAccessResolver, Message, RequestMessage, ToolCallId } from '@deepseek-ai/dsh-llm'
import type {
  AttachmentId,
  AttachmentStore,
  ImageAttachmentRef,
  ImageRequestTarget,
  RequestImageAttachment,
} from '@deepseek-ai/dsh-attachment'
import type { Context as PiContext, ImageContent, Message as PiMessage, TextContent, Tool as PiTool } from '@earendil-works/pi-ai'
import { toPiAssistant } from './replay.ts'
import { requestImageDimensions } from '@deepseek-ai/dsh-attachment'
import { DEFAULT_REQUEST_IMAGE_MAX_BYTES, DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET } from './config.ts'
import { projectRequestImages } from './image-offload.ts'

/** Join the text blocks of a harness message. */
function flattenText(message: RequestMessage): string {
  return message.content
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

/**
 * Pre-0.1.7 hosts carry a tool result as a `tool-result` block nested in user
 * content; 0.1.7 replaced it with its own `tool`-role message. The block type
 * is absent from the current `ContentBlock` union, so it is read structurally
 * to keep one adapter working across both host generations.
 */
interface LegacyToolResultBlock {
  toolCallId: ToolCallId
  content: readonly ContentBlock[]
  isError?: boolean
}

/** Read one block as a pre-0.1.7 nested tool result, or `undefined` when it is not one. */
function legacyToolResult(block: ContentBlock): LegacyToolResultBlock | undefined {
  return (block as { type: string }).type === 'tool-result'
    ? block as unknown as LegacyToolResultBlock
    : undefined
}

/** Every pre-0.1.7 nested tool result in one content list, in order. */
function legacyToolResults(blocks: readonly ContentBlock[]): LegacyToolResultBlock[] {
  const results: LegacyToolResultBlock[] = []
  for (const block of blocks) {
    const result = legacyToolResult(block)
    if (result) results.push(result)
  }
  return results
}

/**
 * Whether content carries an image, descending into pre-0.1.7 nested tool
 * results. The current host's `contentHasImage` only scans the top level,
 * because 0.1.7 never nests a result; this adapter still has to see one.
 */
function contentHasImageDeep(blocks: readonly ContentBlock[]): boolean {
  for (const block of blocks) {
    const result = legacyToolResult(block)
    if (result) {
      if (contentHasImageDeep(result.content)) return true
      continue
    }
    if (block.type === 'image') return true
  }
  return false
}

/** Flatten text inside one tool result, descending into nested results on pre-0.1.7 hosts. */
function toolResultText(blocks: readonly ContentBlock[]): string {
  return blocks.map(block => {
    const result = legacyToolResult(block)
    if (result) return toolResultText(result.content)
    return block.type === 'text' ? block.text : ''
  }).join('')
}

interface ToolMessage {
  toolCallId: ToolCallId
  content: readonly ContentBlock[]
  isError?: boolean
}

/** DSH 0.1.7 moved tool results out of user content into their own role. */
function toolMessage(message: RequestMessage): ToolMessage | undefined {
  if ((message as { role: string }).role !== 'tool') return undefined
  return message as unknown as ToolMessage
}

/** Reject unsupported roles, tool-change blocks, and image roles before replay or image offloading. */
function assertSupportedHistory(messages: readonly RequestMessage[]): void {
  for (const message of messages) {
    // Developer history is persisted for V4; provider serialization is intentionally deferred.
    if (message.role === 'developer') throw new LlmError('Developer messages are not supported yet', 'UNSUPPORTED_CONTENT')
    if (message.content.some(block => block.type === 'tool-addition' || block.type === 'tool-removal')) {
      throw new LlmError('Tool-change blocks require developer role', 'UNSUPPORTED_CONTENT')
    }
    if (message.role !== 'user' && !toolMessage(message) && contentHasImageDeep(message.content)) {
      throw new LlmError(
        `pi-ai cannot represent an image in an in-history ${message.role} message`,
        'UNSUPPORTED_CONTENT',
      )
    }
  }
}

async function userContent(
  blocks: readonly ContentBlock[],
  requestImages: ReadonlyMap<AttachmentId, RequestImageAttachment>,
  resolveImageAccess: ImageAttachmentAccessResolver,
): Promise<string | (TextContent | ImageContent)[]> {
  const content: (TextContent | ImageContent)[] = []
  for (const block of blocks) {
    const legacy = legacyToolResult(block)
    if (legacy) {
      const nested = await userContent(legacy.content, requestImages, resolveImageAccess)
      if (typeof nested === 'string') {
        if (nested.length > 0) content.push({ type: 'text', text: nested })
      } else {
        content.push(...nested)
      }
      continue
    }
    switch (block.type) {
      case 'text':
        if (block.text.length > 0) content.push({ type: 'text', text: block.text })
        break
      case 'image': {
        if (block.offloaded === true) {
          // The current host only replaces top-level offloaded images; a nested
          // one carried over from a pre-0.1.7 log reaches the adapter intact.
          content.push({ type: 'text', text: offloadedImageText(block.attachment, resolveImageAccess(block.attachment)) })
          break
        }
        const version = requestImages.get(block.attachment.attachmentId) as RequestImageAttachment
        content.push({
          type: 'text',
          text: requestImageHandleText(block.attachment, version, resolveImageAccess(block.attachment)),
        })
        content.push({
          type: 'image',
          data: Buffer.from(version.data).toString('base64'),
          mimeType: version.mediaType,
        })
        break
      }
      default:
        // Other merge-extensible blocks are not user-input vocabulary for pi-ai.
        break
    }
  }
  if (content.every(block => block.type === 'text')) return content.map(block => block.text).join('')
  return content
}

function collectImageRefs(
  blocks: readonly ContentBlock[],
  refs: Map<AttachmentId, ImageAttachmentRef>,
): void {
  for (const block of blocks) {
    const legacy = legacyToolResult(block)
    if (legacy) {
      collectImageRefs(legacy.content, refs)
      continue
    }
    if (block.type === 'image' && block.offloaded !== true) refs.set(block.attachment.attachmentId, block.attachment)
  }
}

async function prepareRequestImages(
  messages: readonly RequestMessage[],
  attachments: AttachmentStore,
  budget: PiImageRequestBudget,
  signal?: AbortSignal,
): Promise<Map<AttachmentId, RequestImageAttachment>> {
  const refs = new Map<AttachmentId, ImageAttachmentRef>()
  for (const message of messages) collectImageRefs(message.content, refs)
  const orderedRefs = [...refs.values()]
  const prepared = await Promise.all(orderedRefs.map(
    ref => attachments.readImageRequest(ref, requestImageTarget(ref, budget), signal),
  ))
  const versions = new Map<AttachmentId, RequestImageAttachment>()
  for (const [index, ref] of orderedRefs.entries()) {
    versions.set(ref.attachmentId, prepared[index] as RequestImageAttachment)
  }
  return versions
}

function toolsOf(options: GenerateOptions): PiTool[] | undefined {
  // Deferred definitions are persisted for V4; provider loading is intentionally deferred.
  if (options.tools?.some(tool => tool.deferLoading === true)) {
    throw new LlmError('Deferred tool loading is not supported yet', 'UNSUPPORTED_CONTENT')
  }
  return options.tools?.map(tool => ({
    name: tool.name,
    description: tool.description,
    // ToolSchema.parameters is a JSON Schema object; pi-ai's TSchema
    // (TypeBox) is structurally JSON Schema, so it assigns directly.
    parameters: tool.parameters,
  }))
}

/** The request split into pi-ai's single `systemPrompt` slot and the history that converts to `messages`. */
interface SystemPromptSplit {
  /** Text for pi-ai's `systemPrompt`; `undefined` sends no system prompt. */
  systemPrompt: string | undefined
  /** History messages that convert to pi-ai `messages`. */
  messages: readonly RequestMessage[]
}

/**
 * Select the pi-ai `systemPrompt` source shared by both conversion paths.
 * `options.system` wins when defined and every history message converts,
 * including a leading `system` message, which then folds into a `user`
 * message. Otherwise a leading `system` history message supplies the prompt
 * and leaves the converted history; empty leading text sends no prompt.
 */
function splitSystemPrompt(options: GenerateOptions): SystemPromptSplit {
  if (options.system !== undefined) return { systemPrompt: options.system, messages: options.messages }
  const [first, ...rest] = options.messages
  if (first?.role !== 'system') return { systemPrompt: undefined, messages: options.messages }
  const text = flattenText(first)
  return { systemPrompt: text.length > 0 ? text : undefined, messages: rest }
}

/** Assemble the request-level pi-ai context envelope shared by both conversion paths. */
function piContext(systemPrompt: string | undefined, options: GenerateOptions, messages: PiMessage[]): PiContext {
  const tools = toolsOf(options)
  return {
    ...systemPrompt !== undefined ? { systemPrompt } : {},
    messages,
    ...tools !== undefined && tools.length > 0 ? { tools } : {},
  }
}

function appendAssistant(
  message: Message,
  messages: PiMessage[],
  toolNames: Map<ToolCallId, string>,
  onReplayDegrade?: (reason: string) => void,
): void {
  const assistant = toPiAssistant(message, onReplayDegrade)
  for (const block of assistant.content) {
    if (block.type === 'toolCall') toolNames.set(brandString<ToolCallId>(block.id), block.name)
  }
  messages.push(assistant)
}

function textOnlyContext(options: GenerateOptions, onReplayDegrade?: (reason: string) => void): PiContext {
  assertSupportedHistory(options.messages)
  const split = splitSystemPrompt(options)
  const toolNames = new Map<ToolCallId, string>()
  const messages: PiMessage[] = []
  for (const message of split.messages) {
    if (contentHasImageDeep(message.content)) {
      throw new LlmError('pi-ai image conversion requires the durable attachment service', 'UNSUPPORTED_CONTENT')
    }
    const tool = toolMessage(message)
    if (tool) {
      messages.push({
        role: 'toolResult', toolCallId: tool.toolCallId,
        toolName: toolNames.get(tool.toolCallId) ?? 'unknown',
        content: [{ type: 'text', text: toolResultText(tool.content) || '(no output)' }],
        isError: tool.isError ?? false, timestamp: 0,
      })
      continue
    }
    if (message.role === 'system') {
      // pi-ai has a single systemPrompt slot; a system message that did not
      // supply it folds into a user message to preserve order.
      messages.push({ role: 'user', content: flattenText(message), timestamp: 0 })
      continue
    }
    if (message.role === 'assistant') {
      appendAssistant(message, messages, toolNames, onReplayDegrade)
      continue
    }
    const text = flattenText(message)
    // 0.1.7 carries each tool result as its own `tool`-role message; on older
    // hosts the user message also holds nested `tool-result` blocks.
    const results = legacyToolResults(message.content)
    if (text.length > 0 || results.length === 0) messages.push({ role: 'user', content: text, timestamp: 0 })
    for (const result of results) {
      messages.push({
        role: 'toolResult',
        toolCallId: result.toolCallId,
        toolName: toolNames.get(result.toolCallId) ?? 'unknown',
        content: [{
          type: 'text',
          text: toolResultText(result.content) || '(no output)',
        }],
        isError: result.isError ?? false,
        timestamp: 0,
      })
    }
  }
  return piContext(split.systemPrompt, options, messages)
}

/** Inputs that bind deterministic request images to one current tool execution world. */
export interface PiImageRequestContext {
  /** Durable provider that resolves request-image bytes and provider-owned host objects. */
  attachments: AttachmentStore
  /** Resolve current tool access separately from deterministic request-image versions. */
  resolveImageAccess: ImageAttachmentAccessResolver
  /** Request-level bound on the base64-encoded payload of retained images; omission leaves the bound unchecked. */
  maxRequestImageBytes?: number
  /** Route pixel and raw encoded-byte budgets. */
  requestImagePolicy?: PiImageRequestBudget
}

/** Per-route budgets from which each request image's target is derived. */
export interface PiImageRequestBudget {
  /** Total-pixel budget; larger sources are downscaled proportionally. */
  maxPixels: number
  /** Encoded-byte target for one request image. */
  maxBytes: number
}

/** Deterministic request target for one source under the route budgets. */
function requestImageTarget(ref: ImageAttachmentRef, budget: PiImageRequestBudget): ImageRequestTarget & PiImageRequestBudget {
  // DSH 0.1.5 reads the pixel policy; 0.1.6 reads explicit target dimensions.
  // Supply both contracts so each host retains its own image preparation path.
  return {
    ...requestImageDimensions(ref.width, ref.height, budget.maxPixels),
    maxPixels: budget.maxPixels,
    maxBytes: budget.maxBytes,
  }
}

/**
 * Convert text-only harness history to a synchronous pi-ai Context. Tool
 * result names are recovered from preceding assistant tool calls.
 * @param options - the harness request; `options.system`, else a leading `system` message, maps to pi-ai's single `systemPrompt` slot.
 * @param images - absent; selects the synchronous conversion.
 * @param onReplayDegrade - forwarded to {@link toPiAssistant} for each assistant message.
 * @returns the pi-ai context; `tools` is omitted when the request declares none.
 * @throws {LlmError} `UNSUPPORTED_CONTENT` for images in any history role, including a leading system message.
 */
export function toPiContext(
  options: GenerateOptions,
  images?: undefined,
  onReplayDegrade?: (reason: string) => void,
): PiContext
/**
 * Convert harness history to a pi-ai Context while resolving durable images.
 * Tool result names are recovered from preceding assistant tool calls. On DSH
 * 0.1.5, oldest images over the request budget become transient placeholders.
 * On newer hosts, occurrences the surface marks offloaded become placeholders; when the
 * retained occurrences' exact base64 payload still exceeds
 * `maxRequestImageBytes`, the call fails with `IMAGE_OFFLOAD_REQUIRED` naming
 * how many more oldest occurrences must be offloaded.
 * @param options - the harness request; `options.system`, else a leading `system` message, maps to pi-ai's single `systemPrompt` slot.
 * @param images - attachment provider, current path resolver, and request limits.
 * @param onReplayDegrade - forwarded to {@link toPiAssistant} for each assistant message.
 * @returns the asynchronously resolved pi-ai context.
 */
export function toPiContext(
  options: GenerateOptions,
  images: PiImageRequestContext,
  onReplayDegrade?: (reason: string) => void,
): Promise<PiContext>
export function toPiContext(
  options: GenerateOptions,
  images?: PiImageRequestContext,
  onReplayDegrade?: (reason: string) => void,
): PiContext | Promise<PiContext> {
  return images === undefined
    ? textOnlyContext(options, onReplayDegrade)
    : toPiContextWithImages(options, images, onReplayDegrade)
}

async function toPiContextWithImages(
  options: GenerateOptions,
  images: PiImageRequestContext,
  onReplayDegrade?: (reason: string) => void,
): Promise<PiContext> {
  const { attachments, resolveImageAccess, maxRequestImageBytes } = images
  const requestImagePolicy = images.requestImagePolicy ?? {
    maxPixels: DEFAULT_REQUEST_IMAGE_PIXEL_BUDGET,
    maxBytes: DEFAULT_REQUEST_IMAGE_MAX_BYTES,
  }
  assertSupportedHistory(options.messages)
  const split = splitSystemPrompt(options)
  const projection = {
    maxBytes: maxRequestImageBytes,
    placeholder: (ref: ImageAttachmentRef) => offloadedImageText(ref, resolveImageAccess(ref)),
  }
  const requestMessages = projectRequestImages(split.messages, {
    ...projection, exact: false,
    byteLength: ref => Math.min(ref.bytes, requestImagePolicy.maxBytes),
  })
  const requestImages = await prepareRequestImages(requestMessages, attachments, requestImagePolicy, options.signal)
  const exactMessages = projectRequestImages(requestMessages, {
    ...projection, exact: true,
    byteLength: ref => (requestImages.get(ref.attachmentId) as RequestImageAttachment).bytes,
  })
  const toolNames = new Map<ToolCallId, string>()
  const messages: PiMessage[] = []

  for (const message of exactMessages) {
    const tool = toolMessage(message)
    if (tool) {
      const content = await userContent(tool.content, requestImages, resolveImageAccess)
      messages.push({
        role: 'toolResult', toolCallId: tool.toolCallId,
        toolName: toolNames.get(tool.toolCallId) ?? 'unknown',
        content: typeof content === 'string' ? [{ type: 'text', text: content || '(no output)' }] : content,
        isError: tool.isError ?? false, timestamp: 0,
      })
      continue
    }
    if (message.role === 'system') {
      // pi-ai has a single systemPrompt slot; a system message that did not
      // supply it folds into a user message to preserve order.
      messages.push({ role: 'user', content: flattenText(message), timestamp: 0 })
      continue
    }
    if (message.role === 'assistant') {
      appendAssistant(message, messages, toolNames, onReplayDegrade)
      continue
    }
    // user role: text and images, plus nested `tool-result` blocks on older
    // hosts (each result becomes its own `toolResult` message).
    const regular = message.content.filter(block => legacyToolResult(block) === undefined)
    const content = await userContent(regular, requestImages, resolveImageAccess)
    const results = legacyToolResults(message.content)
    if (content.length > 0 || results.length === 0) {
      messages.push({ role: 'user', content, timestamp: 0 })
    }
    for (const result of results) {
      const resultContent = await userContent(result.content, requestImages, resolveImageAccess)
      messages.push({
        role: 'toolResult',
        toolCallId: result.toolCallId,
        toolName: toolNames.get(result.toolCallId) ?? 'unknown',
        content: typeof resultContent === 'string'
          ? [{ type: 'text', text: resultContent || '(no output)' }]
          : resultContent,
        isError: result.isError ?? false,
        timestamp: 0,
      })
    }
  }

  return piContext(split.systemPrompt, options, messages)
}
