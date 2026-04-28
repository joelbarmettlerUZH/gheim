/** Responses API — text-in / text-out with event-based streaming.
 * Mirrors gheim/openai/_responses.py.
 */
import type OpenAIClass from "openai";
import { Session } from "../core/session.ts";
import { StreamDeanonymizer } from "../core/stream.ts";
import type { Detector } from "../detectors/base.ts";
import { anonymizeValue, type CallExtras, resolveCallSession } from "./_base.ts";

const TEXT_DELTA_EVENTS: Record<string, string> = {
  "response.output_text.delta": "text",
  "response.reasoning_text.delta": "reasoning",
  "response.audio_transcript.delta": "audio",
  "response.function_call_arguments.delta": "function_args",
  "response.custom_tool_call_input.delta": "custom_tool_input",
};

interface ResponsesEvent {
  type?: string;
  delta?: string;
}

interface ResponsesResp {
  output?: Array<{
    content?: Array<{ text?: string }>;
  }>;
}

class ResponsesStream implements AsyncIterable<ResponsesEvent> {
  private bufs: Map<string, StreamDeanonymizer> = new Map();

  constructor(
    private readonly inner: AsyncIterable<ResponsesEvent>,
    private readonly session: Session,
  ) {}

  private bufFor(channel: string): StreamDeanonymizer {
    let buf = this.bufs.get(channel);
    if (!buf) {
      buf = new StreamDeanonymizer(this.session.mapping);
      this.bufs.set(channel, buf);
    }
    return buf;
  }

  async *[Symbol.asyncIterator](): AsyncIterator<ResponsesEvent> {
    for await (const event of this.inner) {
      if (typeof event.type === "string") {
        const channel = TEXT_DELTA_EVENTS[event.type];
        if (channel && typeof event.delta === "string") {
          event.delta = this.bufFor(channel).feed(event.delta);
        }
      }
      yield event;
    }
    // Tail flush per channel.
    for (const [channel, buf] of this.bufs) {
      const tail = buf.flush();
      if (tail) {
        yield { type: `response.${channel}.delta.tail`, delta: tail };
      }
    }
  }
}

interface RawResponses {
  create(body: unknown): Promise<unknown>;
}

export class GheimResponses {
  constructor(
    private readonly clientPromise: Promise<OpenAIClass>,
    private readonly defaultDet: Detector,
  ) {}

  async create(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { session, detector, rest } = resolveCallSession(body, this.defaultDet);
    for (const key of ["input", "instructions"]) {
      if (key in rest) {
        rest[key] = await anonymizeValue(rest[key], session, detector);
      }
    }
    // tools[*].function.description (mirror chat handling)
    const tools = rest.tools as Array<Record<string, unknown>> | undefined;
    if (tools) {
      const newTools: Array<Record<string, unknown>> = [];
      for (const tool of tools) {
        const fn = tool.function as Record<string, unknown> | undefined;
        if (fn && typeof fn.description === "string") {
          const newFn = { ...fn, description: await anonymizeValue(fn.description, session, detector) };
          newTools.push({ ...tool, function: newFn });
        } else {
          newTools.push(tool);
        }
      }
      rest.tools = newTools;
    }
    const client = await this.clientPromise;
    const response = await ((client as unknown as { responses: RawResponses }).responses.create(rest));
    if (rest.stream) {
      return new ResponsesStream(response as AsyncIterable<ResponsesEvent>, session);
    }
    const r = response as ResponsesResp;
    for (const item of r.output ?? []) {
      for (const part of item.content ?? []) {
        if (typeof part.text === "string") {
          part.text = session.restore(part.text);
        }
      }
    }
    return response;
  }
}
