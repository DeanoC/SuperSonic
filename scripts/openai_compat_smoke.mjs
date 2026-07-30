import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";

const requireFromCwd = createRequire(`${process.cwd()}/`);
const { default: OpenAI } = await import(requireFromCwd.resolve("openai"));

const configuredURL =
  process.env.SUPERSONIC_BASE_URL ?? "http://127.0.0.1:8080";
const rootURL = configuredURL.replace(/\/+$/, "").replace(/\/v1$/, "");
const baseURL = `${rootURL}/v1`;
const apiKey = process.env.SUPERSONIC_API_KEY ?? "secret";
const model = process.env.SUPERSONIC_SMOKE_MODEL ?? "local";
const timeout = Number(process.env.SUPERSONIC_REQUEST_TIMEOUT_MS ?? "120000");
const marker = "SUPERSONIC_SMOKE_JSON=";

const client = new OpenAI({ baseURL, apiKey, timeout });

function assert(condition, message, raw) {
  if (!condition) {
    const error = new Error(message);
    if (raw !== undefined) error.raw = raw;
    throw error;
  }
}

function compact(value) {
  return JSON.stringify(value);
}

function visibleText(message) {
  return [message?.content, message?.reasoning_content]
    .filter((value) => typeof value === "string")
    .join("");
}

function chatAssistantResult(message) {
  return (
    visibleText(message).trim().length > 0 ||
    (Array.isArray(message?.tool_calls) && message.tool_calls.length > 0)
  );
}

function responseAssistantResult(response) {
  return response?.output?.some((item) => {
    if (item.type === "function_call") return true;
    if (item.type === "reasoning") {
      return Array.isArray(item.summary) && item.summary.join("").trim().length > 0;
    }
    if (item.type !== "message") return false;
    return item.content?.some(
      (part) =>
        part.type === "output_text" &&
        typeof part.text === "string" &&
        part.text.trim().length > 0,
    );
  });
}

function validateUsage(usage, label) {
  assert(usage && typeof usage === "object", `${label} usage is missing`, usage);
  const promptTokens = usage.prompt_tokens ?? usage.input_tokens;
  const completionTokens = usage.completion_tokens ?? usage.output_tokens;
  const totalTokens = usage.total_tokens;
  for (const [field, value] of Object.entries({
    promptTokens,
    completionTokens,
    totalTokens,
  })) {
    assert(
      Number.isInteger(value) && value >= 0,
      `${label} ${field} must be a non-negative integer`,
      usage,
    );
  }
  assert(promptTokens > 0, `${label} prompt token count must be positive`, usage);
  assert(
    completionTokens > 0,
    `${label} completion token count must be positive`,
    usage,
  );
  assert(
    totalTokens === promptTokens + completionTokens,
    `${label} total token count does not add up`,
    usage,
  );
  return { promptTokens, completionTokens, totalTokens };
}

async function fetchChecked(path, options = {}) {
  const response = await fetch(`${rootURL}${path}`, {
    ...options,
    headers: {
      authorization: `Bearer ${apiKey}`,
      ...(options.body ? { "content-type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  assert(response.ok, `${path} returned HTTP ${response.status}`, body);
  return body;
}

async function main() {
  const unauthorized = await fetch(`${baseURL}/models`);
  const unauthorizedBody = await unauthorized.json();
  assert(unauthorized.status === 401, "missing auth must return HTTP 401", {
    status: unauthorized.status,
    body: unauthorizedBody,
  });
  assert(
    unauthorizedBody?.error?.type === "authentication_error",
    "missing auth must use the OpenAI authentication error envelope",
    unauthorizedBody,
  );

  const models = await client.models.list();
  assert(
    models.data.some((entry) => entry.id === model),
    `model list did not include ${model}`,
    models,
  );
  const retrieved = await client.models.retrieve(model);
  assert(retrieved.id === model, "model retrieve returned the wrong id", retrieved);

  const tokenText = "hello world";
  const tokenize = await fetchChecked("/v1/tokenize", {
    method: "POST",
    body: JSON.stringify({
      model,
      input: tokenText,
      add_special_tokens: false,
    }),
  });
  assert(
    Array.isArray(tokenize.tokens) && tokenize.tokens.length > 0,
    "tokenize returned no tokens",
    tokenize,
  );
  const detokenize = await fetchChecked("/v1/detokenize", {
    method: "POST",
    body: JSON.stringify({
      model,
      tokens: tokenize.tokens,
      skip_special_tokens: true,
    }),
  });
  assert(
    typeof detokenize.text === "string" && detokenize.text.trim().length > 0,
    "detokenize returned no text",
    detokenize,
  );

  const chat = await client.chat.completions.create({
    model,
    messages: [
      { role: "developer", content: "Answer briefly." },
      { role: "user", content: "Reply with the single word hello." },
    ],
    max_completion_tokens: 8,
    temperature: 0,
  });
  const chatMessage = chat.choices?.[0]?.message;
  assert(chatAssistantResult(chatMessage), "Chat returned no assistant result", chat);
  const chatUsage = validateUsage(chat.usage, "Chat");

  const streamStarted = performance.now();
  const chatStream = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: "Reply with the single word hello." }],
    max_completion_tokens: 8,
    temperature: 0,
    stream: true,
    stream_options: { include_usage: true },
  });
  let firstTokenSeconds;
  let sawChatDelta = false;
  let sawChatTerminal = false;
  let chatStreamUsage;
  for await (const chunk of chatStream) {
    const choice = chunk.choices?.[0];
    const delta = choice?.delta;
    const substantive =
      (typeof delta?.content === "string" && delta.content.length > 0) ||
      (typeof delta?.reasoning_content === "string" &&
        delta.reasoning_content.length > 0) ||
      (Array.isArray(delta?.tool_calls) && delta.tool_calls.length > 0);
    if (substantive) {
      sawChatDelta = true;
      if (firstTokenSeconds === undefined) {
        firstTokenSeconds = (performance.now() - streamStarted) / 1000;
      }
    }
    if (choice?.finish_reason) sawChatTerminal = true;
    if (chunk.usage) chatStreamUsage = chunk.usage;
  }
  const streamSeconds = (performance.now() - streamStarted) / 1000;
  assert(sawChatDelta, "Chat stream emitted no content, reasoning, or tool delta");
  assert(sawChatTerminal, "Chat stream emitted no terminal finish_reason");
  assert(chatStreamUsage, "Chat stream emitted no terminal usage chunk");
  const normalizedChatStreamUsage = validateUsage(chatStreamUsage, "Chat stream");

  const completion = await client.completions.create({
    model,
    prompt: "Reply with hello.",
    max_tokens: 4,
    temperature: 0,
  });
  assert(
    typeof completion.choices?.[0]?.text === "string" &&
      completion.choices[0].text.length > 0,
    "legacy Completion returned no text",
    completion,
  );
  validateUsage(completion.usage, "Completion");

  const response = await client.responses.create({
    model,
    input: "Reply with the single word hello.",
    max_output_tokens: 8,
    temperature: 0,
  });
  assert(response.status === "completed", "Responses create did not complete", response);
  assert(
    responseAssistantResult(response),
    "Responses create returned no assistant result",
    response,
  );
  const responseUsage = validateUsage(response.usage, "Responses");
  const fetched = await client.responses.retrieve(response.id);
  assert(fetched.id === response.id, "Responses retrieve returned the wrong id", {
    created: response,
    fetched,
  });
  const deleted = await client.responses.delete(response.id);
  assert(
    deleted.id === response.id && deleted.deleted === true,
    "Responses delete did not delete the created response",
    deleted,
  );

  const responseStream = await client.responses.create({
    model,
    input: "Reply with the single word hello.",
    max_output_tokens: 8,
    temperature: 0,
    stream: true,
  });
  let sawResponseDelta = false;
  let sawResponseCompleted = false;
  let responseStreamUsage;
  for await (const event of responseStream) {
    if (
      event.type === "response.output_text.delta" &&
      typeof event.delta === "string" &&
      event.delta.length > 0
    ) {
      sawResponseDelta = true;
    }
    if (event.type === "response.completed") {
      sawResponseCompleted = true;
      responseStreamUsage = event.response?.usage;
    }
  }
  assert(sawResponseDelta, "Responses stream emitted no output_text delta");
  assert(sawResponseCompleted, "Responses stream emitted no response.completed event");
  const normalizedResponseStreamUsage = validateUsage(
    responseStreamUsage,
    "Responses stream",
  );

  const reasoning = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "user",
        content: "Think briefly about 1 + 1, then answer with only the number.",
      },
    ],
    reasoning_effort: "medium",
    max_completion_tokens: 64,
    temperature: 0,
  });
  const reasoningMessage = reasoning.choices?.[0]?.message;
  const reasoningText = reasoningMessage?.reasoning_content;
  assert(
    chatAssistantResult(reasoningMessage),
    "reasoning request returned no assistant result",
    reasoning,
  );
  const reasoningObserved =
    typeof reasoningText === "string" && reasoningText.trim().length > 0;
  const visibleThinkTags = visibleText(reasoningMessage).includes("<think>");
  assert(!visibleThinkTags, "reasoning leaked <think> tags into SDK fields", reasoning);

  const repeated = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: "Reply with the single word ready." }],
    max_completion_tokens: 8,
    temperature: 0,
  });
  assert(
    chatAssistantResult(repeated.choices?.[0]?.message),
    "repeated warm request returned no assistant result",
    repeated,
  );
  validateUsage(repeated.usage, "Repeated Chat");

  const report = {
    requests: {
      auth: { unauthorized_status: unauthorized.status },
      models: { listed: true, retrieved: true },
      tokenizer: {
        roundtrip: true,
        token_count: tokenize.tokens.length,
      },
      chat: {
        assistant_result: true,
        finish_reason: chat.choices[0].finish_reason,
      },
      chat_stream: {
        saw_delta: sawChatDelta,
        saw_terminal: sawChatTerminal,
        saw_usage: Boolean(chatStreamUsage),
      },
      completions: { assistant_result: true },
      responses: {
        assistant_result: true,
        stored_roundtrip: true,
      },
      responses_stream: {
        saw_delta: sawResponseDelta,
        saw_completed: sawResponseCompleted,
      },
      reasoning: {
        assistant_result: true,
        request_accepted: true,
        reasoning_observed: reasoningObserved,
        visible_think_tags: visibleThinkTags,
      },
      usage_accounting: {
        chat_valid: chatUsage.totalTokens > 0,
        chat_stream_valid: normalizedChatStreamUsage.totalTokens > 0,
        responses_valid: responseUsage.totalTokens > 0,
        responses_stream_valid: normalizedResponseStreamUsage.totalTokens > 0,
      },
      repeated_request: { assistant_result: true },
    },
    throughput: {
      first_token_seconds: firstTokenSeconds,
      prefill_tokens_per_second:
        normalizedChatStreamUsage.promptTokens / firstTokenSeconds,
      decode_tokens_per_second:
        normalizedChatStreamUsage.completionTokens /
        Math.max(streamSeconds - firstTokenSeconds, 0.000001),
    },
  };
  console.log("compat", compact(report));
  console.log(`${marker}${JSON.stringify(report)}`);
}

main().catch((error) => {
  console.error(error.stack ?? error);
  if (error.raw !== undefined) console.error("raw", compact(error.raw));
  process.exitCode = 1;
});
