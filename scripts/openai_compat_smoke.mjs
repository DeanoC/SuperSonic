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

function normalized(value) {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function exactCanary(actual, finishReason, expected = "hello") {
  return {
    expected,
    actual: normalized(actual),
    finish_reason: finishReason,
    passed: normalized(actual) === expected && finishReason === "stop",
  };
}

function responseText(response) {
  return (response?.output ?? [])
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content ?? [])
    .filter((part) => part.type === "output_text")
    .map((part) => part.text ?? "")
    .join("");
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
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
  };
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function fetchChecked(path, options = {}) {
  const response = await fetchWithTimeout(`${rootURL}${path}`, {
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

async function unauthorizedEvidence(path, authorization) {
  const response = await fetchWithTimeout(`${rootURL}${path}`, {
    headers: authorization ? { authorization } : {},
  });
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  assert(response.status === 401, `${path} must reject invalid auth`, {
    status: response.status,
    body,
  });
  assert(
    body?.error?.type === "authentication_error",
    `${path} must use the authentication error envelope`,
    body,
  );
  return { status: response.status, error_type: body.error.type };
}

async function sdkWrongKeyEvidence() {
  const wrongKeyClient = new OpenAI({
    baseURL,
    apiKey: "definitely-wrong",
    timeout,
  });
  try {
    await wrongKeyClient.models.list();
    assert(false, "SDK model route accepted the wrong API key");
  } catch (error) {
    assert(error?.status === 401, "SDK wrong-key error was not HTTP 401", error);
    assert(
      error?.error?.type === "authentication_error",
      "SDK wrong-key error did not preserve the authentication envelope",
      error,
    );
    return { status: error.status, error_type: error.error.type };
  }
}

async function main() {
  const missingKey = await unauthorizedEvidence("/v1/models");
  const wrongKey = await sdkWrongKeyEvidence();
  const protectedRoutes = {};
  for (const path of ["/health", "/ready", "/metrics", "/v1/capabilities"]) {
    protectedRoutes[path] = await unauthorizedEvidence(
      path,
      "Bearer definitely-wrong",
    );
  }

  const models = await client.models.list();
  assert(
    models.data.some((entry) => entry.id === model),
    `model list did not include ${model}`,
    models,
  );
  const retrieved = await client.models.retrieve(model);
  assert(retrieved.id === model, "model retrieve returned the wrong id", retrieved);

  const tokenize = await fetchChecked("/v1/tokenize", {
    method: "POST",
    body: JSON.stringify({
      model,
      input: "hello world",
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
      { role: "developer", content: "Output exactly: hello" },
      { role: "user", content: "Output exactly: hello" },
    ],
    max_completion_tokens: 8,
    temperature: 0,
  });
  const chatChoice = chat.choices?.[0];
  assert(chatChoice?.message, "Chat returned no choice", chat);
  const chatUsage = validateUsage(chat.usage, "Chat");
  const chatSemantic = exactCanary(
    chatChoice.message.content,
    chatChoice.finish_reason,
  );

  const streamStarted = performance.now();
  const chatStream = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: "Output exactly: hello" }],
    max_completion_tokens: 8,
    temperature: 0,
    stream: true,
    stream_options: { include_usage: true },
  });
  let firstTokenSeconds;
  let chatStreamText = "";
  let chatStreamUsage;
  const chatStreamEvents = [];
  for await (const chunk of chatStream) {
    const choice = chunk.choices?.[0];
    const delta = choice?.delta?.content;
    if (typeof delta === "string" && delta.length > 0) {
      chatStreamText += delta;
      if (firstTokenSeconds === undefined) {
        firstTokenSeconds = (performance.now() - streamStarted) / 1000;
      }
    }
    if (choice?.finish_reason !== null && choice?.finish_reason !== undefined) {
      chatStreamEvents.push({
        kind: "terminal",
        finish_reason: choice.finish_reason,
      });
    } else if (chunk.usage) {
      chatStreamEvents.push({ kind: "usage" });
    } else {
      chatStreamEvents.push({ kind: "delta" });
    }
    if (chunk.usage) chatStreamUsage = chunk.usage;
  }
  assert(firstTokenSeconds !== undefined, "Chat stream emitted no text delta");
  const streamSeconds = (performance.now() - streamStarted) / 1000;
  const chatTerminalIndexes = chatStreamEvents
    .map((event, index) => (event.kind === "terminal" ? index : -1))
    .filter((index) => index >= 0);
  const chatUsageIndexes = chatStreamEvents
    .map((event, index) => (event.kind === "usage" ? index : -1))
    .filter((index) => index >= 0);
  assert(chatStreamUsage, "Chat stream emitted no terminal usage");
  const chatStreamUsageCounts = validateUsage(chatStreamUsage, "Chat stream");
  const chatStreamSemantic = {
    ...exactCanary(
      chatStreamText,
      chatTerminalIndexes.length === 1
        ? chatStreamEvents[chatTerminalIndexes[0]].finish_reason
        : null,
    ),
    terminal_count: chatTerminalIndexes.length,
    terminal_last_before_usage:
      chatTerminalIndexes.length === 1 &&
      chatUsageIndexes.length === 1 &&
      chatTerminalIndexes[0] + 1 === chatUsageIndexes[0],
    usage_last:
      chatUsageIndexes.length === 1 &&
      chatUsageIndexes[0] === chatStreamEvents.length - 1,
  };
  chatStreamSemantic.passed =
    chatStreamSemantic.passed &&
    chatStreamSemantic.terminal_count === 1 &&
    chatStreamSemantic.terminal_last_before_usage &&
    chatStreamSemantic.usage_last;

  const completion = await client.completions.create({
    model,
    prompt: "Output exactly: hello",
    max_tokens: 8,
    temperature: 0,
  });
  const completionChoice = completion.choices?.[0];
  assert(completionChoice, "legacy Completion returned no choice", completion);
  const completionUsage = validateUsage(completion.usage, "Completion");
  const completionSemantic = exactCanary(
    completionChoice.text,
    completionChoice.finish_reason,
  );

  const response = await client.responses.create({
    model,
    input: "Output exactly: hello",
    max_output_tokens: 8,
    temperature: 0,
  });
  const responseUsage = validateUsage(response.usage, "Responses");
  const fetched = await client.responses.retrieve(response.id);
  const storedRoundtrip =
    fetched.id === response.id && responseText(fetched) === responseText(response);
  const deleted = await client.responses.delete(response.id);
  assert(
    deleted.id === response.id && deleted.deleted === true,
    "Responses delete did not delete the created response",
    deleted,
  );
  const responseSemantic = {
    expected: "hello",
    actual: normalized(responseText(response)),
    status: response.status,
    stored_roundtrip: storedRoundtrip,
    passed:
      normalized(responseText(response)) === "hello" &&
      response.status === "completed" &&
      storedRoundtrip,
  };

  const responseStream = await client.responses.create({
    model,
    input: "Output exactly: hello",
    max_output_tokens: 8,
    temperature: 0,
    stream: true,
  });
  let responseStreamText = "";
  const responseStreamEvents = [];
  let responseStreamUsage;
  let responseStreamStatus;
  for await (const event of responseStream) {
    responseStreamEvents.push(event.type);
    if (
      event.type === "response.output_text.delta" &&
      typeof event.delta === "string"
    ) {
      responseStreamText += event.delta;
    }
    if (event.type === "response.completed") {
      responseStreamUsage = event.response?.usage;
      responseStreamStatus = event.response?.status;
    }
  }
  const responseCompletedIndexes = responseStreamEvents
    .map((type, index) => (type === "response.completed" ? index : -1))
    .filter((index) => index >= 0);
  const responseStreamUsageCounts = validateUsage(
    responseStreamUsage,
    "Responses stream",
  );
  const responseStreamSemantic = {
    expected: "hello",
    actual: normalized(responseStreamText),
    status: responseStreamStatus,
    terminal_count: responseCompletedIndexes.length,
    terminal_last:
      responseCompletedIndexes.length === 1 &&
      responseCompletedIndexes[0] === responseStreamEvents.length - 1,
    passed:
      normalized(responseStreamText) === "hello" &&
      responseStreamStatus === "completed" &&
      responseCompletedIndexes.length === 1 &&
      responseCompletedIndexes[0] === responseStreamEvents.length - 1,
  };

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
  const reasoningAccepted = Boolean(reasoningMessage);
  const reasoningObserved =
    typeof reasoningMessage?.reasoning_content === "string" &&
    reasoningMessage.reasoning_content.trim().length > 0;
  const visibleThinkTags =
    typeof reasoningMessage?.content === "string" &&
    reasoningMessage.content.includes("<think>");
  const reasoningSemantic = {
    accepted: reasoningAccepted,
    observed: reasoningObserved,
    visible_think_tags: visibleThinkTags,
    passed: reasoningAccepted && reasoningObserved && !visibleThinkTags,
  };

  const repeated = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: "Reply with the single word ready." }],
    max_completion_tokens: 8,
    temperature: 0,
  });
  const repeatedChoice = repeated.choices?.[0];
  assert(repeatedChoice, "repeated Chat returned no choice", repeated);
  const repeatedUsage = validateUsage(repeated.usage, "Repeated Chat");
  const repeatedSemantic = exactCanary(
    repeatedChoice.message?.content,
    repeatedChoice.finish_reason,
    "ready",
  );

  const semantics = {
    chat: chatSemantic,
    chat_stream: chatStreamSemantic,
    completions: completionSemantic,
    responses: responseSemantic,
    responses_stream: responseStreamSemantic,
    reasoning: reasoningSemantic,
    repeated_request: repeatedSemantic,
  };
  semantics.passed = Object.values(semantics).every(
    (section) => section && section.passed === true,
  );

  const report = {
    transport: {
      auth: {
        missing_key: missingKey,
        wrong_key: wrongKey,
        protected_routes: protectedRoutes,
      },
      models: { listed: true, retrieved: true },
      tokenizer: { roundtrip: true, token_count: tokenize.tokens.length },
      chat: { received: true },
      chat_stream: {
        received_delta: chatStreamText.length > 0,
        received_terminal: chatTerminalIndexes.length > 0,
        received_usage: Boolean(chatStreamUsage),
      },
      completions: { received: true },
      responses: { received: true, stored_roundtrip: storedRoundtrip },
      responses_stream: {
        received_delta: responseStreamText.length > 0,
        received_terminal: responseCompletedIndexes.length > 0,
        received_usage: Boolean(responseStreamUsage),
      },
      reasoning: { request_accepted: reasoningAccepted },
      repeated_request: { received: true },
    },
    semantic_quality: semantics,
    usage: {
      chat: chatUsage,
      chat_stream: chatStreamUsageCounts,
      completions: completionUsage,
      responses: responseUsage,
      responses_stream: responseStreamUsageCounts,
      repeated_request: repeatedUsage,
    },
    throughput: {
      first_token_seconds: firstTokenSeconds,
      prefill_tokens_per_second:
        chatStreamUsageCounts.prompt_tokens / firstTokenSeconds,
      decode_tokens_per_second:
        chatStreamUsageCounts.completion_tokens /
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
