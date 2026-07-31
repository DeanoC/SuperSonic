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

const functionDefinition = {
  name: "read_source_file",
  description: "Read a UTF-8 source file from the current coding workspace.",
  parameters: {
    type: "object",
    properties: {
      path: {
        type: "string",
        description: "Workspace-relative source path to read.",
      },
    },
    required: ["path"],
    additionalProperties: false,
  },
};

const chatTool = {
  type: "function",
  function: functionDefinition,
};

const responsesTool = {
  type: "function",
  ...functionDefinition,
};

const codingPrompt =
  "Your entire response must be exactly one call to read_source_file " +
  "with path src/lib.rs. Do not write natural language before or after the call.";

const toolOutput = JSON.stringify({
  path: "src/lib.rs",
  contents: "pub fn protocol_ready() -> bool { true }\n",
});

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

function parseArguments(call, raw) {
  assert(call?.function?.name === functionDefinition.name, "unexpected tool name", raw);
  assert(
    typeof call.function.arguments === "string",
    "tool arguments must be a JSON string",
    raw,
  );
  let args;
  try {
    args = JSON.parse(call.function.arguments);
  } catch {
    assert(false, "tool arguments were not valid JSON", raw);
  }
  assert(
    args &&
      typeof args === "object" &&
      !Array.isArray(args) &&
      Object.keys(args).length === 1 &&
      args.path === "src/lib.rs",
    'tool call arguments must be exactly {"path":"src/lib.rs"}',
    raw,
  );
  return args;
}

function parseResponseCall(item, raw) {
  const normalized = {
    id: item?.call_id ?? item?.id,
    type: "function",
    function: {
      name: item?.name,
      arguments: item?.arguments,
    },
  };
  const args = parseArguments(normalized, raw);
  assert(
    typeof normalized.id === "string" && normalized.id.length > 0,
    "Responses tool call did not include a call_id",
    raw,
  );
  return { call: normalized, args };
}

function chatText(message) {
  return typeof message?.content === "string" ? message.content.trim() : "";
}

function responseText(response) {
  return (response?.output ?? [])
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content ?? [])
    .filter((part) => part.type === "output_text")
    .map((part) => part.text ?? "")
    .join("")
    .trim();
}

async function boundedFetch(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    Math.min(timeout, 5000),
  );
  try {
    return await fetch(`${rootURL}${path}`, {
      ...options,
      headers: {
        authorization: `Bearer ${apiKey}`,
        ...options.headers,
      },
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timer);
  }
}

async function fetchJson(path) {
  const response = await boundedFetch(path);
  const body = await response.json();
  assert(response.ok, `${path} returned HTTP ${response.status}`, body);
  return body;
}

function metricValue(metrics, name) {
  const line = metrics
    .split("\n")
    .map((value) => value.trim())
    .find((value) => value.startsWith(`${name} `));
  assert(line, `metrics did not include ${name}`, metrics);
  const value = Number(line.split(/\s+/)[1]);
  assert(Number.isFinite(value), `${name} was not finite`, line);
  return value;
}

async function schedulerSnapshot() {
  const health = await fetchJson("/health");
  const metricsResponse = await boundedFetch("/metrics");
  const metrics = await metricsResponse.text();
  assert(metricsResponse.ok, "metrics request failed", metrics);
  const snapshot = {
    active_requests: health.active_requests,
    queued_requests: health.queued_requests,
    model_loads_total: metricValue(metrics, "supersonic_model_loads_total"),
    metric_active_requests: metricValue(metrics, "supersonic_active_requests"),
    metric_queued_requests: metricValue(metrics, "supersonic_queued_requests"),
  };
  return snapshot;
}

async function waitForScheduler(predicate, label) {
  const started = performance.now();
  const deadline = started + Math.min(timeout, 120000);
  let snapshot;
  do {
    snapshot = await schedulerSnapshot();
    if (predicate(snapshot)) {
      return {
        snapshot,
        elapsed_seconds: (performance.now() - started) / 1000,
      };
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  } while (performance.now() < deadline);
  assert(false, `scheduler did not reach ${label}`, snapshot);
}

async function awaitAbortClosure(iterator) {
  const deadline = performance.now() + Math.min(timeout, 5000);
  while (performance.now() < deadline) {
    try {
      const result = await Promise.race([
        iterator.next(),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error("abort closure poll timed out")), 500),
        ),
      ]);
      if (result.done) return true;
      const terminal = result.value?.choices?.some(
        (choice) => choice.finish_reason !== null && choice.finish_reason !== undefined,
      );
      assert(!terminal, "stream reached a natural terminal after abort", result.value);
    } catch (error) {
      const name = String(error?.name ?? "");
      const message = String(error?.message ?? "");
      if (
        name.includes("Abort") ||
        message.toLowerCase().includes("abort") ||
        message.toLowerCase().includes("terminated")
      ) {
        return true;
      }
      if (message === "abort closure poll timed out") continue;
      throw error;
    }
  }
  assert(false, "stream did not close after abort");
}

async function cancellationGate() {
  const cancellationStream = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "user",
        content: "List the integers from one to one hundred in words.",
      },
    ],
    max_completion_tokens: 128,
    temperature: 0,
    stream: true,
  });
  const iterator = cancellationStream[Symbol.asyncIterator]();
  let sawDelta = false;
  let terminalBeforeAbort = false;
  while (!sawDelta) {
    const { value: chunk, done } = await iterator.next();
    assert(!done, "cancellation stream closed before a substantive delta");
    terminalBeforeAbort = chunk.choices?.some(
      (choice) => choice.finish_reason !== null && choice.finish_reason !== undefined,
    );
    assert(!terminalBeforeAbort, "cancellation stream terminated before abort", chunk);
    const delta = chunk.choices?.[0]?.delta;
    sawDelta =
      (typeof delta?.content === "string" && delta.content.length > 0) ||
      (typeof delta?.reasoning_content === "string" &&
        delta.reasoning_content.length > 0) ||
      (Array.isArray(delta?.tool_calls) && delta.tool_calls.length > 0);
  }

  const queuedRequest = client.chat.completions.create({
    model,
    messages: [{ role: "user", content: "queued cancellation probe" }],
    max_completion_tokens: 8,
    temperature: 0,
  });
  const before = await waitForScheduler(
    (snapshot) =>
      snapshot.active_requests === 1 &&
      snapshot.queued_requests === 1 &&
      snapshot.metric_active_requests === 1 &&
      snapshot.metric_queued_requests === 1 &&
      snapshot.model_loads_total === 1,
    "one active and one queued request",
  );

  cancellationStream.controller.abort();
  const abortClosed = await awaitAbortClosure(iterator);
  const queuedResult = await queuedRequest;
  assert(
    queuedResult.choices?.[0]?.message,
    "queued request did not complete after cancellation",
    queuedResult,
  );
  const after = await waitForScheduler(
    (snapshot) =>
      snapshot.active_requests === 0 &&
      snapshot.queued_requests === 0 &&
      snapshot.metric_active_requests === 0 &&
      snapshot.metric_queued_requests === 0 &&
      snapshot.model_loads_total === 1,
    "idle scheduler after cancellation",
  );

  const report = {
    nonterminal_delta: sawDelta && !terminalBeforeAbort,
    abort_closed: abortClosed,
    before: before.snapshot,
    after: after.snapshot,
    queued_request_completed: true,
    release_seconds: after.elapsed_seconds,
  };
  console.log("cancellation_release", compact(report));
  return report;
}

async function chatToolLoop() {
  const started = performance.now();
  const first = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: codingPrompt }],
    tools: [chatTool],
    tool_choice: "auto",
    max_completion_tokens: 128,
    temperature: 0,
  });
  const firstMessage = first.choices?.[0]?.message;
  const calls = firstMessage?.tool_calls ?? [];
  assert(
    Array.isArray(calls) && calls.length === 1,
    "Chat did not generate exactly one valid tool call",
    first,
  );
  assert(
    first.choices[0].finish_reason === "tool_calls",
    "Chat tool call did not finish with tool_calls",
    first,
  );
  assert(
    typeof calls[0].id === "string" && calls[0].id.length > 0,
    "Chat tool call did not include an id",
    first,
  );
  const suffix = chatText(firstMessage);
  assert(suffix === "", "Chat tool call included suffix content", first);
  const args = parseArguments(calls[0], first);

  const second = await client.chat.completions.create({
    model,
    messages: [
      { role: "user", content: codingPrompt },
      {
        role: "assistant",
        content: firstMessage.content ?? null,
        tool_calls: calls,
      },
      {
        role: "tool",
        tool_call_id: calls[0].id,
        content: toolOutput,
      },
    ],
    tools: [chatTool],
    tool_choice: "auto",
    max_completion_tokens: 64,
    temperature: 0,
  });
  const secondMessage = second.choices?.[0]?.message;
  const finalText = chatText(secondMessage);
  const finalCallCount = secondMessage?.tool_calls?.length ?? 0;
  assert(finalText.length > 0, "Chat continuation returned no assistant text", second);
  assert(
    second.choices[0].finish_reason === "stop",
    "Chat continuation did not finish with stop",
    second,
  );
  assert(finalCallCount === 0, "Chat continuation returned an unhandled call", second);

  return {
    call_count: calls.length,
    valid_tool_call: true,
    call_id: calls[0].id,
    tool_name: calls[0].function.name,
    arguments: args,
    finish_reason: first.choices[0].finish_reason,
    suffix_content: suffix,
    continuation: {
      text: finalText,
      finish_reason: second.choices[0].finish_reason,
      tool_call_count: finalCallCount,
    },
    elapsed_seconds: (performance.now() - started) / 1000,
  };
}

async function responsesToolLoop() {
  const started = performance.now();
  const first = await client.responses.create({
    model,
    input: codingPrompt,
    tools: [responsesTool],
    tool_choice: "auto",
    max_output_tokens: 128,
    temperature: 0,
  });
  const calls = (first.output ?? []).filter(
    (item) => item.type === "function_call",
  );
  assert(
    calls.length === 1,
    "Responses did not generate exactly one valid function_call",
    first,
  );
  assert(first.status === "completed", "Responses tool call did not complete", first);
  const suffix = responseText(first);
  assert(suffix === "", "Responses tool call included suffix content", first);
  const { call, args } = parseResponseCall(calls[0], first);

  const second = await client.responses.create({
    model,
    previous_response_id: first.id,
    input: [
      {
        type: "function_call_output",
        call_id: call.id,
        output: toolOutput,
      },
    ],
    tools: [responsesTool],
    tool_choice: "auto",
    max_output_tokens: 64,
    temperature: 0,
  });
  const finalText = responseText(second);
  const finalCalls = (second.output ?? []).filter(
    (item) => item.type === "function_call",
  );
  assert(
    second.status === "completed" && finalText.length > 0,
    "Responses continuation returned no terminal assistant text",
    second,
  );
  assert(
    finalCalls.length === 0,
    "Responses continuation returned an unhandled function_call",
    second,
  );

  return {
    call_count: calls.length,
    valid_tool_call: true,
    call_id: call.id,
    tool_name: call.function.name,
    arguments: args,
    status: first.status,
    suffix_content: suffix,
    continuation: {
      text: finalText,
      status: second.status,
      tool_call_count: finalCalls.length,
    },
    elapsed_seconds: (performance.now() - started) / 1000,
  };
}

async function main() {
  const report = {
    requests: {},
    cancellation: null,
  };
  let phase = "cancellation";
  try {
    report.cancellation = await cancellationGate();
    phase = "chat_tool_loop";
    report.requests.chat_tool_loop = await chatToolLoop();
    phase = "responses_tool_loop";
    report.requests.responses_tool_loop = await responsesToolLoop();
  } catch (error) {
    const partial = {
      ...report,
      failure: {
        phase,
        message: String(error?.message ?? error),
        raw: error?.raw ?? null,
      },
    };
    console.log("agent_tool_failure", compact(partial));
    console.log(`${marker}${JSON.stringify(partial)}`);
    throw error;
  }
  console.log("agent_tool", compact(report));
  console.log(`${marker}${JSON.stringify(report)}`);
}

main().catch((error) => {
  console.error(error.stack ?? error);
  if (error.raw !== undefined) console.error("raw", compact(error.raw));
  process.exitCode = 1;
});
