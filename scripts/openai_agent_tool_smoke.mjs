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
    typeof args?.path === "string" && args.path.endsWith("src/lib.rs"),
    "tool call did not request src/lib.rs",
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

function chatTextResult(message) {
  return [message?.content, message?.reasoning_content].some(
    (value) => typeof value === "string" && value.trim().length > 0,
  );
}

function responseTextResult(response) {
  return response?.output?.some((item) => {
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

async function fetchJson(path) {
  const response = await fetch(`${rootURL}${path}`, {
    headers: { authorization: `Bearer ${apiKey}` },
  });
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

async function schedulerRelease() {
  const started = performance.now();
  const deadline = started + Math.min(timeout, 120000);
  let health;
  do {
    health = await fetchJson("/health");
    if (health.active_requests === 0 && health.queued_requests === 0) {
      const metricsResponse = await fetch(`${rootURL}/metrics`, {
        headers: { authorization: `Bearer ${apiKey}` },
      });
      const metrics = await metricsResponse.text();
      assert(metricsResponse.ok, "metrics failed after stream abort", metrics);
      const active = metricValue(metrics, "supersonic_active_requests");
      const queued = metricValue(metrics, "supersonic_queued_requests");
      const loads = metricValue(metrics, "supersonic_model_loads_total");
      assert(active === 0 && queued === 0, "metrics retained scheduler work", metrics);
      assert(loads === 1, "stream abort changed the model load count", metrics);
      return {
        scheduler_released: true,
        active_requests: health.active_requests,
        queued_requests: health.queued_requests,
        release_seconds: (performance.now() - started) / 1000,
      };
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  } while (performance.now() < deadline);
  assert(false, "scheduler did not release after stream abort", health);
}

async function main() {
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
  let sawDelta = false;
  for await (const chunk of cancellationStream) {
    const delta = chunk.choices?.[0]?.delta;
    const substantive =
      (typeof delta?.content === "string" && delta.content.length > 0) ||
      (typeof delta?.reasoning_content === "string" &&
        delta.reasoning_content.length > 0) ||
      (Array.isArray(delta?.tool_calls) && delta.tool_calls.length > 0);
    if (substantive) {
      sawDelta = true;
      cancellationStream.controller.abort();
      break;
    }
  }
  assert(sawDelta, "cancellation stream produced no delta before termination");
  const released = await schedulerRelease();
  console.log(
    "cancellation_release",
    compact({ saw_delta: sawDelta, ...released }),
  );

  const chatStarted = performance.now();
  const firstChat = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: codingPrompt }],
    tools: [chatTool],
    tool_choice: "auto",
    max_completion_tokens: 128,
    temperature: 0,
  });
  const firstChatMessage = firstChat.choices?.[0]?.message;
  const chatCalls = firstChatMessage?.tool_calls;
  assert(
    Array.isArray(chatCalls) && chatCalls.length > 0,
    "Chat did not generate a valid tool call",
    firstChat,
  );
  const chatArgs = parseArguments(chatCalls[0], firstChat);

  const secondChat = await client.chat.completions.create({
    model,
    messages: [
      { role: "user", content: codingPrompt },
      {
        role: "assistant",
        content: firstChatMessage.content ?? null,
        tool_calls: chatCalls,
      },
      {
        role: "tool",
        tool_call_id: chatCalls[0].id,
        content: toolOutput,
      },
    ],
    tools: [chatTool],
    tool_choice: "auto",
    max_completion_tokens: 64,
    temperature: 0,
  });
  const secondChatMessage = secondChat.choices?.[0]?.message;
  assert(
    chatTextResult(secondChatMessage),
    "Chat tool-result continuation returned no subsequent assistant text",
    secondChat,
  );
  const chatSeconds = (performance.now() - chatStarted) / 1000;

  const responsesStarted = performance.now();
  const firstResponse = await client.responses.create({
    model,
    input: codingPrompt,
    tools: [responsesTool],
    tool_choice: "auto",
    max_output_tokens: 128,
    temperature: 0,
  });
  const responseCallItem = firstResponse.output?.find(
    (item) => item.type === "function_call",
  );
  assert(
    responseCallItem,
    "Responses did not generate a valid function_call",
    firstResponse,
  );
  const { call: responseCall, args: responseArgs } = parseResponseCall(
    responseCallItem,
    firstResponse,
  );

  const secondResponse = await client.responses.create({
    model,
    previous_response_id: firstResponse.id,
    input: [
      {
        type: "function_call_output",
        call_id: responseCall.id,
        output: toolOutput,
      },
    ],
    tools: [responsesTool],
    tool_choice: "auto",
    max_output_tokens: 64,
    temperature: 0,
  });
  assert(
    responseTextResult(secondResponse),
    "Responses tool-result continuation returned no subsequent assistant text",
    secondResponse,
  );
  const responsesSeconds = (performance.now() - responsesStarted) / 1000;

  const report = {
    requests: {
      chat_tool_loop: {
        valid_tool_call: true,
        tool_name: chatCalls[0].function.name,
        path: chatArgs.path,
        assistant_result: true,
        elapsed_seconds: chatSeconds,
      },
      responses_tool_loop: {
        valid_tool_call: true,
        tool_name: responseCall.function.name,
        path: responseArgs.path,
        assistant_result: true,
        elapsed_seconds: responsesSeconds,
      },
    },
    cancellation: {
      aborted_after_first_delta: true,
      saw_delta: sawDelta,
      ...released,
    },
  };
  console.log("agent_tool", compact(report));
  console.log(`${marker}${JSON.stringify(report)}`);
}

main().catch((error) => {
  console.error(error.stack ?? error);
  if (error.raw !== undefined) console.error("raw", compact(error.raw));
  process.exitCode = 1;
});
