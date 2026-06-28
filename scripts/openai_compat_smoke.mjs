import { createRequire } from "node:module";

const requireFromCwd = createRequire(`${process.cwd()}/`);
const { default: OpenAI } = await import(requireFromCwd.resolve("openai"));

const rootURL = process.env.SUPERSONIC_BASE_URL ?? "http://127.0.0.1:8080";
const baseURL = rootURL.endsWith("/v1") ? rootURL : `${rootURL}/v1`;
const apiKey = process.env.SUPERSONIC_API_KEY ?? "secret";
const model = process.env.SUPERSONIC_SMOKE_MODEL ?? "local";

const client = new OpenAI({ baseURL, apiKey });

const compact = (value) => JSON.stringify(value);

async function main() {
  const models = await client.models.list();
  console.log("models", models.data.map((m) => m.id).join(","));

  const retrieved = await client.models.retrieve(model);
  console.log("model", retrieved.id);

  console.log("[smoke] chat.completions");
  const chat = await client.chat.completions.create({
    model,
    messages: [
      { role: "developer", content: "Answer briefly." },
      { role: "user", content: "Say hi" },
    ],
    max_completion_tokens: 4,
    temperature: 0,
  });
  console.log(
    "chat",
    compact({
      finish: chat.choices[0].finish_reason,
      content: chat.choices[0].message.content,
      usage: chat.usage,
    }),
  );

  const stream = await client.chat.completions.create({
    model,
    messages: [{ role: "user", content: "Say hi" }],
    max_tokens: 4,
    temperature: 0,
    stream: true,
    stream_options: { include_usage: true },
  });
  let streamed = "";
  let sawUsage = false;
  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content;
    if (delta) streamed += delta;
    if (chunk.usage) sawUsage = true;
  }
  console.log("chat_stream", compact({ streamed, sawUsage }));

  const completion = await client.completions.create({
    model,
    prompt: "Say hi",
    max_tokens: 3,
    temperature: 0,
  });
  console.log(
    "completion",
    compact({
      finish: completion.choices[0].finish_reason,
      text: completion.choices[0].text,
      usage: completion.usage,
    }),
  );

  console.log("[smoke] responses");
  const response = await client.responses.create({
    model,
    input: "Say hi",
    max_output_tokens: 3,
    temperature: 0,
  });
  console.log(
    "response",
    compact({
      id: response.id,
      status: response.status,
      output: response.output?.map((item) => item.type),
    }),
  );

  const fetched = await client.responses.retrieve(response.id);
  console.log("response_get", fetched.id);

  const deleted = await client.responses.delete(response.id);
  console.log("response_delete", compact(deleted));

  console.log("[smoke] tokenize");
  const tokenize = await fetch(baseURL.replace(/\/v1$/, "") + "/tokenize", {
    method: "POST",
    headers: {
      authorization: `Bearer ${apiKey}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model,
      input: "hello world",
      add_special_tokens: false,
    }),
  }).then((r) => r.json());
  console.log("tokenize", compact({ count: tokenize.tokens.length }));

  const metrics = await fetch(baseURL.replace(/\/v1$/, "") + "/metrics", {
    headers: { authorization: `Bearer ${apiKey}` },
  }).then((r) => r.text());
  console.log("metrics", metrics.includes("supersonic_active_requests"));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
