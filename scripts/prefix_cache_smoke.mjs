const rootURL = process.env.SUPERSONIC_BASE_URL ?? "http://127.0.0.1:8080";
const baseURL = rootURL.endsWith("/v1") ? rootURL : `${rootURL}/v1`;
const apiKey = process.env.SUPERSONIC_API_KEY;
const model = process.env.SUPERSONIC_SMOKE_MODEL ?? "local";
const prompt =
  process.env.SUPERSONIC_PREFIX_CACHE_PROMPT ??
  "Please answer briefly. ".repeat(80);
const mode = process.env.SUPERSONIC_PREFIX_CACHE_MODE ?? "auto";
const retention = process.env.SUPERSONIC_PROMPT_CACHE_RETENTION ?? "in_memory";

const headers = { "content-type": "application/json" };
if (apiKey) headers.authorization = `Bearer ${apiKey}`;

async function postJSON(path, body) {
  const resp = await fetch(`${baseURL}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = text;
  }
  if (!resp.ok) {
    throw new Error(`${path} failed with ${resp.status}: ${text}`);
  }
  return data;
}

async function getText(path) {
  const resp = await fetch(`${rootURL.replace(/\/v1$/, "")}${path}`, {
    headers: apiKey ? { authorization: `Bearer ${apiKey}` } : undefined,
  });
  const text = await resp.text();
  if (!resp.ok) {
    throw new Error(`${path} failed with ${resp.status}: ${text}`);
  }
  return text;
}

async function getJSON(path) {
  const text = await getText(path);
  return JSON.parse(text);
}

function cachedTokens(resp) {
  return resp?.usage?.prompt_tokens_details?.cached_tokens ?? 0;
}

function outputText(resp) {
  if (resp.choices?.[0]?.text !== undefined) {
    return resp.choices[0].text;
  }
  return resp.choices?.[0]?.message?.content ?? "";
}

function promptTokens(resp) {
  return resp?.usage?.prompt_tokens ?? 0;
}

async function selectEndpoint() {
  if (mode === "completions" || mode === "chat") {
    return mode;
  }
  const capabilities = await getJSON("/v1/capabilities");
  return capabilities.chat ? "chat" : "completions";
}

async function main() {
  const endpoint = await selectEndpoint();
  const cacheFields = {
    prompt_cache_key: "supersonic-prefix-cache-smoke",
    prompt_cache_retention: retention,
    user: "supersonic-smoke",
  };
  const body =
    endpoint === "chat"
      ? {
          model,
          messages: [{ role: "user", content: prompt }],
          max_tokens: 1,
          temperature: 0,
          ...cacheFields,
        }
      : {
          model,
          prompt,
          max_tokens: 1,
          temperature: 0,
          ...cacheFields,
        };
  const path = endpoint === "chat" ? "/chat/completions" : "/completions";

  const first = await postJSON(path, body);
  const second = await postJSON(path, body);
  const firstCached = cachedTokens(first);
  const secondCached = cachedTokens(second);

  if (firstCached !== 0) {
    throw new Error(`expected first request to miss cache, got ${firstCached}`);
  }
  if (secondCached <= 0) {
    throw new Error(`expected second request to hit cache, got ${secondCached}`);
  }
  if (outputText(first) !== outputText(second)) {
    throw new Error(
      `deterministic outputs differ: ${JSON.stringify(first)} vs ${JSON.stringify(second)}`,
    );
  }

  const extendedBody =
    endpoint === "chat"
      ? {
          ...body,
          messages: [
            { role: "user", content: prompt },
            { role: "assistant", content: outputText(first) || "ok" },
            { role: "user", content: "Continue." },
          ],
        }
      : {
          ...body,
          prompt: `${prompt} Continue.`,
        };
  const extended = await postJSON(path, extendedBody);
  const extendedCached = cachedTokens(extended);
  const extendedPromptTokens = promptTokens(extended);
  if (extendedCached <= 0) {
    throw new Error(`expected extended request to hit prefix cache, got ${extendedCached}`);
  }
  if (extendedCached >= extendedPromptTokens) {
    throw new Error(
      `expected extended request to be a partial-prefix hit, got cached=${extendedCached} prompt=${extendedPromptTokens}`,
    );
  }

  const metrics = await getText("/metrics");
  const hasHitMetric = /supersonic_prefix_cache_hits\s+[1-9]/.test(metrics);
  if (!hasHitMetric) {
    throw new Error("metrics did not report a prefix-cache hit");
  }

  console.log(
    "prefix_cache_smoke",
    JSON.stringify({
      first_cached_tokens: firstCached,
      second_cached_tokens: secondCached,
      extended_cached_tokens: extendedCached,
      extended_prompt_tokens: extendedPromptTokens,
      endpoint,
      retention,
      text: outputText(second),
    }),
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
