# Jev API notes

Read on 2026-10-24 from TypeSafe's docs (docs.typesafe.ai, including the OpenAPI spec at
api.typesafe.ai/openapi.json) and OpenRouter's Jev pages and API reference. Where something is not
documented it says "unknown". Nothing here was tested against the live API yet.

## Models

- TypeSafe names the model `jev-1.13.0`. OpenRouter calls it `typesafe/jev-1.13` and reports the
  serving version as `typesafe/jev-1.13-20260917` in responses.
- `jev-1.14.0` was released on 2026-10-14. On TypeSafe, `jev-latest` and `jev-preview` both point
  to it. On OpenRouter the alias is `~typesafe/jev-latest`.
- The response's `model` field names the model that actually answered. It can differ from the name
  or alias sent. We record it per run.
- jab sends `jev-latest` by default. Both services accept that alias. Set `JEV_MODEL` to pin a
  version, for example `jev-1.14.0` on TypeSafe or `typesafe/jev-1.14` on OpenRouter.
- Probabilities differ between models on the same question, so a threshold tuned on one version
  has to be tuned again on another (OpenRouter's Jev guide says this directly).

## Endpoints and auth

Both services use the same request and response shape.

| | URL | Key |
|---|---|---|
| TypeSafe direct | `POST https://api.typesafe.ai/v1/systemone` | TypeSafe key |
| OpenRouter | `POST https://openrouter.ai/api/v1/systemone` | OpenRouter key |

- Auth on both is `Authorization: Bearer <key>`.
- OpenRouter also has an alpha `POST /api/alpha/decisions` endpoint. Its schema in the reference is
  identical to `/api/v1/systemone`. We use `/api/v1/systemone`, which is the non-alpha one and the
  one TypeSafe's own SDKs talk to.
- Because the shapes match, one backend class serves both, with a different URL and key.
- OpenRouter accepts bare names like `jev-1.13` and maps them to `typesafe/jev-1.13`.

## Request

```json
{"model": "...", "state": "...", "questions": {"<id>": {"type": "...", "instructions": "...", "criteria": "..."}}}
```

- `state` is a string, an object or an array. All questions in the call are about this state.
- `questions` maps ids we choose to question objects. At least one is required. The model never
  sees the id. Answers come back under the same id.
- `instructions` and each criteria description can be a string, an object, an array or null.
  Objects are for structured guidance, for example "covers", "does not cover", "examples".
- `assets` (images) exist but need a special header and are out of scope for us.
- OpenRouter also accepts `session_id`, `user`, `provider` and `trace`. We do not need them.

### Question types

| Type | `criteria` | Limits | Answer fields |
|---|---|---|---|
| `choice` | object, option name to description | 1 to 255 options | `choice`, `confidence`, `probabilities` |
| `score` | array of level descriptions, lowest first | 1 to 10 levels | `score`, `confidence`, `legend`, `probabilities` |
| `noul` | optional object with `true` and `false` descriptions | none | `noul` only |

- `choice`: `choice` is the option with the highest probability. `probabilities` has one entry per
  option and sums to about 1.
- `score`: `score` is the probability-weighted average level, so it can fall between levels.
  `probabilities` and `legend` are keyed by level number as a string ("0", "1", ...).
- `noul`: `noul` is the probability of yes, 0 to 1. There is no `confidence` field.
- The OpenAPI spec allows a score with a single level. The Score docs page says at least two. We
  require two, since a one-level score cannot say anything.
- A fourth type, `box`, locates things in images. jev-1.13 rejects it with a 400 on OpenRouter.
  We do not support it.

## Response

```json
{"model": "...", "answers": {"<id>": {"type": "choice", "choice": "...", "confidence": 0.75, "probabilities": {}}}, "usage": {"input_tokens": 476, "output_tokens": 70}}
```

- `usage.input_tokens` is billed. `usage.output_tokens` is reported but free.
- OpenRouter adds a top-level `id`, a `provider` and `usage.cost` in USD. In OpenRouter's schema
  `usage.cost` is optional, so we cannot assume it is always there.
- TypeSafe direct returns no cost field. Cost has to be computed from the price.
- No cached-token field on either service.
- No reasoning or explanation is returned.

## Confidence

- `choice`: confidence = (p_max - 1/n) / (1 - 1/n), where n is the number of options. It is 0 for a
  flat distribution and 1 when all probability is on one option. It is not the top probability:
  0.61 on the top option came with confidence 0.42 in one docs example.
- `score`: confidence is 1 minus the expected distance from the most likely level, divided by the
  expected distance an even spread has from the middle level, floored at 0. Mass on a neighbouring
  level costs less than mass far away.
- `noul`: none returned. The docs suggest using the probability itself with 0.7 for yes and 0.3 for
  no as a starting point, or computing |2p - 1| for a confidence-style number.
- The docs do not say confidence is calibrated against accuracy. They say to set thresholds on your
  own data.

What we do with this, per the plan: use the API's `confidence` for choice and score, and the top
probability for noul (that is max(p, 1 - p)). An item's confidence is the lowest across its
questions. Note that for noul this is on a different scale from choice confidence: a noul at
p = 0.7 gets 0.7 here, while a two-option choice at 0.7 would get 0.4 from the API.

## Limits

- Request body: 32 MiB, compressed or not. Nesting: 64 levels in `state`, `instructions` and
  `criteria`. Numbers must be finite. UTF-8 JSON, no BOM.
- Context on TypeSafe: 64k tokens per request and 32k per question by default. A
  `TypeSafe-Danger-Zone-Long-Context: true` header raises both to 262k.
- Context on OpenRouter: the model listing shows 262,144 tokens for jev-1.13. OpenRouter's guide says
  requests with more than about 32,000 bytes of text go to TypeSafe's long-context mode, which
  TypeSafe says is slower and less accurate. We cap the state at 32,000 bytes by default for this
  reason and send longer items to review instead.
- Too many tokens for the context gives a 400 with `max_tokens_exceeded`.
- Max questions per call: unknown.
- Max state size apart from the context and body limits: unknown.
- Rate limits: unknown. No numbers and no rate-limit headers are documented.

## Errors

TypeSafe direct:

| Status | Meaning | Retry |
|---|---|---|
| 400 | broken request rule or over the context length | no |
| 401 | missing or bad key | no |
| 413 | body over 32 MiB | no |
| 415 | unsupported `Content-Encoding` | no |
| 422 | body failed validation, with a list of field errors | no |
| 429 | rate limited | yes, with exponential backoff |
| 529 | overloaded | yes, after a short delay |

OpenRouter adds 402 (out of credits), 403, 404, 500, 502 (upstream error), 503 and 524 (edge
timeout). Its errors look like `{"error": {"code": 429, "message": "..."}}`.

- Whether the server sends `Retry-After`: unknown. TypeSafe's Python SDK honours it if present.
- TypeSafe's SDK retries 408, 429 and 500 to 599, three attempts in total, starting at 0.5 s and
  doubling up to 5 s.
- 422 error details can include a preview of the rejected input. Error handling must never log or
  print the response body as is, because it may contain item text.

## Price

$0.042 per million input tokens on both TypeSafe and OpenRouter. Output tokens are free.

## Writing good questions

From the primitives pages and the jev-1.13 limitations page.

- Put all questions for an item in one call. They are answered in parallel against the same state,
  and one call is cheaper than several.
- Give each choice option a one-line description. When options are easy to confuse, describe what
  each covers and does not cover, with examples that look like real inputs.
- Add an "other" option when the list may not cover every input. "None of the above" alone is
  unreliable. A separate noul ("does this match any of these at all?") works better.
- Score levels must be descriptive. Numeric-only levels like "0", "1", "2" perform badly. Describe
  situations, not degrees. One dimension per score question.
- Phrase a noul so a high value means yes. Ask one condition per noul.
- Keep instructions and criteria consistent with each other.

## Known weaknesses

TypeSafe has a limitations page for jev-1.13 only. Nothing is documented for jev-1.14 yet, so
assume these still apply until measured. Shortened:

- Answers the literal question, not the intent.
- Weak at arithmetic, counting, dates and non-decimal numbers like hex. Do those in code.
- Score levels are weakly calibrated. Use the score against a threshold, not as a measurement.
- Multi-hop reasoning and double negatives lower accuracy.
- Accuracy drops as irrelevant content in the state grows.
- Hostile or misleading text in the state can steer answers.
- Leans toward the first option in a choice.
- Not trained to generate text.

## Unknown

- Rate limits on either service, and whether `Retry-After` is sent.
- Max questions per call.
- Whether confidence is calibrated.
- How long either service keeps request data. Neither Jev page covers data retention.
- Whether OpenRouter always includes `usage.cost` for Jev. It is optional in the schema.
- Which context limit OpenRouter actually applies without long-context mode.
- Known weaknesses of jev-1.14.
