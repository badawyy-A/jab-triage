# Privacy

jab sends items to a third party. This page says what goes where, what stays on your machine, and
how jab makes you say yes first.

## What is sent, and where

Each item is one HTTPS request to the backend named by `JEV_BACKEND`:

| `JEV_BACKEND` | Requests go to | Who sees them |
|---|---|---|
| `openrouter` | `https://openrouter.ai/api/v1/systemone` | OpenRouter, then TypeSafe, which runs Jev |
| `typesafe` | `https://api.typesafe.ai/v1/systemone` | TypeSafe |

A request holds:

- the item's state: the preset's `state_template` filled in with the item's `text` and the `meta`
  fields the template names
- the preset's questions and options
- the model name, `jev-latest` unless `JEV_MODEL` says otherwise
- your API key, in the `Authorization` header

Item ids are not sent. Meta fields the template does not use are not sent. Items whose state is
larger than `--max-bytes` (32,000 by default) are not sent at all and go straight to review.

How long OpenRouter or TypeSafe keep request data is not covered in their Jev docs. Check their
privacy policies before sending anything you would not want stored.

## You have to say yes

`jab ask` and `jab triage` never send anything on the first try. Without `--yes` they print where
the items would go, the model, and the template that shows which fields of each item are sent. Then
they exit with code 3 and nothing sent. The notice shows the template, never item text.

Run the same command again with `--yes` to send. An agent using jab should show you the notice and
ask before adding `--yes`.

## What stays on your machine

- `confident.jsonl` holds ids, answers and confidence only.
- `needs_review.jsonl`, `audit.jsonl`, and the review lines of `jab ask` output also hold each
  item's `text` and `meta`, so the agent can read them without opening the whole input. Treat these
  files like the input.
- `summary.json` holds counts, settings, token usage and cost. No item text.

## Logs and errors

- jab never logs item text unless you pass `--debug`. With `--debug` it logs each state it sends,
  to stderr.
- Error messages name items by id and position, never by text.
- API error bodies are never printed, because some echo the rejected input back. jab prints the
  HTTP status and, for validation errors, only the field paths.

## Keys

Keys come only from environment variables: `OPENROUTER_API_KEY` or `TYPESAFE_API_KEY`. jab never
prints, logs or writes a key. `jab status` says only whether a key is set.
