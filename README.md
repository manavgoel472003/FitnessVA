# Fitness VA (Colab) — Overview + Function Notes
To see the fitness_va_colab.ipynb upload it to colab, apparently GitHub is not making the contents visible.
Link to Demo and Data : https://drive.google.com/drive/folders/1Oh9CbyBbZkowW_jft85TF63mV0P22MBD?usp=drive_link

This README is aligned with `NLP/fitness_va_colab_inf.ipynb` (the Colab “inference” notebook). It uses a simple pipeline:

- plan what to search
- gather context (offline ExerciseDB + optional web snippets)
- draft an answer
- refine for safety/formatting

## Running in Colab (what you need)
- **Google Drive files** (as used in the notebook):
  - `exercises.json` (ExerciseDB dataset)
  - `user_profile.json` (your profile)
- **HuggingFace token (needed for 14B)**:
  - To pull `Qwen/Qwen2.5-14B-Instruct-1M`, you must use *your own* HuggingFace token (and have access to download the model).
  - In Colab, either run `huggingface_hub.login(<your_token>)` (as shown in the notebook) or set `HF_TOKEN` in the environment.

## How to use it
- Easiest: call `ask("your question")` for a one-off answer + logs.
- If you want a chat loop, keep a running `messages` list and call `run_agent_once(agent, messages)` each turn (no code changes required).

## Config and Flags

These are all to allow changes to the agent, without explicitly chaning the code
- `AgentConfig`: stores model + generation settings + file paths. 
- `SearchPlan`: carries the planner result (web query + offline keywords + plan-needed flag).
- `USE_ADVANCED_PROMPTS`: toggles advanced vs basic prompts.

## Loaders
### Model loaders
- `build_llm(config)`: loads the HF model/tokenizer and wraps it in a LangChain chat interface.

### Data loaders
- `load_exercise_dataset(path)`: loads `exercises.json` with safe fallbacks.
- `load_active_profile(path)`: loads active profile from `user_profile.json`.

## Helpers
All these are functions that are tools or either help in building context or refine the repsones.
### Intent + keyword helpers
- `looks_like_workout_request(text)`: checks if the user is asking for a plan/routine. 
- `extract_muscle_keywords(text, limit)`: extracts muscle terms from text.
- `llm_muscle_keywords(call_llm_fn, question, limit)`: LLM-assisted muscle extraction + normalization.

### Web search + summarization
- `search_duckduckgo(query, max_results)`: fetches a few DDG hits.
- `fetch_url_text(url, max_chars)`: fetches a page and strips HTML.
- `chunk_text(text, max_words)`: splits long text into manageable chunks, this avoids huge prompts and keeps summaries stable.
- `summarize_text(call_llm_fn, instruction, text, word_limit)`: summarizes text with a word cap, to compresses web context into something usable.
- `enrich_with_summaries(results, call_llm_fn)`: chooses a good link, chunk-summarizes, then merges.

### Offline dataset search
- `search_exercise_db(dataset, keywords, limit)`: scores exercises by token matches and returns short summaries.

### Output cleanup
- `truncate_words(text, limit)`: hard caps generated text. Avoid overrun text
- `clean_model_output(text)`: strips model artifacts and normalizes whitespace.
- `polish_output(text)`: removes leftover labels and tidies Markdown.

### Plan → draft → refine (the agent “brain”)
- `plan_queries(call_llm_fn, question, profile, history_text)`: produces a `SearchPlan`.
- `draft_response(call_llm_fn, question, profile, history_text, search_plan, ddg_context, offline_context)`: writes the first answer draft.
- `refine_response(call_llm_fn, question, draft)`: To refine the intial draft.

## Agent helpers
- `SingleResponseFitnessAgent`: encapsulates the one-turn pipeline.
- `run_agent_once(agent, messages)`: runs one turn and returns updated `messages`. Why: lets you build a chat loop without changing the agent.
- `ask(query)`: convenience wrapper that also captures logs. Why: easiest way to demo and debug.

## Evaluation / A-B / Security helpers
- `evaluate_models(model_ids, questions)`: runs the same queries across model IDs and saves CSV/JSON.
- `run_ab_test(model_id, queries)`: runs with `USE_ADVANCED_PROMPTS` on/off and saves CSV/JSON.
- `run_security_tests(model_ids, prompts)`: runs injection probes and saves JSON. 

## Output artifacts (saved by the test cells)
- `evaluation_results.csv` / `evaluation_results.json`
- `advanced_toggle_results.csv` / `advanced_toggle_results.json`
- `security_results.json`
- `analysis.md` (summary takeaways)


