from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, HumanMessage

from fitness_agent import AgentConfig, build_fitness_agent, run_agent_once


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Personal Fitness Virtual Assistant agent built with LangGraph."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional one-shot prompt. If omitted, the agent starts an interactive chat loop.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID to use for local generation (defaults to a 3B model for 8GB GPUs).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Maximum tokens the local model can generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature for the local LLM.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling configuration.",
    )
    parser.add_argument(
        "--architecture-doc",
        default="fitness_va_architecture_design.md",
        help="Path to the architecture overview document used by the agent.",
    )
    parser.add_argument(
        "--exercise-dataset",
        default="exercisedb-api/src/data/exercises.json",
        help="Path to the offline ExerciseDB dataset.",
    )
    parser.add_argument(
        "--user-profile",
        default="user_profile.json",
        help="Path to the JSON file where the agent stores the local user profile.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of conversation turns for interactive chat.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        architecture_doc=Path(args.architecture_doc).resolve(),
        exercise_db=Path(args.exercise_dataset).resolve(),
        user_profile=Path(args.user_profile).resolve(),
    )


def run_prompt(agent, prompt: str) -> None:
    messages: List = [HumanMessage(content=prompt)]
    response_messages = run_agent_once(agent, messages)
    final_message = response_messages[-1]
    print(final_message.content)


def run_chat(agent, max_turns: int) -> None:
    messages: List = []
    for turn in range(max_turns):
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            print("Exiting chat loop.")
            break
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))
        response_messages = run_agent_once(agent, messages)
        ai_message = response_messages[-1]
        # Update conversation memory with the latest assistant message.
        messages = response_messages
        if isinstance(ai_message, AIMessage):
            print(f"Assistant> {ai_message.content}")
        else:
            print("Assistant> [No content returned]")
    else:
        print("Max turns reached.")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    try:
        agent = build_fitness_agent(config)
    except FileNotFoundError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.prompt:
        run_prompt(agent, args.prompt)
    else:
        run_chat(agent, args.max_turns)


if __name__ == "__main__":
    main()
