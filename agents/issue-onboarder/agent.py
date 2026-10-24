"""Issue Onboarder Agent.

A Google Agent Development Kit (ADK) agent powered by gemini-2.5-flash that analyzes
open-source repository issues, inspects local codebase files via custom tools, and
generates clear, actionable, beginner-friendly onboarding guides for new contributors.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from local .env or parent environments
load_dotenv()

# Google ADK imports
try:
    from google.adk.agents import Agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
except ImportError:
    # Allow importing module even when dependencies are being resolved
    Agent = None
    Runner = None
    InMemorySessionService = None
    types = None


# --- Custom Tools ---

def read_local_code_file(file_path: str) -> str:
    """Reads and returns the contents of a local code or documentation file.

    Use this tool to inspect relevant files in the repository (e.g. source files,
    configuration, README.md, CONTRIBUTING.md, unit tests) to understand the issue context
    and provide accurate guidance.

    Args:
        file_path: Relative or absolute path to the local file to read (e.g. 'README.md', 'src/main.py').

    Returns:
        The content of the file as text, or an error message if the file cannot be accessed.
    """
    try:
        path = Path(file_path).resolve()
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if path.is_dir():
            return f"Error: '{file_path}' is a directory, not a file."

        # Safety check: prevent opening oversized binary/blob files
        max_bytes = 25_000
        if path.stat().st_size > max_bytes:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_bytes)
            return content + f"\n\n[Warning: File exceeded {max_bytes} bytes and was truncated]"

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as exc:
        return f"Error reading file '{file_path}': {str(exc)}"


def list_repository_files(directory_path: str = ".") -> str:
    """Lists files and folders within a given directory in the repository.

    Use this tool to explore the folder structure, locate relevant subdirectories,
    and find documentation or source files.

    Args:
        directory_path: Relative or absolute path to the directory (defaults to current directory '.').

    Returns:
        A list of files and folders in the directory, or an error message.
    """
    try:
        dir_path = Path(directory_path).resolve()
        if not dir_path.exists():
            return f"Error: Directory '{directory_path}' does not exist."
        if not dir_path.is_dir():
            return f"Error: Path '{directory_path}' is a file, not a directory."

        ignored_patterns = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}
        entries = []
        for entry in sorted(dir_path.iterdir()):
            if entry.name in ignored_patterns:
                continue
            entry_type = "[DIR]" if entry.is_dir() else "[FILE]"
            entries.append(f"{entry_type} {entry.name}")

        return "\n".join(entries) if entries else "(Empty directory)"
    except Exception as exc:
        return f"Error listing directory '{directory_path}': {str(exc)}"


# --- Agent Definition ---

AGENT_NAME = "issue_onboarder"
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gemini-3.8-flash")

AGENT_INSTRUCTION = """\
You are an expert Open-Source Mentor and Contributor Onboarding Assistant.
Your goal is to help first-time and beginner developers successfully understand,
implement, test, and submit pull requests for issues in this repository.

When given an open-source issue, bug report, or feature request:
1. Examine the repository structure using `list_repository_files` and read relevant project
   files (such as `README.md`, `CONTRIBUTING.md`, source code, and tests) using `read_local_code_file`.
2. Analyze the problem thoroughly and locate the exact components and files involved.
3. Produce a structured, beginner-friendly Onboarding Plan formatted in Markdown:
   - **1. Issue Summary & Context**: A clear, non-jargon explanation of what the issue is asking for.
   - **2. Prerequisites & Environment Setup**: What tools, language runtime, and virtual environment steps are needed.
   - **3. Codebase Walkthrough**: Specific files, classes, and functions involved, explaining how they work together.
   - **4. Step-by-Step Implementation Guide**: An actionable sequence of steps to solve the issue, including code snippets or pseudo-code where helpful.
   - **5. Local Testing & Verification**: Concrete instructions on how to run tests or verify the change locally.
   - **6. Git & Pull Request Checklist**: Branch naming, commit guidelines, and tips for submitting a clean PR based on repository standards.

Maintain an encouraging, empathetic, and instructive tone throughout.
"""

# ADK Root Agent declaration (used by ADK CLI and runners)
if Agent is not None:
    root_agent = Agent(
        name=AGENT_NAME,
        model=DEFAULT_MODEL,
        description="Analyzes open-source issues and repository code to create beginner-friendly contributor onboarding plans.",
        instruction=AGENT_INSTRUCTION,
        tools=[read_local_code_file, list_repository_files],
    )
else:
    root_agent = None


# --- Asynchronous Runner & CLI Execution ---

async def run_onboarder(issue_prompt: str, model_name: str = DEFAULT_MODEL) -> None:
    """Runs the issue onboarder agent for a given issue prompt using Google ADK."""
    if Agent is None or Runner is None or InMemorySessionService is None:
        print("Error: 'google-adk' is not installed. Please run:")
        print("    pip install -r requirements.txt")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[!] Error: GEMINI_API_KEY is not configured.")
        print("    To run the agent with Gemini 2.5 Flash:")
        print("    1. Copy .env.example to .env: Copy-Item .env.example .env")
        print("    2. Set GEMINI_API_KEY=your_api_key in .env")
        print("    (Get a free key from Google AI Studio: https://aistudio.google.com/)\n")
        return

    os.environ["GEMINI_API_KEY"] = api_key

    print(f"\n[Issue Onboarder Agent | Model: {model_name}]")
    print("-" * 60)
    print(f"Analyzing Issue:\n{issue_prompt}\n")
    print("-" * 60)
    print("Generating contributor onboarding plan...\n")

    user_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Please review this open-source repository issue and generate a beginner-friendly contributor onboarding plan:\n\n{issue_prompt}")]
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            agent = Agent(
                name=AGENT_NAME,
                model=model_name,
                description="Analyzes open-source issues and repository code to create beginner-friendly contributor onboarding plans.",
                instruction=AGENT_INSTRUCTION,
                tools=[read_local_code_file, list_repository_files],
            )

            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name=AGENT_NAME,
                user_id="contributor",
                state={}
            )

            runner = Runner(
                app_name=AGENT_NAME,
                agent=agent,
                session_service=session_service,
            )

            events = runner.run_async(
                session_id=session.id,
                user_id="contributor",
                new_message=user_message,
            )

            async for event in events:
                # Display tool invocations so user sees real-time progress
                if hasattr(event, "get_function_calls"):
                    for fc in event.get_function_calls():
                        args_summary = ", ".join(f"{k}={v!r}" for k, v in (fc.args or {}).items())
                        print(f"\n[Agent Tool Call] {fc.name}({args_summary})", flush=True)

                if hasattr(event, "get_function_responses"):
                    for fr in event.get_function_responses():
                        print(f"[Agent Tool Result] Processed response from {fr.name}\n", flush=True)

                # Display generated text or thinking indicators
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "thought", False):
                            # Output a thinking dot so user knows the model is reasoning
                            print(".", end="", flush=True)
                        elif part.text:
                            print(part.text, end="", flush=True)

            print("\n\n" + "-" * 60)
            print("Onboarding plan complete!")
            return

        except Exception as exc:
            err_msg = str(exc)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < max_retries:
                wait_time = attempt * 4
                print(f"\n[!] Model '{model_name}' temporary 503 spike. Waiting {wait_time}s before automatic retry ({attempt}/{max_retries})...")
                await asyncio.sleep(wait_time)
                continue
            print(f"\n[!] Error during agent execution: {exc}")
            break


def main() -> None:
    """CLI entry point for running the Issue Onboarder Agent."""
    parser = argparse.ArgumentParser(
        description="Issue Onboarder Agent - Generates beginner-friendly contributor onboarding plans for open-source issues."
    )
    parser.add_argument(
        "--issue",
        "-i",
        type=str,
        help="Description or text of the issue to analyze."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model identifier (default: {DEFAULT_MODEL})"
    )

    args = parser.parse_args()

    issue_text = args.issue
    if not issue_text:
        print("=== Issue Onboarder Agent ===")
        print("Enter or paste the open-source issue title and description below.")
        print("(Press Enter, then Ctrl+Z and Enter on Windows or Ctrl+D on Linux/macOS when finished):")
        try:
            issue_text = sys.stdin.read().strip()
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)

    if not issue_text:
        print("No issue text provided. Exiting.")
        sys.exit(0)

    asyncio.run(run_onboarder(issue_prompt=issue_text, model_name=args.model))


if __name__ == "__main__":
    main()

