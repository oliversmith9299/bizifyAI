"""
Orchestrator — Startup Discovery Pipeline
==========================================
Runs the full pipeline in order:

  Step 1: ProfileAnalysis.py
          Input : questionnaireOutput.json + skills.json
          Output: profileAnalysis.json

  Step 2: ProblemDiscovery.py
          Input : profileAnalysis.json + questionnaireOutput.json
          Output: problems.json

  Step 3: personalizeideachat.py
          Input : all above + idea_output.json (running session)
          Output: idea_output.json (updated each turn)

Usage:
  python orchestrator.py              → run full pipeline
  python orchestrator.py --step 2     → run from step 2 onward (skip ProfileAnalysis)
  python orchestrator.py --step 3     → run only IdeaChat (skip both analysis steps)
  python orchestrator.py --only 1     → run ONLY ProfileAnalysis and stop
  python orchestrator.py --only 2     → run ONLY ProblemDiscovery and stop
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("orchestrator")

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AGENTS_DIR = os.path.join(BASE_DIR, "agents")
DATA_DIR   = os.path.join(BASE_DIR, "data")

AGENTS = {
    1: os.path.join(AGENTS_DIR, "ProfileAnalysis.py"),
    2: os.path.join(AGENTS_DIR, "ProblemDiscovery.py"),
    3: os.path.join(AGENTS_DIR, "personalizeideachat.py"),
}

# Files each step REQUIRES before it can run
REQUIRED_INPUTS = {
    1: [
        os.path.join(DATA_DIR, "questionnaireOutput.json"),
        os.path.join(DATA_DIR, "skills.json"),
    ],
    2: [
        os.path.join(DATA_DIR, "profileAnalysis.json"),
        os.path.join(DATA_DIR, "questionnaireOutput.json"),
    ],
    3: [
        os.path.join(DATA_DIR, "profileAnalysis.json"),
        os.path.join(DATA_DIR, "problems.json"),
        os.path.join(DATA_DIR, "questionnaireOutput.json"),
        os.path.join(DATA_DIR, "skills.json"),
    ],
}

# Files each step is expected to produce
EXPECTED_OUTPUTS = {
    1: os.path.join(DATA_DIR, "profileAnalysis.json"),
    2: os.path.join(DATA_DIR, "problems.json"),
    3: os.path.join(DATA_DIR, "idea_output.json"),
}

STEP_NAMES = {
    1: "ProfileAnalysis   — Build founder profile from questionnaire + skills",
    2: "ProblemDiscovery  — Discover and validate real customer problems",
    3: "IdeaChat          — Generate and refine startup idea interactively",
}


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def print_banner():
    print("\n" + "═" * 58)
    print("  🚀  Startup Discovery Pipeline — Orchestrator")
    print("═" * 58)
    print(f"  Data dir  : {DATA_DIR}")
    print(f"  Agents dir: {AGENTS_DIR}")
    print("═" * 58 + "\n")


def print_step_header(step: int):
    print("\n" + "─" * 58)
    print(f"  STEP {step}: {STEP_NAMES[step]}")
    print("─" * 58)


def check_inputs(step: int) -> bool:
    """Verify all required input files exist before running a step."""
    missing = [f for f in REQUIRED_INPUTS[step] if not os.path.exists(f)]
    if missing:
        log.error(f"Step {step} is missing required input files:")
        for f in missing:
            log.error(f"  ✗ {f}")
        return False
    return True


def check_output_produced(step: int) -> bool:
    """Verify the step actually wrote its output file."""
    out = EXPECTED_OUTPUTS[step]
    if not os.path.exists(out):
        log.error(f"Step {step} finished but output file was NOT created: {out}")
        return False

    # Also verify it's valid JSON (for steps 1 and 2)
    if step in (1, 2):
        try:
            with open(out) as f:
                data = json.load(f)
            # Sanity check: non-empty output
            if not data:
                log.error(f"Step {step} output is empty JSON: {out}")
                return False
            log.info(f"✅ Output verified: {os.path.basename(out)}")
            return True
        except json.JSONDecodeError as e:
            log.error(f"Step {step} output is invalid JSON: {e}")
            return False

    return True


def run_step(step: int) -> bool:
    """
    Run a single pipeline step as a subprocess.
    Returns True if succeeded, False if failed.
    Step 3 (IdeaChat) is interactive — we hand control to it directly.
    """
    script = AGENTS[step]

    if not os.path.exists(script):
        log.error(f"Agent script not found: {script}")
        return False

    print_step_header(step)

    if not check_inputs(step):
        log.error(f"Aborting — fix missing inputs before running step {step}.")
        return False

    start = time.time()

    if step == 3:
        # IdeaChat is interactive — run it in the same process so the user
        # can type in the terminal normally. We exec() it directly.
        log.info("Starting interactive IdeaChat agent...")
        log.info("(The pipeline hands control to you now — type your responses below)\n")
        try:
            # Run interactively in same terminal — inherit stdin/stdout
            result = subprocess.run(
                [sys.executable, script],
                cwd=AGENTS_DIR,
            )
            elapsed = time.time() - start
            if result.returncode == 0:
                log.info(f"IdeaChat session ended. ({elapsed:.1f}s)")
                return check_output_produced(step)
            else:
                log.error(f"IdeaChat exited with code {result.returncode}")
                return False
        except KeyboardInterrupt:
            log.info("\nIdeaChat interrupted by user.")
            return False

    else:
        # Steps 1 and 2 are non-interactive — capture output and stream it
        log.info(f"Running {os.path.basename(script)}...")
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=AGENTS_DIR,
                capture_output=False,   # stream stdout/stderr to terminal live
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                log.error(f"Step {step} failed (exit code {result.returncode}) after {elapsed:.1f}s")
                return False

            log.info(f"Step {step} completed in {elapsed:.1f}s")
            return check_output_produced(step)

        except KeyboardInterrupt:
            log.warning(f"\nStep {step} interrupted by user.")
            return False
        except Exception as e:
            log.error(f"Step {step} raised an unexpected error: {e}")
            return False


def print_summary(results: dict):
    print("\n" + "═" * 58)
    print("  Pipeline Summary")
    print("═" * 58)
    for step, (name_short, success) in results.items():
        icon = "✅" if success else "❌"
        print(f"  {icon}  Step {step}: {name_short}")
    print("═" * 58)

    all_ok = all(s for _, s in results.values())
    if all_ok:
        idea_path = EXPECTED_OUTPUTS[3]
        print(f"\n  🎉 Pipeline complete! Your idea is saved to:")
        print(f"     {idea_path}\n")
    else:
        print("\n  ⚠️  Pipeline stopped due to errors above.\n")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Startup Discovery Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py              Run full pipeline (steps 1 → 2 → 3)
  python orchestrator.py --step 2     Start from step 2 (skip ProfileAnalysis)
  python orchestrator.py --step 3     Start from step 3 (skip analysis, go to chat)
  python orchestrator.py --only 1     Run only ProfileAnalysis
  python orchestrator.py --only 2     Run only ProblemDiscovery
        """
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3], default=1,
        help="Start pipeline from this step (default: 1 = full run)"
    )
    parser.add_argument(
        "--only", type=int, choices=[1, 2, 3],
        help="Run ONLY this step and stop"
    )
    args = parser.parse_args()

    print_banner()

    # Determine which steps to run
    if args.only is not None:
        steps_to_run = [args.only]
        log.info(f"Running only Step {args.only}: {STEP_NAMES[args.only]}")
    else:
        steps_to_run = list(range(args.step, 4))
        log.info(f"Running steps: {steps_to_run}")

    results = {}

    for step in steps_to_run:
        success = run_step(step)
        results[step] = (STEP_NAMES[step].split("—")[0].strip(), success)

        if not success:
            log.error(f"Pipeline halted at step {step}.")
            # Still print summary for steps that ran
            print_summary(results)
            sys.exit(1)

        # Brief pause between steps so log output is readable
        if step < steps_to_run[-1]:
            time.sleep(0.5)

    print_summary(results)


if __name__ == "__main__":
    main()