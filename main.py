"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
  python main.py --mode demo          # Run one end-to-end demo
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_cli():
    """Run CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Run web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation():
    """Run system evaluation."""
    import yaml
    from dotenv import load_dotenv
    from src.autogen_orchestrator import AutoGenOrchestrator
    
    # Load environment variables
    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize AutoGen orchestrator
    print("Initializing AutoGen orchestrator...")
    orchestrator = AutoGenOrchestrator(config)
    
    # For now, run a simple test query
    # TODO: Integrate with SystemEvaluator for full evaluation
    # Suggested implementation:
    # - Import SystemEvaluator from src/evaluation/evaluator.py
    # - Load test queries from data/example_queries.json
    # - Run batch evaluation and print/save the report summary
    from src.evaluation.evaluator import SystemEvaluator

    print("\n" + "=" * 70)
    print("RUNNING BATCH EVALUATION")
    print("=" * 70)

    evaluator = SystemEvaluator(config, orchestrator=orchestrator)
    report = await evaluator.evaluate_system("data/example_queries.json")

    if "error" in report:
        print(f"\nEvaluation error: {report['error']}")
        return

    summary = report.get("summary", {})
    scores = report.get("scores", {})
    print(f"\nTotal Queries: {summary.get('total_queries', 0)}")
    print(f"Successful: {summary.get('successful', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Overall Average Score: {scores.get('overall_average', 0.0):.3f}")

    print("\nScores by Criterion:")
    for criterion, score in scores.get("by_criterion", {}).items():
        print(f"  - {criterion}: {score:.3f}")

    print("\nDetailed evaluation artifacts were saved to outputs/.")


def run_autogen():
    """Run AutoGen example."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


async def run_demo(config_path: str = "config.yaml"):
    """Run one complete query through agents, safety, tools, and judge."""
    import logging
    import yaml
    from dotenv import load_dotenv
    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.judge import LLMJudge

    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(name)s - %(message)s")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    query = "What are the key principles of accessible UI design?"

    print("\n" + "=" * 80)
    print("END-TO-END MULTI-AGENT DEMO")
    print("=" * 80)

    print("\n## User Query")
    print(query)

    print("\nRunning Planner -> Researcher -> Writer -> Critic...")
    orchestrator = AutoGenOrchestrator(config)
    result = orchestrator.process_query(query)

    metadata = result.get("metadata", {})

    print("\n## Agent Traces")
    for agent, actions in metadata.get("agent_traces", {}).items():
        print(f"\n### {agent}")
        for action in actions:
            preview = str(action.get("details", "")).replace("\n", " ")
            print(f"- [{action.get('order', '?')}] {preview[:300]}")

    print("\n## Final Answer")
    print(result.get("response", "No response generated"))

    print("\n## Sources")
    print(f"Sources gathered: {metadata.get('num_sources', 0)}")
    print(f"Evidence strength: {metadata.get('evidence_strength', 'Unknown')}")
    for tool, sources in metadata.get("source_groups", {}).items():
        print(f"\n### {tool} ({len(sources)} source(s))")
        for source in sources[:10]:
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            print(f"- {title} {f'({url})' if url else ''}")

    print("\n## Safety Events")
    safety_events = metadata.get("safety_events", [])
    if not safety_events:
        print("No safety events recorded.")
    for event in safety_events:
        print(
            f"- {event.get('type', 'unknown')} | "
            f"role={event.get('role', 'unknown')} | "
            f"decision={event.get('decision_type', event.get('action', 'allow'))} | "
            f"layer={event.get('policy_layer', 'unknown')} | "
            f"safe={event.get('safe', True)}"
        )

    print("\n## Judge Scores")
    judge = LLMJudge(config)
    evaluation = await judge.evaluate(
        query=query,
        response=result.get("response", ""),
        sources=metadata.get("sources", []),
        evidence_context={
            "evidence_strength": metadata.get("evidence_strength"),
            "tool_traces": metadata.get("tool_traces", []),
            "citation_mapping": metadata.get("citation_mapping", []),
            "agent_traces": metadata.get("agent_traces", {}),
        },
    )
    print(f"Overall Score: {evaluation.get('overall_score', 0.0):.3f}")
    for criterion, score_data in evaluation.get("criterion_scores", {}).items():
        print(f"- {criterion}: {score_data.get('score', 0.0):.3f}")
        print(f"  Reasoning: {score_data.get('reasoning', '')[:240]}")

    print("\nDemo complete.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen", "demo"],
        default="autogen",
        help="Mode to run: cli, web, evaluate, autogen, or demo (default: autogen)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation())
    elif args.mode == "autogen":
        run_autogen()
    elif args.mode == "demo":
        asyncio.run(run_demo(args.config))


if __name__ == "__main__":
    main()
