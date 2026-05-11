"""
System Evaluator
Runs batch evaluations and generates reports.

Example usage:
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator with orchestrator
    evaluator = SystemEvaluator(config, orchestrator=my_orchestrator)
    
    # Run evaluation
    report = await evaluator.evaluate_system("data/test_queries.json")
    
    # Results are automatically saved to outputs/
"""

from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import inspect

from .judge import LLMJudge


class SystemEvaluator:
    """
    Evaluates the multi-agent system using test queries and LLM-as-a-Judge.

    TODO: YOUR CODE HERE
    - Load test queries from file
    - Run system on all test queries
    - Collect and aggregate results
    - Generate evaluation report
    - Perform error analysis
    """

    def __init__(self, config: Dict[str, Any], orchestrator=None):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary (from config.yaml)
            orchestrator: The orchestrator to evaluate
        """
        self.config = config
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("evaluation.evaluator")

        # Load evaluation configuration from config.yaml
        eval_config = config.get("evaluation", {})
        self.enabled = eval_config.get("enabled", True)
        self.max_test_queries = eval_config.get("num_test_queries", None)
        
        # Initialize judge (passes config to load judge model settings and criteria)
        self.judge = LLMJudge(config)

        # Evaluation results
        self.results: List[Dict[str, Any]] = []
        
        self.logger.info(f"SystemEvaluator initialized (enabled={self.enabled})")

    async def evaluate_system(
        self,
        test_queries_path: str = "data/test_queries.json"
    ) -> Dict[str, Any]:
        """
        Run full system evaluation.

        Args:
            test_queries_path: Path to test queries JSON file

        Returns:
            Evaluation results and statistics

        TODO: YOUR CODE HERE
        - Load test queries
        - Run system on each query
        - Evaluate each response
        - Aggregate results
        - Generate report
        """
        # Check if evaluation is enabled in config.yaml
        if not self.enabled:
            self.logger.warning("Evaluation is disabled in config.yaml")
            return {"error": "Evaluation is disabled in configuration"}
        
        self.logger.info("Starting system evaluation")
        # Load test queries
        test_queries = self._load_test_queries(test_queries_path)
        self.logger.info(f"Loaded {len(test_queries)} test queries")

        # Evaluate each query
        for i, test_case in enumerate(test_queries, 1):
            self.logger.info(f"Evaluating query {i}/{len(test_queries)}")

            try:
                result = await self._evaluate_query(test_case)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating query {i}: {e}")
                self.results.append({
                    "query": test_case.get("query", ""),
                    "error": str(e)
                })

        # Aggregate results
        report = self._generate_report()

        # Save results
        self._save_results(report)

        return report

    async def _evaluate_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test query.

        Args:
            test_case: Test case with query and optional ground truth

        Returns:
            Evaluation result for this query

        This shows how to integrate with the orchestrator.
        """
        query = test_case.get("query", "")
        ground_truth = test_case.get("ground_truth")
        expected_sources = test_case.get("expected_sources", [])

        # Run through orchestrator if available
        if self.orchestrator:
            try:
                # Call orchestrator's process_query method
                # TODO: YOUR CODE HERE
                # Need to implement this in their orchestrator
                response_data = self.orchestrator.process_query(query)
                if inspect.isawaitable(response_data):
                    response_data = await response_data
                
                # If process_query is async, use:
                # response_data = await self.orchestrator.process_query(query)
                
            except Exception as e:
                self.logger.error(f"Error processing query through orchestrator: {e}")
                response_data = {
                    "query": query,
                    "response": f"Error: {str(e)}",
                    "citations": [],
                    "metadata": {"error": str(e)}
                }
        else:
            # Placeholder for testing without orchestrator
            self.logger.warning("No orchestrator provided, using deterministic fallback response")
            response_data = {
                "query": query,
                "response": "Evaluation could not run the research system because no orchestrator was connected.",
                "citations": [],
                "metadata": {"num_sources": 0}
            }

        metadata = response_data.get("metadata", {})
        evidence_context = self._build_evidence_context(metadata)

        # Evaluate response using LLM-as-a-Judge with traces and citation mappings.
        evaluation = await self.judge.evaluate(
            query=query,
            response=response_data.get("response", ""),
            sources=metadata.get("sources", []),
            ground_truth=ground_truth,
            evidence_context=evidence_context,
        )

        return {
            "query": query,
            "response": response_data.get("response", ""),
            "evaluation": evaluation,
            "metadata": metadata,
            "evidence_context": evidence_context,
            "ground_truth": ground_truth,
            "expected_sources": expected_sources
        }

    def _build_evidence_context(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Package evidence traces for judge prompts and reproducible reports."""
        sources = self._filter_valid_sources(metadata.get("sources", []))
        tool_traces = metadata.get("tool_traces", [])
        citation_mapping = metadata.get("citation_mapping", [])
        source_groups = self._group_sources_by_tool(sources)
        return {
            "num_sources": len(sources),
            "evidence_strength": metadata.get("evidence_strength", self._calculate_evidence_strength(sources, tool_traces)),
            "sources": sources,
            "source_groups": source_groups,
            "tool_traces": tool_traces,
            "citation_mapping": citation_mapping,
            "agent_traces": metadata.get("agent_traces", {}),
        }

    def _filter_valid_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only deduplicated tool-origin external sources for judging."""
        filtered = []
        seen = set()
        for source in sources:
            tool = source.get("tool")
            title = str(source.get("title", "")).strip()
            url = str(source.get("url", "")).strip()
            paper_id = str(source.get("paper_id", "")).strip()
            combined = f"{title} {source.get('snippet', '')}".lower()
            if tool not in {"web_search", "paper_search"}:
                continue
            if not url and not paper_id:
                continue
            if url and not self._is_valid_external_url(url):
                continue
            if any(term in combined for term in ["planner", "writer", "critic", "researcher", "score:", "instructions"]):
                continue
            key = (url or paper_id or title).lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(source)
        return filtered

    def _is_valid_external_url(self, url: str) -> bool:
        """Validate URL shape for report/evaluation source inputs."""
        import re
        return bool(re.match(r'^https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/|$)', url))

    def _group_sources_by_tool(self, sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Regroup filtered sources by originating tool."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for source in sources:
            groups.setdefault(source.get("tool", "unknown"), []).append(source)
        return groups

    def _calculate_evidence_strength(
        self,
        sources: List[Dict[str, Any]],
        tool_traces: List[Dict[str, Any]],
    ) -> str:
        """Mirror orchestrator evidence labels for fallback evaluation paths."""
        tools = {source.get("tool") for source in sources}
        if len(sources) >= 5 and {"web_search", "paper_search"}.issubset(tools):
            return "High"
        if len(sources) >= 3 or tool_traces:
            return "Medium"
        return "Low"

    def _load_test_queries(self, path: str) -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.

        TODO: YOUR CODE HERE
        - Create test query dataset
        - Load and validate queries
        """
        path_obj = Path(path)
        if not path_obj.exists():
            self.logger.warning(f"Test queries file not found: {path}")
            return []

        with open(path_obj, 'r') as f:
            queries = json.load(f)

        # Validate a small, beginner-friendly schema for each query record.
        if not isinstance(queries, list):
            self.logger.warning("Test query file must contain a list of query objects")
            return []
        valid_queries = []
        for item in queries:
            if isinstance(item, dict) and item.get("query"):
                valid_queries.append(item)
            else:
                self.logger.warning(f"Skipping invalid test query entry: {item}")
        queries = valid_queries

        # Limit number of queries if configured in config.yaml
        if self.max_test_queries and len(queries) > self.max_test_queries:
            self.logger.info(f"Limiting to {self.max_test_queries} queries (from config.yaml)")
            queries = queries[:self.max_test_queries]

        return queries

    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate evaluation report with statistics and analysis.

        TODO: YOUR CODE HERE
        - Calculate aggregate statistics
        - Identify best/worst performing queries
        - Analyze errors
        - Generate visualizations (optional)
        """
        if not self.results:
            return {"error": "No results to report"}

        # Calculate statistics
        total_queries = len(self.results)
        successful = [r for r in self.results if "error" not in r]
        failed = [r for r in self.results if "error" in r]

        # Aggregate scores
        criterion_scores = {}
        overall_scores = []
        source_counts = []
        evidence_strength_counts = {"High": 0, "Medium": 0, "Low": 0}

        for result in successful:
            evaluation = result.get("evaluation", {})
            overall_scores.append(evaluation.get("overall_score", 0.0))
            evidence_context = result.get("evidence_context", {})
            source_counts.append(evidence_context.get("num_sources", 0))
            strength = evidence_context.get("evidence_strength", "Low")
            evidence_strength_counts[strength] = evidence_strength_counts.get(strength, 0) + 1

            # Collect scores by criterion
            for criterion, score_data in evaluation.get("criterion_scores", {}).items():
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                criterion_scores[criterion].append(score_data.get("score", 0.0))

        # Calculate averages
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        avg_sources = sum(source_counts) / len(source_counts) if source_counts else 0.0

        avg_criterion_scores = {}
        for criterion, scores in criterion_scores.items():
            avg_criterion_scores[criterion] = sum(scores) / len(scores) if scores else 0.0

        # Find best and worst
        best_result = max(successful, key=lambda r: r.get("evaluation", {}).get("overall_score", 0.0)) if successful else None
        worst_result = min(successful, key=lambda r: r.get("evaluation", {}).get("overall_score", 0.0)) if successful else None

        # Keep error analysis simple: surface failed queries and low-scoring criteria.
        low_score_threshold = 0.5
        low_scoring_queries = [
            {
                "query": r.get("query", ""),
                "score": r.get("evaluation", {}).get("overall_score", 0.0)
            }
            for r in successful
            if r.get("evaluation", {}).get("overall_score", 0.0) < low_score_threshold
        ]
        failed_queries = [
            {
                "query": r.get("query", ""),
                "error": r.get("error", "Unknown error")
            }
            for r in failed
        ]

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_queries": total_queries,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / total_queries if total_queries > 0 else 0.0
            },
            "scores": {
                "overall_average": avg_overall,
                "by_criterion": avg_criterion_scores
            },
            "evidence": {
                "average_sources_per_query": avg_sources,
                "min_sources": min(source_counts) if source_counts else 0,
                "max_sources": max(source_counts) if source_counts else 0,
                "strength_counts": evidence_strength_counts,
            },
            "best_result": {
                "query": best_result.get("query", "") if best_result else "",
                "score": best_result.get("evaluation", {}).get("overall_score", 0.0) if best_result else 0.0
            } if best_result else None,
            "worst_result": {
                "query": worst_result.get("query", "") if worst_result else "",
                "score": worst_result.get("evaluation", {}).get("overall_score", 0.0) if worst_result else 0.0
            } if worst_result else None,
            "error_analysis": {
                "low_scoring_queries": low_scoring_queries,
                "failed_queries": failed_queries,
            },
            "detailed_results": self.results
        }

        return report

    def _save_results(self, report: Dict[str, Any]):
        """
        Save evaluation results to file.

        TODO: YOUR CODE HERE
        - Save detailed results
        - Generate visualizations
        - Create summary report
        """
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"evaluation_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Evaluation results saved to {results_file}")

        # Save summary
        summary_file = output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            summary = report.get("summary", {})
            f.write(f"Total Queries: {summary.get('total_queries', 0)}\n")
            f.write(f"Successful: {summary.get('successful', 0)}\n")
            f.write(f"Failed: {summary.get('failed', 0)}\n")
            f.write(f"Success Rate: {summary.get('success_rate', 0.0):.2%}\n\n")

            scores = report.get("scores", {})
            f.write(f"Overall Average Score: {scores.get('overall_average', 0.0):.3f}\n\n")

            evidence = report.get("evidence", {})
            f.write("Evidence Summary:\n")
            f.write(f"  Average Sources per Query: {evidence.get('average_sources_per_query', 0.0):.2f}\n")
            f.write(f"  Min Sources: {evidence.get('min_sources', 0)}\n")
            f.write(f"  Max Sources: {evidence.get('max_sources', 0)}\n")
            f.write(f"  Evidence Strength Counts: {evidence.get('strength_counts', {})}\n\n")

            f.write("Scores by Criterion:\n")
            for criterion, score in scores.get("by_criterion", {}).items():
                f.write(f"  {criterion}: {score:.3f}\n")

            error_analysis = report.get("error_analysis", {})
            f.write("\nError Analysis:\n")
            f.write(f"  Low-scoring queries: {len(error_analysis.get('low_scoring_queries', []))}\n")
            f.write(f"  Failed queries: {len(error_analysis.get('failed_queries', []))}\n")

        self.logger.info(f"Summary saved to {summary_file}")

    def export_for_report(self, output_path: str = "outputs/report_data.json"):
        """
        Export data formatted for inclusion in technical report.

        """
        if not self.results:
            self.logger.warning("No results to export")
            return
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        # Format data for report
        report_data = {
            "evaluation_date": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "results": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Report data exported to {output_path}")


async def example_simple_evaluation():
    """
    Example 1: Simple evaluation without orchestrator
    Tests the evaluation pipeline with mock responses
    
    Usage:
        import asyncio
        from src.evaluation.evaluator import example_simple_evaluation
        asyncio.run(example_simple_evaluation())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("=" * 70)
    print("EXAMPLE 1: Simple Evaluation (No Orchestrator)")
    print("=" * 70)
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create test queries in memory (no file needed)
    test_queries = [
        {
            "query": "What is the capital of France?",
            "ground_truth": "Paris is the capital of France."
        },
        {
            "query": "What are the benefits of exercise?",
            "ground_truth": "Exercise improves physical health, mental wellbeing, and reduces disease risk."
        }
    ]
    
    # Save test queries temporarily
    test_file = Path("data/test_queries_example.json")
    test_file.parent.mkdir(exist_ok=True)
    with open(test_file, 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    # Initialize evaluator without orchestrator
    evaluator = SystemEvaluator(config, orchestrator=None)
    
    print("\nRunning evaluation on test queries...")
    print("Note: Using deterministic fallback responses since no orchestrator is connected\n")
    
    # Run evaluation
    report = await evaluator.evaluate_system(str(test_file))
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal Queries: {report['summary']['total_queries']}")
    print(f"Successful: {report['summary']['successful']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Overall Average Score: {report['scores']['overall_average']:.3f}\n")
    
    print("Scores by Criterion:")
    for criterion, score in report['scores']['by_criterion'].items():
        print(f"  {criterion}: {score:.3f}")
    
    print(f"\nDetailed results saved to outputs/")
    
    # Clean up
    test_file.unlink()


async def example_with_orchestrator():
    """
    Example 2: Evaluation with orchestrator
    Shows how to connect the evaluator to your multi-agent system
    
    Usage:
        import asyncio
        from src.evaluation.evaluator import example_with_orchestrator
        asyncio.run(example_with_orchestrator())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("=" * 70)
    print("EXAMPLE 2: Evaluation with Orchestrator")
    print("=" * 70)
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize orchestrator
    try:
        from src.autogen_orchestrator import AutoGenOrchestrator
        orchestrator = AutoGenOrchestrator(config)
        print("\nOrchestrator initialized successfully")
    except Exception as e:
        print(f"\nCould not initialize orchestrator: {e}")
        print("This example requires a working orchestrator implementation")
        return
    
    # Create test queries
    test_queries = [
        {
            "query": "What are the key principles of accessible user interface design?",
            "ground_truth": "Key principles include perceivability, operability, understandability, and robustness."
        }
    ]
    
    test_file = Path("data/test_queries_orchestrator.json")
    test_file.parent.mkdir(exist_ok=True)
    with open(test_file, 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    # Initialize evaluator with orchestrator
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)
    
    print("\nRunning evaluation with real orchestrator...")
    print("This will actually query your multi-agent system\n")
    
    # Run evaluation
    report = await evaluator.evaluate_system(str(test_file))
    
    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTotal Queries: {report['summary']['total_queries']}")
    print(f"Overall Average Score: {report['scores']['overall_average']:.3f}\n")
    
    print("Scores by Criterion:")
    for criterion, score in report['scores']['by_criterion'].items():
        print(f"  {criterion}: {score:.3f}")
    
    # Show detailed result for first query
    if report['detailed_results']:
        result = report['detailed_results'][0]
        print("\n" + "=" * 70)
        print("DETAILED RESULT (First Query)")
        print("=" * 70)
        print(f"\nQuery: {result['query']}")
        print(f"\nResponse: {result['response'][:200]}...")
        print(f"\nOverall Score: {result['evaluation']['overall_score']:.3f}")
    
    print(f"\nFull results saved to outputs/")
    
    # Clean up
    test_file.unlink()


# For direct execution
if __name__ == "__main__":
    import asyncio
    
    print("Running SystemEvaluator Examples\n")
    
    # Run example 1
    asyncio.run(example_simple_evaluation())
    
    print("\n\n")
    
    # Run example 2 (if orchestrator is available)
    asyncio.run(example_with_orchestrator())
