"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Dict, Any
import yaml
import logging
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

# Load environment variables
load_dotenv()

class CLI:
    """
    Command-line interface for the research assistant.

    TODO: YOUR CODE HERE
    - Implement interactive prompt loop
    - Display agent traces clearly
    - Show citations and sources
    - Indicate safety events (blocked/sanitized)
    - Handle user commands (help, quit, clear, etc.)
    - Format output nicely
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize CLI.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize AutoGen orchestrator
        try:
            self.orchestrator = AutoGenOrchestrator(self.config)
            self.logger = logging.getLogger("cli")
            self.logger.info("AutoGen orchestrator initialized successfully")
        except Exception as e:
            self.logger = logging.getLogger("cli")
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

        self.running = True
        self.query_count = 0

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

    async def run(self):
        """
        Main CLI loop.

        TODO: YOUR CODE HERE
        - Implement interactive loop
        - Handle user input
        - Process queries through orchestrator
        - Display results
        - Handle errors gracefully
        """
        self._print_welcome()

        while self.running:
            try:
                # Get user input
                query = input("\nEnter your research query (or 'help' for commands): ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._print_goodbye()
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif query.lower() == 'stats':
                    self._print_stats()
                    continue

                # Process query
                print("\n" + "=" * 70)
                print("Processing your query...")
                print("=" * 70)
                
                try:
                    # Process through orchestrator (synchronous call, not async)
                    result = self.orchestrator.process_query(query)
                    self.query_count += 1
                    
                    # Display result
                    self._display_result(result)
                    
                except Exception as e:
                    print(f"\nError processing query: {e}")
                    logging.exception("Error processing query")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self._print_goodbye()
                break
            except Exception as e:
                print(f"\nError: {e}")
                logging.exception("Error in CLI loop")

    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 70)
        print(f"  {self.config['system']['name']}")
        print(f"  Topic: {self.config['system']['topic']}")
        print("=" * 70)
        print("\nWelcome! Ask me anything about your research topic.")
        print("Type 'help' for available commands, or 'quit' to exit.\n")

    def _print_help(self):
        """Print help message."""
        print("\nAvailable commands:")
        print("  help    - Show this help message")
        print("  clear   - Clear the screen")
        print("  stats   - Show system statistics")
        print("  quit    - Exit the application")
        print("\nOr enter a research query to get started!")

    def _print_goodbye(self):
        """Print goodbye message."""
        print("\nThank you for using the Multi-Agent Research Assistant!")
        print("Goodbye!\n")

    def _clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    def _print_stats(self):
        """Print system statistics."""
        print("\nSystem Statistics:")
        print(f"  Queries processed: {self.query_count}")
        print(f"  System: {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic: {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Model: {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")
        if hasattr(self.orchestrator, "safety_manager"):
            stats = self.orchestrator.safety_manager.get_safety_stats()
            print(f"  Safety events: {stats.get('total_events', 0)}")
            print(f"  Safety violations: {stats.get('violations', 0)}")

    def _display_result(self, result: Dict[str, Any]):
        """Display query result with formatting."""
        print("\n" + "=" * 70)
        print("RESPONSE")
        print("=" * 70)

        # Check for errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return

        # Display response
        response = result.get("response", "")
        print(f"\n{response}\n")

        # Extract and display citations from conversation
        metadata = result.get("metadata", {})
        citations = metadata.get("citations") or self._extract_citations(result)
        if citations:
            print("\n" + "-" * 70)
            print("📚 CITATIONS")
            print("-" * 70)
            for i, citation in enumerate(citations, 1):
                print(f"[{i}] {citation}")

        self._display_evidence(metadata)

        # Display metadata
        if metadata:
            print("\n" + "-" * 70)
            print("📊 METADATA")
            print("-" * 70)
            print(f"  • Messages exchanged: {metadata.get('num_messages', 0)}")
            print(f"  • Sources gathered: {metadata.get('num_sources', 0)}")
            print(f"  • Agents involved: {', '.join(metadata.get('agents_involved', []))}")
            # TODO: Display safety events and refusal/sanitization status here
            # Suggested implementation:
            # - Read safety metadata returned by the orchestrator
            # - Print which policy category was triggered
            # - Show whether the response was refused or sanitized
            safety_events = metadata.get("safety_events", [])
            if safety_events:
                print(f"  • Safety action: {metadata.get('safety_action', 'allow')}")
                print(f"  • Safety events: {len(safety_events)}")
                latest_event = safety_events[-1]
                for violation in latest_event.get("violations", []):
                    print(f"    - {violation.get('validator', violation.get('category', 'policy'))}: {violation.get('reason', 'No reason provided')}")
            if metadata.get("refused"):
                print("  • Response was refused by safety policy")
            elif metadata.get("sanitized"):
                print("  • Response was sanitized by safety policy")

            evaluation = metadata.get("evaluation")
            if evaluation:
                print(f"  • Evaluation score: {evaluation.get('overall_score', 0.0):.3f}")

        # Display conversation summary if verbose mode
        if self._should_show_traces():
            self._display_conversation_summary(result.get("conversation_history", []))

        print("=" * 70 + "\n")

    def _display_evidence(self, metadata: Dict[str, Any]):
        """Display tool traces, grouped sources, and citation mappings."""
        if not metadata:
            return

        print("\n" + "-" * 70)
        print("🔎 EVIDENCE")
        print("-" * 70)
        print(f"Evidence Strength: {metadata.get('evidence_strength', 'Unknown')}")

        tool_traces = metadata.get("tool_traces", [])
        if tool_traces:
            print("\nResearcher Tool Calls:")
            for trace in tool_traces:
                print(f"  [{trace.get('order', '?')}] {trace.get('tool', 'unknown')} query=\"{trace.get('query', '')}\"")
                print(f"      {trace.get('summary', '')}")
                for result in trace.get("results", [])[:3]:
                    print(f"      - {result.get('title', 'Untitled')} ({result.get('url', '')})")

        source_groups = metadata.get("source_groups", {})
        if source_groups:
            print("\nSources by Tool:")
            for tool, sources in source_groups.items():
                print(f"  {tool}: {len(sources)} source(s)")
                for source in sources[:5]:
                    print(f"    - {source.get('title', 'Untitled')} ({source.get('url', '')})")

        citation_mapping = metadata.get("citation_mapping", [])
        if citation_mapping:
            print("\nEvidence → Citation Mapping:")
            for mapping in citation_mapping[:5]:
                support = ", ".join(
                    source.get("title", "Untitled")
                    for source in mapping.get("sources", [])[:2]
                )
                print(f"  Claim {mapping.get('claim_id')}: {mapping.get('claim', '')[:120]}...")
                print(f"    Supports: {support or 'No mapped source'}")
    
    def _extract_citations(self, result: Dict[str, Any]) -> list:
        """Extract citations/URLs from conversation history."""
        citations = []
        
        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")
            
            # Find URLs in content
            import re
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
            
            for url in urls:
                if url not in citations:
                    citations.append(url)
        
        return citations[:10]  # Limit to top 10

    def _should_show_traces(self) -> bool:
        """Check if agent traces should be displayed."""
        # Check config for verbose mode
        return self.config.get("ui", {}).get("verbose", False)

    def _display_conversation_summary(self, conversation_history: list):
        """Display a summary of the agent conversation."""
        if not conversation_history:
            return
            
        print("\n" + "-" * 70)
        print("🔍 CONVERSATION SUMMARY")
        print("-" * 70)
        
        for i, msg in enumerate(conversation_history, 1):
            agent = msg.get("source", "Unknown")
            content = msg.get("content", "")
            
            # Truncate long content
            preview = content[:150] + "..." if len(content) > 150 else content
            preview = preview.replace("\n", " ")
            
            print(f"\n{i}. {agent}:")
            print(f"   {preview}")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Run CLI
    cli = CLI(config_path=args.config)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
