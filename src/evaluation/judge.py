"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)
    
    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )
    
    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        # This includes: provider, name, temperature, max_tokens
        self.model_config = config.get("models", {}).get("judge", {})

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        # Each criterion has: name, weight, description
        self.criteria = config.get("evaluation", {}).get("criteria", [])
        
        # Initialize the configured judge client, with Groq as the beginner-friendly fallback.
        self.provider = self.model_config.get("provider", "groq")
        self.client = None
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not Groq:
                self.logger.warning("groq package not installed; judge API calls are unavailable")
            elif not api_key:
                self.logger.warning("GROQ_API_KEY not found in environment")
            self.client = Groq(api_key=api_key) if Groq and api_key else None
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if api_key and OpenAI:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
            elif os.getenv("GROQ_API_KEY") and Groq:
                self.logger.warning("Configured judge provider unavailable; falling back to Groq")
                self.provider = "groq"
                self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            else:
                self.logger.warning("No judge API key found in environment")
        
        self.logger.info(f"LLMJudge initialized with {len(self.criteria)} criteria")
 
    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None,
        evidence_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            self.logger.info(f"Evaluating criterion: {criterion_name}")

            # TODO: Implement actual LLM judging
            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth,
                evidence_context=evidence_context,
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight

        # Calculate overall score
        results["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0.0

        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        evidence_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using Groq API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        # Create judge prompt
        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth,
            evidence_context=evidence_context,
        )

        # Call LLM API to get judgment
        try:
            judgment = await self._call_judge_llm(prompt)
            score_value, reasoning = self._parse_judgment(judgment)
            
            score = {
                "score": score_value,  # 0-1 scale
                "reasoning": reasoning,
                "criterion": criterion_name
            }
        except Exception as e:
            self.logger.error(f"Error judging criterion {criterion_name}: {e}")
            score = {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "criterion": criterion_name
            }

        return score

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str],
        evidence_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a prompt for the judge LLM.

        TODO: YOUR CODE HERE
        - Create effective judge prompts
        - Include clear scoring rubric
        - Provide examples if helpful
        """
        prompt = f"""You are an expert evaluator. Evaluate the following response based on the criterion: {criterion_name}.

Criterion Description: {description}

Scoring Rubric:
- 1.0 = excellent, directly satisfies the criterion with no important gaps
- 0.7 = good, mostly satisfies the criterion with minor gaps
- 0.4 = partial, addresses the criterion but misses important details
- 0.0 = poor, fails the criterion or contains serious problems

Query: {query}

Response:
{response}
"""

        sources = self._filter_valid_sources(sources or [])

        if sources:
            source_preview = "\n".join(
                f"- [{source.get('tool', 'unknown')}] {source.get('title', 'Untitled')}: {source.get('url', '')}"
                for source in sources[:10]
            )
            prompt += f"\n\nSources Used ({len(sources)}):\n{source_preview}"

        if evidence_context:
            prompt += "\n\nStructured Evidence Context:"
            prompt += f"\nEvidence Strength: {evidence_context.get('evidence_strength', 'Unknown')}"
            prompt += f"\nTool Trace Summary:\n{self._format_tool_traces(evidence_context.get('tool_traces', []))}"
            prompt += f"\nCitation Mapping:\n{self._format_citation_mapping(evidence_context.get('citation_mapping', []))}"
            prompt += f"\nAgent Trace Summary:\n{self._format_agent_trace_summary(evidence_context.get('agent_traces', {}))}"

        if ground_truth:
            prompt += f"\n\nExpected Response:\n{ground_truth}"

        prompt += """

Please evaluate the response on a scale of 0.0 to 1.0 for this criterion.
Return only valid JSON in the following format:
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your score>"
}
"""

        return prompt

    def _filter_valid_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prevent prompt/rubric/agent text from entering judge evidence context."""
        import re
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
            if url and not re.match(r'^https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/|$)', url):
                continue
            if any(term in combined for term in ["planner", "writer", "critic", "researcher", "score:", "instructions"]):
                continue
            key = (url or paper_id or title).lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(source)
        return filtered

    def _format_tool_traces(self, tool_traces: List[Dict[str, Any]]) -> str:
        """Format tool execution traces compactly for the judge prompt."""
        if not tool_traces:
            return "- No tool traces captured"
        lines = []
        for trace in tool_traces[:8]:
            result_titles = [
                result.get("title", "Untitled")
                for result in trace.get("results", [])[:3]
            ]
            lines.append(
                f"- #{trace.get('order', '?')} {trace.get('tool', 'unknown')} "
                f"query={trace.get('query', '')!r}; results={result_titles}"
            )
        return "\n".join(lines)

    def _format_citation_mapping(self, citation_mapping: List[Dict[str, Any]]) -> str:
        """Format claim-to-source mappings for evidence-quality judging."""
        if not citation_mapping:
            return "- No citation mapping captured"
        lines = []
        for mapping in citation_mapping[:8]:
            source_titles = [
                source.get("title", "Untitled")
                for source in mapping.get("sources", [])[:2]
            ]
            lines.append(
                f"- Claim {mapping.get('claim_id')}: {mapping.get('claim', '')[:160]} "
                f"=> sources={source_titles}"
            )
        return "\n".join(lines)

    def _format_agent_trace_summary(self, agent_traces: Dict[str, Any]) -> str:
        """Format agent trace counts without overwhelming the judge prompt."""
        if not agent_traces:
            return "- No agent traces captured"
        return "\n".join(
            f"- {agent}: {len(actions)} message/action(s)"
            for agent, actions in agent_traces.items()
        )

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call LLM API to get judgment.
        Uses model configuration from config.yaml (models.judge section).
        """
        if not self.client:
            raise ValueError("Judge client not initialized. Check API key environment variables.")
        
        try:
            # Load model settings from config.yaml (models.judge)
            model_name = self.model_config.get("name", "llama-3.1-8b-instant")
            if self.provider == "groq" and model_name.startswith("openai/"):
                model_name = os.getenv("JUDGE_MODEL", "llama-3.1-8b-instant")
            temperature = self.model_config.get("temperature", 0.3)
            max_tokens = self.model_config.get("max_tokens", 1024)
            
            self.logger.debug(f"Calling judge API with provider={self.provider}, model={model_name}")
            
            # Use the same chat-completions shape for Groq and OpenAI-compatible endpoints.
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide your evaluations in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            response = chat_completion.choices[0].message.content
            self.logger.debug(f"Received response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling Groq API: {e}")
            raise

    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse LLM judgment response.
        
        """
        try:
            # Clean up the response - remove markdown code blocks if present
            judgment_clean = judgment.strip()
            if judgment_clean.startswith("```json"):
                judgment_clean = judgment_clean[7:]
            elif judgment_clean.startswith("```"):
                judgment_clean = judgment_clean[3:]
            if judgment_clean.endswith("```"):
                judgment_clean = judgment_clean[:-3]
            judgment_clean = judgment_clean.strip()

            # Parse JSON
            result = json.loads(judgment_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Raw judgment: {judgment[:200]}")
            # Simple fallback: recover a numeric score if the model added prose.
            import re
            match = re.search(r'"?score"?\s*[:=]\s*([01](?:\.\d+)?)', judgment)
            if match:
                score = max(0.0, min(1.0, float(match.group(1))))
                return score, judgment[:500]
            return 0.0, f"Error parsing judgment: Invalid JSON"
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            return 0.0, f"Error parsing judgment: {str(e)}"



async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with LLMJudge
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_basic_evaluation
        asyncio.run(example_basic_evaluation())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    # Test case (similar to Lab 5)
    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)
    
    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"
    
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Ground Truth: {ground_truth}\n")
    
    # Evaluate
    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )
    
    print(f"Overall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result['criterion_scores'].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    Reasoning: {score_data['reasoning'][:100]}...")
        print()


async def example_compare_responses():
    """
    Example 2: Compare multiple responses
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_compare_responses
        asyncio.run(example_compare_responses())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    print("=" * 70)
    print("EXAMPLE 2: Compare Multiple Responses")
    print("=" * 70)
    
    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."
    
    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]
    
    print(f"\nQuery: {query}\n")
    print(f"Ground Truth: {ground_truth}\n")
    
    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n{'='*70}")
        print(f"Response {i}:")
        print(f"{response}")
        print(f"{'='*70}")
        
        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )
        
        results.append(result)
        
        print(f"\nOverall Score: {result['overall_score']:.3f}")
        print("\nCriterion Scores:")
        for criterion, score_data in result['criterion_scores'].items():
            print(f"  {criterion}: {score_data['score']:.3f}")
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result['overall_score']:.3f}")
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['overall_score'])
    print(f"\nBest Response: Response {best_idx + 1}")


# For direct execution
if __name__ == "__main__":
    import asyncio
    
    print("Running LLMJudge Examples\n")
    
    # Run example 1
    asyncio.run(example_basic_evaluation())
    
    print("\n\n")
    
    # Run example 2
    asyncio.run(example_compare_responses())
