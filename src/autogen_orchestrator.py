"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
"""

import logging
import asyncio
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.agents.autogen_agents import create_research_team
from src.guardrails import SafetyManager
from src.tools.citation_tool import CitationTool
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    
    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        
        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        
        self.logger.info("Research team created successfully")
        
        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

        # Student implementation: keep safety centralized while preserving AutoGen flow.
        self.safety_manager = SafetyManager(config)

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")
        
        try:
            # Run the async query processing
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, 
                        self._process_query_async(query, max_rounds)
                    ).result()
            else:
                result = loop.run_until_complete(self._process_query_async(query, max_rounds))
            
            self.logger.info("Query processing complete")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }
    
    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Async implementation of query processing.
        
        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds
            
        Returns:
            Dictionary containing results
        """
        input_safety = self.safety_manager.check_input_safety(query)
        safety_events = self.safety_manager.get_safety_events()
        if not input_safety.get("safe", True):
            return {
                "query": query,
                "response": input_safety.get("message", "I cannot process this request due to safety policies."),
                "conversation_history": [],
                "metadata": {
                    "num_messages": 0,
                    "num_sources": 0,
                    "sources": [],
                    "citations": [],
                    "plan": "",
                    "research_findings": [],
                    "critique": "",
                    "agents_involved": [],
                    "safety_events": safety_events,
                    "safety_action": input_safety.get("action", "refuse"),
                    "refused": True,
                }
            }

        query_to_process = input_safety.get("query", query)

        # Create task message
        task_message = f"""Research Query: {query_to_process}

Please work together to answer this query comprehensively:
1. Planner: Create a research plan
2. Researcher: Gather evidence from web and academic sources
3. Writer: Synthesize findings into a well-cited response
4. Critic: Evaluate the quality and provide feedback"""
        
        # Run the team
        result = await self.team.run(task=task_message)
        
        # Extract conversation history
        messages = []
        for message in result.messages:
            messages.append(self._message_to_dict(message))

        recovered_tool_outputs = await self._recover_missing_tool_outputs(messages)
        if recovered_tool_outputs:
            messages.extend(recovered_tool_outputs)
        
        # Extract final response
        final_response = ""
        if messages:
            # Get the last message from Writer or Critic
            for msg in reversed(messages):
                if msg.get("source") in ["Writer", "Critic"]:
                    final_response = msg.get("content", "")
                    break
        
        # If no response found, use the last message
        if not final_response and messages:
            final_response = messages[-1].get("content", "")
        
        result_data = self._extract_results(query, messages, final_response)
        sources = result_data.get("metadata", {}).get("sources", [])
        output_safety = self.safety_manager.check_output_safety(
            result_data.get("response", ""),
            sources=sources
        )
        result_data["response"] = output_safety.get("response", result_data.get("response", ""))
        result_data["metadata"]["safety_events"] = self.safety_manager.get_safety_events()
        result_data["metadata"]["safety_action"] = output_safety.get("action", "allow")
        result_data["metadata"]["sanitized"] = (
            output_safety.get("action") == "sanitize" and not output_safety.get("safe", True)
        )
        result_data["metadata"]["refused"] = (
            output_safety.get("action") == "refuse" and not output_safety.get("safe", True)
        )
        result_data["metadata"]["violations"] = output_safety.get("violations", [])
        self._export_session(result_data)
        return result_data

    def _message_to_dict(self, message: Any) -> Dict[str, Any]:
        """Preserve text plus AutoGen tool execution metadata for evidence extraction."""
        content = message.content if hasattr(message, 'content') else str(message)
        content_text, tool_name = self._stringify_message_content(content)
        return {
            "source": getattr(message, "source", "Unknown"),
            "content": content_text,
            "message_type": type(message).__name__,
            "tool_name": tool_name,
        }

    def _stringify_message_content(self, content: Any) -> tuple[str, Optional[str]]:
        """Convert AutoGen message content, including tool results, into searchable text."""
        if isinstance(content, str):
            return content, self._extract_tool_name(content)

        parts = []
        tool_name = None
        items = content if isinstance(content, list) else [content]
        for item in items:
            item_name = getattr(item, "name", None)
            item_content = getattr(item, "content", None)
            if isinstance(item, dict):
                item_name = item.get("name", item.get("tool", item_name))
                item_content = item.get("content", item.get("result", item_content))
            if item_name:
                tool_name = item_name
                parts.append(f"Tool: {item_name}")
            if item_content is not None:
                parts.append(str(item_content))
            else:
                parts.append(str(item))

        content_text = "\n".join(parts)
        return content_text, tool_name or self._extract_tool_name(content_text)

    def _extract_tool_name(self, content: str) -> Optional[str]:
        """Extract web_search/paper_search from tool metadata or serialized tool output."""
        match = re.search(r'"name"\s*:\s*"(web_search|paper_search)"', content)
        if match:
            return match.group(1)
        match = re.search(r'\bTool:\s*(web_search|paper_search)\b', content)
        if match:
            return match.group(1)
        match = re.search(r'\b(web_search|paper_search)\b', content)
        return match.group(1) if match else None

    async def _recover_missing_tool_outputs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recover evidence when vLLM emitted tool-call text but AutoGen did not persist results."""
        if self._extract_sources(messages):
            return []

        tool_calls = self._extract_researcher_tool_calls(messages)
        if not tool_calls:
            return []

        recovered_messages = []
        for call in tool_calls:
            tool_name = call.get("tool")
            arguments = call.get("arguments", {})
            query = arguments.get("query", "")
            if not query:
                continue
            try:
                self.logger.debug(f"Recovering missing {tool_name} output for query: {query}")
                if tool_name == "web_search":
                    output = await asyncio.to_thread(
                        web_search,
                        query=query,
                        provider=arguments.get("provider", "tavily"),
                        max_results=int(arguments.get("max_results", 5)),
                    )
                elif tool_name == "paper_search":
                    output = await asyncio.to_thread(
                        paper_search,
                        query=query,
                        max_results=int(arguments.get("max_results", 5)),
                        year_from=arguments.get("year_from"),
                    )
                else:
                    continue
            except Exception as exc:
                self.logger.warning(f"Could not recover {tool_name} output: {exc}")
                continue

            self.logger.debug(f"Raw recovered {tool_name} output before filtering:\n{output[:2000]}")
            recovered_messages.append({
                "source": "Researcher",
                "content": f"Tool: {tool_name}\n{output}",
                "message_type": "RecoveredToolExecution",
                "tool_name": tool_name,
            })

        if recovered_messages:
            self.logger.info(f"Recovered {len(recovered_messages)} missing Researcher tool output(s)")
        return recovered_messages

    def _extract_researcher_tool_calls(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse Researcher-emitted JSON tool calls for web_search and paper_search."""
        tool_calls = []
        seen = set()
        for msg in messages:
            if msg.get("source") != "Researcher":
                continue
            content = msg.get("content", "")
            blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
            if not blocks:
                blocks = re.findall(r"\{[^{}]*(?:\"tool\"|\"name\")\s*:\s*\"(?:web_search|paper_search)\".*?\}", content, flags=re.DOTALL)
            for block in blocks:
                try:
                    payload = json.loads(block)
                except json.JSONDecodeError:
                    self.logger.debug(f"Skipping unparsable Researcher tool call: {block[:300]}")
                    continue
                tool_name = payload.get("tool") or payload.get("name")
                if tool_name not in {"web_search", "paper_search"}:
                    continue
                arguments = payload.get("arguments", {})
                key = (tool_name, json.dumps(arguments, sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                tool_calls.append({"tool": tool_name, "arguments": arguments})
        return tool_calls

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""
        
        for msg in messages:
            source = msg.get("source", "")
            content = self._normalize_tool_output_text(msg.get("content", ""))
            
            if source == "Planner" and not plan:
                plan = content
            
            elif source == "Researcher":
                research_findings.append(content)
            
            elif source == "Critic":
                critique = content
        
        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\n1.") + finding.count("\n2.") + finding.count("\n3.")
        
        sources = self._extract_sources(messages)
        citations = self._format_citations(sources)
        agent_traces = self._extract_agent_traces(messages)
        tool_traces = self._extract_tool_traces(messages, sources)
        citation_mapping = self._build_citation_mapping(final_response, sources)
        source_groups = self._group_sources_by_tool(sources)
        evidence_strength = self._calculate_evidence_strength(sources, tool_traces)
        
        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()
        
        return {
            "query": query,
            "response": final_response,
            "sources": sources,
            "tool_calls": tool_traces,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": len(sources) if sources else num_sources,
                "sources": sources,
                "citations": citations,
                "source_groups": source_groups,
                "tool_traces": tool_traces,
                "citation_mapping": citation_mapping,
                "evidence_strength": evidence_strength,
                "agent_traces": agent_traces,
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
            }
        }

    def _extract_sources(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique source records from all tool result text in agent messages."""
        sources = []
        seen_keys = set()
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')

        for msg in messages:
            if not self._is_researcher_tool_output_message(msg):
                continue
            content = self._normalize_tool_output_text(msg.get("content", ""))
            tool_type = msg.get("tool_name") or self._infer_tool_type(content)
            if not self._is_tool_output_content(content, tool_type):
                continue
            self.logger.debug(f"Raw {tool_type} output before source filtering:\n{content[:2000]}")
            sources.extend(self._parse_numbered_tool_sources(content, tool_type, seen_keys))

            for url in url_pattern.findall(content):
                clean_url = url.rstrip(").,;")
                inferred_tool = tool_type or "web_search"
                source = {
                    "type": "paper" if inferred_tool == "paper_search" else "webpage",
                    "tool": inferred_tool,
                    "tool_type": inferred_tool,
                    "title": self._guess_source_title(content, clean_url),
                    "url": clean_url,
                    "site_name": clean_url.split("/")[2] if "://" in clean_url else "",
                    "snippet": self._guess_source_snippet(content, clean_url),
                }
                self._append_source_if_unique(sources, source, seen_keys)

        filtered_sources = self._filter_valid_sources(sources)
        self.logger.info(f"Extracted {len(filtered_sources)} source(s) from Researcher tool output")
        return filtered_sources

    def _is_researcher_tool_output_message(self, msg: Dict[str, Any]) -> bool:
        """Accept Researcher-owned tool execution output, not Planner/Writer/Critic text."""
        source = msg.get("source", "")
        content = msg.get("content", "")
        message_type = msg.get("message_type", "")
        tool_name = msg.get("tool_name")
        if source == "Researcher" and (tool_name or self._infer_tool_type(content)):
            return True
        if "ToolCallExecution" in message_type and tool_name in {"web_search", "paper_search"}:
            return True
        return False

    def _normalize_tool_output_text(self, content: str) -> str:
        """Normalize serialized tool result strings so parsers can read result lines."""
        match = re.search(r"content=(['\"])(.*?)\1,\s*name=(['\"])(web_search|paper_search)\3", content, flags=re.DOTALL)
        if match:
            return f"Tool: {match.group(4)}\n{match.group(2).encode('utf-8').decode('unicode_escape')}"
        return content.replace("\\n", "\n")

    def _is_tool_output_content(self, content: str, tool_type: Optional[str]) -> bool:
        """Only accept actual web_search/paper_search result blocks as source input."""
        lowered = content.lower()
        has_tool_result_phrase = (
            "found " in lowered
            and ("web search result" in lowered or "academic paper" in lowered)
        )
        has_tool_marker = "web_search" in lowered or "paper_search" in lowered
        has_url = bool(re.search(r'https?://', content))
        has_serialized_result = "tool:" in lowered or "functionexecutionresult" in lowered
        return bool(tool_type and has_url and (has_tool_result_phrase or has_serialized_result or has_tool_marker))

    def _infer_tool_type(self, content: str) -> Optional[str]:
        """Infer which research tool produced a flattened result block."""
        lowered = content.lower()
        if "academic papers" in lowered or "semantic scholar" in lowered or "paper_search" in lowered:
            return "paper_search"
        if "web search results" in lowered or "web_search" in lowered or "brave" in lowered or "tavily" in lowered:
            return "web_search"
        return None

    def _parse_numbered_tool_sources(
        self,
        content: str,
        tool_type: Optional[str],
        seen_keys: set,
    ) -> List[Dict[str, Any]]:
        """Parse formatted web_search/paper_search result blocks into source records."""
        parsed_sources = []
        current = None
        current_tool = tool_type

        for raw_line in content.splitlines():
            line = raw_line.strip()
            line_tool = self._infer_tool_type(line)
            if line_tool:
                if current:
                    self._append_source_if_unique(parsed_sources, current, seen_keys)
                    current = None
                current_tool = line_tool
                continue

            numbered = re.match(r'^(\d+)\.\s+(.+)$', line)
            if numbered:
                if current:
                    self._append_source_if_unique(parsed_sources, current, seen_keys)
                inferred_tool = current_tool or self._infer_tool_type(content) or "unknown"
                current = {
                    "type": "paper" if inferred_tool == "paper_search" else "webpage",
                    "tool": inferred_tool or "unknown",
                    "tool_type": inferred_tool or "unknown",
                    "title": numbered.group(2).strip(),
                    "url": "",
                    "snippet": "",
                    "authors": [],
                    "authors_text": "",
                    "year": "",
                    "citation_count": "",
                }
                continue

            if not current:
                continue

            if line.startswith("URL:"):
                current["url"] = line.replace("URL:", "", 1).strip().rstrip(").,;")
            elif line.startswith("Authors:"):
                authors_text = line.replace("Authors:", "", 1).strip()
                current["authors_text"] = authors_text
                current["authors"] = [
                    {"name": author.strip()}
                    for author in authors_text.replace(" et al.", "").split(",")
                    if author.strip()
                ]
            elif line.startswith("Year:"):
                year_text = line.replace("Year:", "", 1).strip()
                year_match = re.search(r'\b(19|20)\d{2}\b', year_text)
                citation_match = re.search(r'Citations:\s*(\d+)', year_text)
                current["year"] = int(year_match.group(0)) if year_match else year_text
                current["citation_count"] = int(citation_match.group(1)) if citation_match else ""
            elif line.startswith("Abstract:"):
                current["snippet"] = line.replace("Abstract:", "", 1).strip()
            elif line and not line.startswith(("Published:", "Venue:")) and not current.get("snippet"):
                current["snippet"] = line

        if current:
            self._append_source_if_unique(parsed_sources, current, seen_keys)

        return parsed_sources

    def _append_source_if_unique(
        self,
        sources: List[Dict[str, Any]],
        source: Dict[str, Any],
        seen_keys: set,
    ):
        """Deduplicate by URL when available, otherwise normalized title."""
        if not self._is_valid_source(source):
            return
        if source.get("tool") in ("", "unknown", None):
            source["tool"] = self._infer_tool_from_source(source)
            source["type"] = "paper" if source["tool"] == "paper_search" else "webpage"
        source["tool_type"] = source.get("tool")
        if source.get("authors_text") and not source.get("author"):
            source["author"] = source.get("authors_text")
        key = (source.get("url") or source.get("title", "")).strip().lower()
        if not key or key in seen_keys:
            return
        seen_keys.add(key)
        sources.append(source)

    def _filter_valid_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply strict source rules and final URL/title deduplication."""
        filtered = []
        seen = set()
        for source in sources:
            if not self._is_valid_source(source):
                continue
            self._normalize_source_author(source)
            key = (source.get("url") or source.get("paper_id") or source.get("title", "")).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            filtered.append(source)
        return filtered

    def _is_valid_source(self, source: Dict[str, Any]) -> bool:
        """Reject prompt/rubric/reasoning text and keep only external tool evidence."""
        tool = source.get("tool")
        title = str(source.get("title", "")).strip()
        url = str(source.get("url", "")).strip()
        paper_id = str(source.get("paper_id", "")).strip()
        combined_text = f"{title} {source.get('snippet', '')}".lower()

        if tool not in {"web_search", "paper_search"}:
            return False
        if not url and not paper_id:
            return False
        if url and not self._is_valid_external_url(url):
            return False

        forbidden_terms = [
            "planner", "writer", "critic", "researcher",
            "analyze", "extract", "conclusion", "evaluation",
            "score:", "research plan", "system prompt", "instructions",
        ]
        if any(term in combined_text for term in forbidden_terms):
            return False
        return True

    def _is_valid_external_url(self, url: str) -> bool:
        """Validate that URL looks like an external web or academic source."""
        return bool(re.match(r'^https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/|$)', url))

    def _normalize_source_author(self, source: Dict[str, Any]):
        """Replace missing authors with the URL domain, or remove invalid sources upstream."""
        authors = source.get("authors")
        if authors:
            return
        url = source.get("url", "")
        if "://" not in url:
            return
        domain = url.split("/")[2]
        source["authors"] = [{"name": domain}]
        source["authors_text"] = domain
        source["author"] = domain
        if not source.get("year"):
            source["year"] = "n.d."

    def _infer_tool_from_source(self, source: Dict[str, Any]) -> str:
        """Classify fallback sources by URL/title when explicit tool context is absent."""
        text = f"{source.get('url', '')} {source.get('title', '')}".lower()
        if "semanticscholar" in text or "paper" in text or "doi.org" in text:
            return "paper_search"
        return "web_search"

    def _guess_source_title(self, content: str, url: str) -> str:
        """Use nearby result text as a readable title when tool output is flattened."""
        for line in content.splitlines():
            if url in line:
                continue
            stripped = line.strip()
            if stripped and re.match(r'^\d+\.', stripped):
                return stripped.split(".", 1)[1].strip()
        return url

    def _guess_source_snippet(self, content: str, url: str) -> str:
        """Find a nearby snippet line after a URL in flattened tool output."""
        lines = content.splitlines()
        for index, line in enumerate(lines):
            if url in line and index + 1 < len(lines):
                return lines[index + 1].strip()
        return ""

    def _format_citations(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Format extracted source records with the existing citation helper."""
        citation_tool = CitationTool(style="apa")
        return [citation_tool.format_citation(source) for source in sources]

    def _extract_agent_traces(self, messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
        """Group conversation messages by agent for trace display."""
        traces: Dict[str, List[Dict[str, str]]] = {}
        for index, msg in enumerate(messages, 1):
            agent = msg.get("source", "Unknown")
            traces.setdefault(agent, []).append({
                "action_type": "message",
                "order": index,
                "details": msg.get("content", "")[:500],
            })
        return traces

    def _extract_tool_traces(
        self,
        messages: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build visible Researcher tool traces from tool-call/result text."""
        traces = []
        order = 1
        for msg in messages:
            content = msg.get("content", "")
            tool_type = self._infer_tool_type(content)
            if msg.get("source") != "Researcher" and not tool_type:
                continue

            requested_tool = self._parse_tool_request(content)
            if requested_tool:
                traces.append({
                    "order": order,
                    "tool": requested_tool.get("tool", "unknown"),
                    "query": requested_tool.get("query", ""),
                    "results": [],
                    "summary": "Tool request emitted by Researcher",
                })
                order += 1

            if tool_type:
                matching_sources = [s for s in sources if s.get("tool") == tool_type]
                traces.append({
                    "order": order,
                    "tool": tool_type,
                    "query": self._extract_tool_query(content),
                    "results": [
                        {
                            "title": source.get("title", ""),
                            "snippet": source.get("snippet", ""),
                            "url": source.get("url", ""),
                        }
                        for source in matching_sources[:10]
                    ],
                    "summary": f"{len(matching_sources)} unique {tool_type} source(s) captured",
                })
                order += 1
        traced_tools = {trace.get("tool") for trace in traces}
        for tool_type, grouped_sources in self._group_sources_by_tool(sources).items():
            if tool_type in traced_tools:
                continue
            traces.append({
                "order": order,
                "tool": tool_type,
                "query": "",
                "results": [
                    {
                        "title": source.get("title", ""),
                        "snippet": source.get("snippet", ""),
                        "url": source.get("url", ""),
                    }
                    for source in grouped_sources[:10]
                ],
                "summary": f"{len(grouped_sources)} unique {tool_type} source(s) captured",
            })
            order += 1
        return traces

    def _parse_tool_request(self, content: str) -> Optional[Dict[str, str]]:
        """Parse explicit JSON/function-style tool requests from Researcher output."""
        tool_match = re.search(r'"tool"\s*:\s*"(web_search|paper_search)"', content)
        query_match = re.search(r'"query"\s*:\s*"([^"]+)"', content)
        if tool_match:
            return {
                "tool": tool_match.group(1),
                "query": query_match.group(1) if query_match else "",
            }
        call_match = re.search(r'\b(web_search|paper_search)\((.*?)\)', content, flags=re.DOTALL)
        if call_match:
            query_match = re.search(r'["\']([^"\']+)["\']', call_match.group(2))
            return {
                "tool": call_match.group(1),
                "query": query_match.group(1) if query_match else "",
            }
        return None

    def _extract_tool_query(self, content: str) -> str:
        """Extract the original query from formatted tool output when present."""
        match = re.search(r"for '([^']+)'", content)
        return match.group(1) if match else ""

    def _group_sources_by_tool(self, sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group sources by retrieval path for UI transparency."""
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for source in sources:
            groups.setdefault(source.get("tool", "unknown"), []).append(source)
        return groups

    def _build_citation_mapping(
        self,
        final_response: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Map answer sentences to likely supporting sources using title/URL overlap."""
        mappings = []
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', final_response) if len(s.strip()) > 40]
        for index, sentence in enumerate(sentences[:12], 1):
            supporting = self._match_sources_to_sentence(sentence, sources)
            mappings.append({
                "claim_id": index,
                "claim": sentence,
                "sources": supporting[:3],
            })
        return mappings

    def _match_sources_to_sentence(
        self,
        sentence: str,
        sources: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Match claims to sources by citation URL/domain/title token overlap."""
        lowered_sentence = sentence.lower()
        matches = []
        for source in sources:
            title = source.get("title", "")
            url = source.get("url", "")
            domain = url.split("/")[2].lower() if "://" in url else ""
            title_terms = [term for term in re.findall(r'\w+', title.lower()) if len(term) > 4]
            overlap = sum(1 for term in title_terms if term in lowered_sentence)
            if (domain and domain in lowered_sentence) or overlap >= 2:
                matches.append({
                    "title": title,
                    "url": url,
                    "tool": source.get("tool", "unknown"),
                })
        if not matches and sources:
            first_source = sources[0]
            matches.append({
                "title": first_source.get("title", ""),
                "url": first_source.get("url", ""),
                "tool": first_source.get("tool", "unknown"),
            })
        return matches

    def _calculate_evidence_strength(
        self,
        sources: List[Dict[str, Any]],
        tool_traces: List[Dict[str, Any]],
    ) -> str:
        """Summarize evidence sufficiency for UI and evaluation reports."""
        source_tools = {source.get("tool") for source in sources}
        has_mixed_sources = "web_search" in source_tools and "paper_search" in source_tools
        if len(sources) >= 5 and has_mixed_sources:
            return "High"
        if len(sources) >= 3 or tool_traces:
            return "Medium"
        return "Low"

    def _export_session(self, result_data: Dict[str, Any]):
        """Save a human-readable JSON artifact for each completed run."""
        try:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"session_{timestamp}.json"
            metadata = result_data.get("metadata", {})

            session_data = {
                "user_query": result_data.get("query", ""),
                "agent_trace": self._format_session_agent_trace(result_data.get("conversation_history", [])),
                "tool_calls": metadata.get("tool_traces", []),
                "sources": metadata.get("sources", []),
                "final_answer": result_data.get("response", ""),
                "safety_events": metadata.get("safety_events", []),
                "judge_scores": metadata.get("evaluation", result_data.get("evaluation", None)),
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Session exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export session JSON: {e}")

    def _format_session_agent_trace(
        self,
        conversation_history: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Group Planner, Researcher, Writer, and Critic outputs for session export."""
        trace = {
            "Planner": [],
            "Researcher": [],
            "Writer": [],
            "Critic": [],
        }
        for message in conversation_history:
            source = message.get("source", "")
            if source in trace:
                trace[source].append(message.get("content", ""))
        return trace

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        """
        return workflow


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.
    
    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)
    
    # Print workflow visualization
    print(orchestrator.visualize_workflow())
    
    # Example query
    query = "What are the latest trends in human-computer interaction research?"
    
    print(f"\nProcessing query: {query}\n")
    print("=" * 70)
    
    # Process query
    result = orchestrator.process_query(query)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    demonstrate_usage()

