"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Check for harmful content in responses
    - Verify factual consistency
    - Detect potential misinformation
    - Remove PII (personal identifiable information)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # TODO: Initialize guardrail framework
        # Suggested implementation:
        # - Read output safety settings from config
        # - Decide which checks should block vs sanitize
        # - Optionally initialize Guardrails AI / NeMo Guardrails validators
        self.safety_config = config.get("safety", config)
        self.prohibited_categories = self.safety_config.get(
            "prohibited_categories",
            ["harmful_content", "personal_attacks", "misinformation", "off_topic_queries"]
        )
        self.agent_control_terms = {"terminate", "stopmessage", "approved - research complete", "needs revision"}

    def validate(
        self,
        response: str,
        sources: List[Dict[str, Any]] = None,
        role: str = None,
    ) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for harmful content
        - Check for PII
        - Verify claims against sources
        - Check for bias
        """
        violations = []
        resolved_role = self._resolve_role(response, role)

        # TODO: Implement actual validation
        # Suggested implementation:
        # 1. Run helper checks such as _check_pii() and _check_harmful_content()
        # 2. If sources are available, compare claims/citations against them
        # 3. Decide whether to redact, refuse, or allow the response
        # 4. Return sanitized_output for UI display when applicable

        # Layer 1: pure AutoGen control signals are orchestration metadata, not content.
        if self._is_control_signal_only(response):
            return {
                "valid": True,
                "violations": [],
                "sanitized_output": response,
                "role": resolved_role,
                "policy_layer": "control_signal",
                "decision": "allowed",
                "reason": "AutoGen control signal excluded from content safety classification",
            }

        is_evaluation_output = self._is_evaluation_output(response, resolved_role)

        # Layer 2/3 shared checks: never relax PII, malicious instructions, or injection.
        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)

        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        injection_violations = self._check_prompt_injection(response)
        violations.extend(injection_violations)

        misinformation_violations = self._check_misinformation_patterns(response)
        violations.extend(misinformation_violations)

        # Layer 2: normal generated content gets the full output safety policy.
        if not is_evaluation_output:
            bias_violations = self._check_bias(response)
            violations.extend(bias_violations)

        if sources and not is_evaluation_output:
            consistency_violations = self._check_factual_consistency(response, sources)
            violations.extend(consistency_violations)

        blocking_severities = {"high", "medium"}
        is_valid = not any(v.get("severity") in blocking_severities for v in violations)
        policy_layer = "evaluation_relaxed" if is_evaluation_output else "normal_output"
        decision = "relaxed_allowed" if is_evaluation_output and is_valid else "allowed"
        if not is_valid:
            decision = "blocked"
        return {
            "valid": is_valid,
            "violations": violations,
            "sanitized_output": self._sanitize(response, violations) if violations else response,
            "role": resolved_role,
            "policy_layer": policy_layer,
            "decision": decision,
            "reason": self._decision_reason(policy_layer, is_valid, violations),
        }

    def _resolve_role(self, text: str, role: str = None) -> str:
        """Use caller-provided role when available, otherwise infer conservatively."""
        if role:
            return role.lower()
        lowered = text.lower()
        if self._is_control_signal_only(text):
            return "system"
        if self._looks_like_critic_evaluation(lowered):
            return "critic"
        if "research findings" in lowered or "sources" in lowered:
            return "researcher"
        if "draft" in lowered or "references" in lowered:
            return "writer"
        if "research plan" in lowered or "steps" in lowered:
            return "planner"
        return "unknown"

    def _is_control_signal_only(self, text: str) -> bool:
        """Return True only for standalone orchestration control messages."""
        normalized = re.sub(r"[\s`*_#:\-]+", " ", text).strip().lower()
        return normalized in {"terminate", "stopmessage", "stop message"}

    def _is_evaluation_output(self, text: str, role: str) -> bool:
        """
        Detect Critic/evaluation outputs that should use relaxed output checks.

        Relaxed mode still runs PII, harmful-content, injection, and misinformation
        checks; it only avoids treating rubric/control vocabulary as unsafe.
        """
        lowered = text.lower()
        if role == "critic":
            return True
        return self._looks_like_critic_evaluation(lowered)

    def _looks_like_critic_evaluation(self, lowered_text: str) -> bool:
        """Identify scoring/rubric text without relying on one brittle keyword."""
        if any(term in lowered_text for term in self.agent_control_terms):
            return True
        evaluation_markers = [
            "evaluation of research",
            "evaluation summary",
            "score:",
            "relevance",
            "evidence quality",
            "completeness",
            "accuracy",
            "clarity",
        ]
        marker_count = sum(1 for marker in evaluation_markers if marker in lowered_text)
        return marker_count >= 2

    def _decision_reason(
        self,
        policy_layer: str,
        is_valid: bool,
        violations: List[Dict[str, Any]],
    ) -> str:
        """Create a concise reason for logs and UI metadata."""
        if not is_valid and violations:
            return violations[0].get("reason", "Output violated safety policy")
        if policy_layer == "evaluation_relaxed":
            return "Critic/evaluation output allowed under relaxed policy after core safety checks"
        return "Output passed configured safety checks"

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Expand regex checks for emails, phone numbers, SSNs, addresses, etc.
        - Use a stronger PII detection library if desired
        - Return violation metadata needed for redaction
        """
        violations = []

        # Simple regex patterns for common PII
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches
                })

        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """Detect outputs that try to override or reveal system/developer prompts."""
        violations = []
        injection_patterns = [
            r"\bignore (all )?(previous|prior) instructions\b",
            r"\breveal (the )?(system|developer) prompt\b",
            r"\bdisregard (the )?(system|developer) message\b",
            r"\bjailbreak\b",
        ]
        lowered = text.lower()
        for pattern in injection_patterns:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Prompt-injection pattern detected: {pattern}",
                    "severity": "high"
                })
                break
        return violations

    def _check_misinformation_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Flag explicit fabrication instructions or admissions in generated output."""
        violations = []
        misinformation_patterns = [
            r"\bfabricate(d)?\s+(sources|citations|evidence)\b",
            r"\bmake up\s+(sources|citations|evidence|data)\b",
            r"\bfake\s+(sources|citations|evidence)\b",
        ]
        lowered = text.lower()
        for pattern in misinformation_patterns:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "misinformation",
                    "reason": f"Potential fabrication pattern detected: {pattern}",
                    "severity": "medium"
                })
                break
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Detect unsafe instructions, hateful content, or violent guidance
        - Use a moderation model, guardrail validator, or rule-based policy check
        - Return severity levels so the caller knows whether to refuse or sanitize
        """
        violations = []

        # Keep this conservative: flag unsafe instructions, not agent status text.
        harmful_patterns = [
            r"\bstep[- ]by[- ]step\s+(attack|exploit|intrusion)\b",
            r"\bhow to\s+(exploit|bypass security|make a bomb|build a weapon|poison someone)\b",
            r"\binstructions\s+for\s+(self[- ]harm|making a bomb|building a weapon)\b",
            r"\bprovide\s+(exploit|malware|phishing)\s+(code|steps|instructions)\b",
        ]
        lowered = text.lower()
        for pattern in harmful_patterns:
            if re.search(pattern, lowered):
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"Unsafe instructional content matched pattern: {pattern}",
                    "severity": "high"
                })

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Compare claims in the response against the retrieved evidence
        - Verify that citations actually support the statements made
        - Optionally use an LLM-based verifier or a citation-grounding check
        """
        violations = []

        # Basic citation grounding check: warn when sources exist but are not cited.
        if not response.strip():
            violations.append({
                "validator": "factual_consistency",
                "reason": "Response is empty and cannot be verified against sources",
                "severity": "medium"
            })
            return violations

        has_citation_marker = bool(re.search(r'https?://|\[Source:|\[\d+\]', response))
        if sources and not has_citation_marker:
            violations.append({
                "validator": "factual_consistency",
                "reason": "Response uses sources but does not show visible citations",
                "severity": "low"
            })

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Look for stereotypes, blanket generalizations, or discriminatory language
        - Decide whether to redact, revise, or refuse the output
        """
        violations = []
        # Flag blanket claims that are easy to revise without blocking the answer.
        bias_patterns = [
            r"\ball\s+\w+\s+are\b",
            r"\b(always|never)\s+because of their\b",
            r"\binferior\b",
            r"\bsuperior race\b",
        ]
        for pattern in bias_patterns:
            if re.search(pattern, text.lower()):
                violations.append({
                    "validator": "bias",
                    "reason": "Potentially biased or overgeneralized language detected",
                    "severity": "medium"
                })
                break
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Redact matched PII spans
        - Replace unsafe sections with placeholder text
        - Optionally return a refusal message for severe violations
        """
        sanitized = text

        # Redact PII
        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, "[REDACTED]")
            elif violation.get("severity") == "high":
                # High-severity output can be fully redacted when sanitize mode is configured.
                sanitized = "[REDACTED unsafe content]"

        return sanitized
