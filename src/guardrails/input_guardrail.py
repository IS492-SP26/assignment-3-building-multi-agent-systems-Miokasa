"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List


class InputGuardrail:
    """
    Guardrail for checking input safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define validation rules
    - Implement custom validators
    - Handle different types of violations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # TODO: Initialize guardrail framework
        # Suggested implementation:
        # - Read safety settings from config.yaml
        # - Store min/max query length thresholds
        # - Prepare policy categories such as harmful content,
        #   prompt injection, and off-topic queries
        # - Optionally initialize Guardrails AI / NeMo Guardrails here
        self.safety_config = config.get("safety", config)
        self.system_config = config.get("system", {})
        self.min_length = self.safety_config.get("min_query_length", 5)
        self.max_length = self.safety_config.get("max_query_length", 2000)
        self.topic = self.system_config.get(
            "topic",
            self.safety_config.get("topic", "HCI Research")
        )
        self.prohibited_categories = self.safety_config.get(
            "prohibited_categories",
            ["harmful_content", "personal_attacks", "misinformation", "off_topic_queries"]
        )

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for toxic language
        - Check for prompt injection attempts
        - Check query length and format
        - Check for off-topic queries
        """
        violations = []

        # TODO: Implement actual validation
        # Suggested implementation:
        # 1. Normalize the input (strip spaces, lowercase copy for keyword checks)
        # 2. Add length checks using thresholds from config
        # 3. Call helper methods like _check_toxic_language(),
        #    _check_prompt_injection(), and _check_relevance()
        # 4. Decide whether violations should block, sanitize, or warn
        # 5. Return both the raw violations and a sanitized_input if applicable
        normalized_query = query.strip()

        # Basic deterministic checks keep safety available without extra services.
        if len(normalized_query) < self.min_length:
            violations.append({
                "validator": "length",
                "reason": "Query too short",
                "severity": "low"
            })

        if len(normalized_query) > self.max_length:
            violations.append({
                "validator": "length",
                "reason": "Query too long",
                "severity": "medium"
            })

        violations.extend(self._check_toxic_language(normalized_query))
        violations.extend(self._check_prompt_injection(normalized_query))
        violations.extend(self._check_relevance(normalized_query))

        blocking_severities = {"high", "medium"}
        return {
            "valid": not any(v.get("severity") in blocking_severities for v in violations),
            "violations": violations,
            "sanitized_input": normalized_query
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Use a moderation API, Guardrails validator, or keyword/rule-based classifier
        - Return a list of violations with validator name, reason, and severity
        - Mark clearly unsafe requests as high severity
        """
        violations = []
        # Lightweight policy keywords cover common unsafe request categories.
        harmful_keywords = {
            "harmful_content": [
                "build a weapon",
                "make a bomb",
                "self harm",
                "suicide instructions",
                "poison someone",
            ],
            "personal_attacks": [
                "hate speech",
                "racial slur",
                "harass",
                "dox",
            ],
            "misinformation": [
                "fake evidence",
                "fabricate sources",
                "make up citations",
            ],
        }

        lowered = text.lower()
        for category, keywords in harmful_keywords.items():
            if category not in self.prohibited_categories:
                continue
            for keyword in keywords:
                if keyword in lowered:
                    violations.append({
                        "validator": category,
                        "reason": f"Potentially unsafe request detected: {keyword}",
                        "severity": "high"
                    })
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Detect phrases like \"ignore previous instructions\",
        #   attempts to reveal system prompts, or role-confusion attacks
        - Consider whether the result should block the request or sanitize it
        """
        violations = []
        # Check for common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
            "developer message",
            "reveal your prompt",
            "show your system prompt",
            "jailbreak",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high"
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        TODO: YOUR CODE HERE
        Suggested implementation:
        - Compare the query to the configured topic in config.yaml
        - Use keyword heuristics or an LLM classifier
        - Return low/medium severity violations for off-topic requests
        """
        violations = []
        # Treat off-topic detection as a warning so useful broad research still works.
        if "off_topic_queries" not in self.prohibited_categories:
            return violations

        hci_keywords = [
            "hci", "human-computer", "user", "users", "ux", "ui",
            "interface", "interaction", "design", "accessibility",
            "usability", "research", "ai", "ar", "voice", "education",
            "healthcare", "visualization", "mobile", "prototype",
        ]
        lowered = query.lower()
        if self.topic and self.topic.lower() not in lowered:
            if not any(keyword in lowered for keyword in hci_keywords):
                violations.append({
                    "validator": "relevance",
                    "reason": f"Query may be outside the configured topic: {self.topic}",
                    "severity": "low"
                })
        return violations
