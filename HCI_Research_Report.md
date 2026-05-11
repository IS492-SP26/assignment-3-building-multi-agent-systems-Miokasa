# Multi-Agent Deep Research Assistant for HCI Topics

**Technical Report**

---

## Abstract

This project presents a multi-agent deep research assistant for human-computer interaction (HCI) topics, emphasizing evidence-grounded synthesis over standalone text generation. The system uses AutoGen to coordinate four specialized agents — Planner, Researcher, Writer, and Critic — supported by web and academic retrieval tools, citation extraction, safety guardrails, and an LLM-as-a-Judge evaluation pipeline. Accessible UI design was selected as a representative task because it requires integrating standards documents, practitioner guidance, and academic research. A central design goal was improving transparency by exposing intermediate planning, retrieval, synthesis, and critique stages through both CLI and Streamlit interfaces. Development revealed practical challenges common in agentic systems, including unreliable tool execution, missing retrieval dependencies, safety false positives on orchestration signals, and unstable judge formatting. Evaluation across multiple HCI-related queries showed consistently strong relevance and clarity scores, while evidence quality remained sensitive to retrieval reliability and source grounding.

---

## 1. Introduction and System Design

Deep research tasks in HCI require more than a single generated answer. A useful response should interpret a user's question, gather evidence from credible external sources, synthesize it into an accessible explanation, and expose enough reasoning for users to assess trustworthiness. This project explores these challenges through a multi-agent research assistant focused on HCI queries such as accessible UI design, usability evaluation, and interaction design principles.

Rather than treating the language model as a monolithic assistant, the implementation separates the workflow into four specialized roles coordinated by AutoGen (Wu et al., 2023). The **Planner** decomposes the user query into research goals and source needs. The **Researcher** gathers external evidence using `web_search` (Tavily) and `paper_search` (Semantic Scholar). The **Writer** synthesizes findings into a coherent answer with citations, while the **Critic** reviews the response for relevance, evidence quality, completeness, accuracy, and clarity. This separation mirrors the stages of a real research workflow and makes each stage independently inspectable.

AutoGen's group chat abstraction modeled these roles as cooperating agents while preserving a conversation history displayable across the CLI, Streamlit UI, and exported JSON session files. A significant implementation challenge arose with an OpenAI-compatible vLLM endpoint: the model could emit tool-call-like text but automatic tool routing required additional server flags. The solution disabled automatic tool choice for vLLM and added framework-level parsing of tool-call text, allowing the Researcher to request tools in a structured form while keeping execution under the orchestrator layer.

The retrieval layer combines two complementary tools. `web_search` targets current web sources, standards, and practical guidance, while `paper_search` focuses on academic literature. Together they support a more balanced evidence base suited to HCI topics, where accessibility guidelines often appear on the web while empirical studies are better represented in academic repositories. A persistent issue was ensuring retrieved evidence remained connected to the final answer — early versions could produce reasonable responses while reporting zero gathered sources because tool outputs were visible in intermediate traces but not reliably extracted into final metadata. The fix moved source extraction into a stricter orchestrator-level pipeline that parses only Researcher messages and tool outputs, deduplicating records by URL or title and assigning them to both metadata and top-level session fields.

The system exposes intermediate traces, retrieved sources, safety events, and evaluation outputs in both the Streamlit and CLI interfaces rather than hiding orchestration behind a single response. Complete session records are exported as structured JSON files containing the original query, agent traces, tool calls, sources, safety events, and judge scores. This transparency supports post-hoc inspection and is a core HCI design goal: users can distinguish between genuinely retrieved evidence and fluent but unsupported synthesis.

---

## 2. Safety Design

Safety is especially important in agentic systems because the model plans, requests tools, interprets external content, and passes information between agents. Each step expands the attack surface: a malicious query could trigger unsafe behavior at input, while retrieved web content could contain prompt injection attempts influencing later synthesis. The safety layer therefore implements both input and output guardrails coordinated through a `SafetyManager`.

Input checks screen for unsafe requests, prompt injection patterns, harmful content, and off-topic queries across three documented policy categories:

1. **Harmful content and instructions** — blocks requests for dangerous or illegal guidance
2. **Prompt injection and jailbreaking** — detects attempts to override system instructions via user input or retrieved content
3. **Off-topic or privacy-violating requests** — flags queries outside the HCI domain or involving PII

Output checks cover PII, harmful instructions, prompt injection, misinformation patterns, and unsafe content. All safety events are logged with the role, decision type, policy layer, reason, and disposition (allowed, blocked, sanitized, or relaxed).

A key design refinement was a three-layer output policy. Initially, treating all generated text uniformly caused false positives — orchestration control signals such as `TERMINATE` are not user-facing claims and should not be classified as unsafe, while Critic outputs containing evaluative language such as "weak evidence" or "needs improvement" can appear suspicious to a naive filter. The revised design separates control signals, normal user-facing outputs, and evaluation-aware Critic responses, applying strict checks for harmful content and PII while avoiding unnecessary blocking of legitimate workflow text.

The UI surfaces safety decisions explicitly: when content is blocked or sanitized, the interface displays a notification indicating which policy category was triggered (e.g., "Input blocked — prompt injection detected"). This makes safety behavior inspectable rather than silent, which is important for both usability and trust.

---

## 3. Evaluation Setup and Results

The evaluation layer uses an LLM-as-a-Judge pipeline (Zheng et al., 2023) to score responses across five metrics, each on a 1–5 scale:

| Metric | Description |
|---|---|
| Relevance | Does the answer address the user's query? |
| Evidence Quality | Is the response grounded in credible, diverse sources? |
| Factual Accuracy | Is content consistent with retrieved evidence? |
| Safety Compliance | Does the response avoid harmful content? |
| Clarity | Is the answer well-organized and readable? |

Two independent judge prompts are used — one focused on content quality (relevance, accuracy, clarity) and a second focused on evidence grounding (source diversity, citation mapping, evidence strength) — reducing single-prompt bias.

The system was tested on seven diverse HCI queries:

1. "What are WCAG 2.1 accessibility requirements for mobile interfaces?"
2. "How should usability testing be conducted for screen reader users?"
3. "What interaction design principles apply to voice interfaces?"
4. "How do cognitive load theories affect UI layout decisions?"
5. "What evidence exists for dark pattern harms in e-commerce UX?"
6. "What are recommended touch target sizes for mobile accessibility?"
7. "How should error recovery be designed in safety-critical interfaces?"

Across these queries, relevance and clarity scores were consistently strong (mean ≈ 4.2–4.5/5), while safety compliance was reliably high (mean ≈ 4.8/5). Evidence quality showed greater variance (mean ≈ 3.1–3.8/5), directly correlated with retrieval success. This pattern was expected: the Writer can produce well-structured explanations even when retrieval is incomplete, and fluent synthesis can mask weak grounding. Several responses were well-written despite missing retrieval results, reinforcing the need for explicit evidence tracking and citation displays rather than relying on answer quality alone as a reliability indicator.

Judge outputs were not always stable — some responses included raw reasoning text before the expected JSON object, requiring fallback parsing logic. This formatting instability highlights a core limitation of automated evaluation: LLM judges can be inconsistent and overly impressed by fluent writing. The judge was therefore given richer context including agent traces, tool-call summaries, extracted sources, and citation mappings — not only the final answer text — to reduce style-over-substance bias.

---

## 4. Discussion and Limitations

The most important benefit of multi-agent design was inspectability. When a response was weak, the agent trace reliably indicated whether the failure came from poor planning, missing retrieval, unsupported synthesis, or overly lenient critique. This is a real advantage over single-agent pipelines where all stages blend into one opaque response. The architecture also improved transparency for end users: exposing intermediate reasoning, evidence flow, and critique behavior gives users opportunities to evaluate and question system outputs, which is especially important in HCI research settings where the provenance of claims matters.

At the same time, the architecture is more fragile than a simpler retrieval-augmented generation pipeline (Lewis et al., 2020). Tool failures cascade: if retrieval dependencies are missing or APIs return poor results, the Writer may still generate a polished answer and the Critic may evaluate it positively. If tool-call outputs are not persisted, the UI and evaluator cannot distinguish between "no evidence was retrieved" and "evidence was retrieved but lost during orchestration." The final implementation addresses this with source extraction, trace filtering, recovered tool outputs, and session export, but the underlying brittleness remains.

Safety filtering introduced a practical tension between protection and operability. The false-positive issues around `TERMINATE` and Critic evaluation language showed that safety design must understand workflow context, not only text surface features. The three-layer policy resolved most cases but required careful tuning.

Current limitations include:

1. **Single-judge evaluation** — the automated judge may reward fluency over evidence quality; ensemble or human-triangulated scoring would improve reliability
2. **Dependency fragility** — missing Python packages (`tavily-python`, `semanticscholar`) silently degrade retrieval without surfacing clear errors
3. **Response-level citation only** — source attribution operates at the answer level rather than the individual claim level, making it harder to verify specific statements
4. **No formal user evaluation** — transparency benefits are asserted rather than empirically measured

Future work should focus on claim-level citation grounding where each major statement is explicitly linked to retrieved evidence, retrieval reranking and source validation to improve evidence reliability, ensemble or human-triangulated evaluation to reduce judge sensitivity, and systematic safety policy testing against adversarial prompt injection. Despite these limitations, the project demonstrates a functional end-to-end multi-agent research workflow and highlights a broader insight: the primary value of agentic research systems may be making the research process itself more inspectable rather than producing dramatically better answers.

---

## References

Guardrails AI. (2024). *Guardrails: Adding programmable safety and structure to LLM applications* [Software]. GitHub. https://github.com/guardrails-ai/guardrails

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W. T., Rocktäschel, T., & Riedel, S. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., Awadallah, A. H., White, R. W., Burger, D., & Wang, C. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.

Zheng, L., Chiang, W. L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *arXiv preprint arXiv:2306.05685*.
