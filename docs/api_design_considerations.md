# API Usability Considerations for RAI Framework

This document synthesizes research on API usability and LLM-compatible API design, providing information and perspectives to RAI core developers and maintainers as they explore API designs that serve multiple audiences: Application Developers, Extension Developers, Core Developers, and LLM agents. This is an active exploration, not a finalized set of rules.

## Table of Contents

-   [Classic Usability Dimensions](#classic-usability-dimensions)
    -   [Progressive Disclosure Pattern](#progressive-disclosure-pattern)
-   [LLM App Design Patterns for RAI](#llm-app-design-patterns-for-rai)
-   [RAI Framework Audience Analysis](#rai-framework-audience-analysis)
    -   [Multiple Audiences with Clear Separation](#multiple-audiences-with-clear-separation)
    -   [API Similarities & Differences](#api-similarities--differences)
    -   [Use Case Differences](#use-case-differences)
    -   [Research Opportunities for RAI](#research-opportunities-for-rai)
-   [Tiered API Design for RAI](#tiered-api-design-for-rai)
    -   [Three-Tier Structure](#three-tier-structure)
    -   [Key Design Considerations](#key-design-considerations)
    -   [Common Usability Concerns in RAI](#common-usability-concerns-in-rai)
    -   [Validation Approaches](#validation-approaches)
-   [References: Detailed Research Findings](#references-detailed-research-findings)
    -   [Classic API Usability Research](#classic-api-usability-research)
    -   [Abstraction Level Trade-offs in Robotics](#abstraction-level-trade-offs-in-robotics)
    -   [Progressive Disclosure](#progressive-disclosure)
    -   [Designing APIs for LLM Agents](#designing-apis-for-llm-agents)
-   [Reading List](#reading-list)

---

## Classic Usability Dimensions

The [Cognitive Dimensions Framework](#cognitive-dimensions-framework) provides a systematic way to evaluate API design. Key dimensions may be relevant to RAI API evaluation:

-   Abstraction Level: Range of abstraction exposed and usable by target developers
-   Progressive Evaluation: Ability to test partially completed code
-   Penetrability: Ease of exploring and understanding API components
-   Consistency: How much can be inferred once part of the API is learned
-   Domain Correspondence: How clearly API components map to the robotics domain
-   Role Expressiveness: How apparent the relationship between components and the program is

Key findings from empirical studies:

-   48% of developers had to understand implementation details to use APIs (Piccioni et al., 2013)
-   Method placement dramatically affects learnability—developers were 2-11x faster when methods were on expected classes (Stylos & Myers, 2008)
-   High abstraction levels improve usability but reduce control—the ideal level depends on task and audience (Diprose et al., 2016)

### Progressive Disclosure Pattern

Progressive disclosure (Nielsen, 1995) defers advanced features to reduce cognitive load. For APIs, this could translate to tiered or layered API design:

-   Start with simple, intent-based methods
-   Reveal complexity progressively as needed
-   Maintain consistent patterns across tiers

For technical contexts, "layered API design" or "tiered abstraction" is preferred over UX terminology.

## LLM App Design Patterns for RAI

From emerging LLM API design research (2023-2025), patterns that may be useful for evaluating RAI APIs across all extensions:

_Semantic Clarity:_

-   Use self-descriptive field names (e.g., `temperature_celsius` not `temp`)
-   Provide rich metadata (units, data types, relationships)
-   Make responses easily interpretable without complex parsing

_Error Handling:_

-   Return actionable error messages with suggestions
-   Include context about what went wrong and how to fix it
-   Enable self-correction and intelligent retry

_Granularity Balance:_

-   Avoid overly broad APIs (inefficient data transfer)
-   Avoid overly fine-grained APIs (too many calls)
-   Strike balance for single-call efficiency with composability

_Self-Descriptive Design:_

-   Use endpoint/method names that create natural language understanding
-   Example: `dataProcessing/endpoint` rather than generic names

_Structured Responses:_

-   Provide structured, machine-readable data
-   Minimize LLM parsing and inference overhead
-   Focus LLM effort on reasoning, not data extraction

These patterns may help evaluate whether RAI APIs are optimized for both human developers and LLM agents.

## RAI Framework Audience Analysis

### Multiple Audiences with Clear Separation

RAI serves four distinct roles at different architectural levels:

-   Application Developers: Design and configure the system (choose Agents, Connectors, Tools)
-   Extension Developers: Extend RAI by creating custom tools and components
-   Core Developers: Implement new framework components (connectors, agents)
-   LLM agents: Consume Tools at runtime via tool-calling mechanisms

The paper states Tools are "compatible with langchain, enabling seamless integration with tool-calling-enabled LLMs" while also being "used by Agents utilizing other decision-making mechanisms."

### API Similarities & Differences

_Similarities:_

-   All roles interact with the Tools abstraction (Section 2.1)
-   All access the same underlying capabilities (ROS 2 services, perception, navigation)

_Differences:_

-   Application Developers: Configure Agents (Conversational, StateBased), Connectors (ROS2Connector), and system architecture
-   Extension Developers: Create custom tools extending `BaseROS2Tool`, implement domain-specific functionality
-   Core Developers: Implement new connector types, agent types, framework infrastructure
-   LLM agents: Invoke Tools at runtime (CallROS2Service, GetDistanceToObjects)

_Key difference:_ Application Developers architect the system; LLM agents execute within it. Extension and Core Developers extend the framework itself. RAI maintains clear separation: Application Developers configure, LLM agents operate, Extension/Core Developers extend. However, APIs need to serve multiple audiences simultaneously, which may benefit from tiered design that supports progressive disclosure.

### Use Case Differences

_Application Developers:_

-   System design: "A series of experiments was conducted using different architecture configurations" (Section 3.1)
-   Debugging: "advantages such as...easier debugging"
-   Multi-agent setup: Choosing between single agent vs. multi-agent architectures (Figure 6)

_Extension Developers:_

-   Tool creation: Extending `BaseROS2Tool` to create domain-specific tools
-   Component extension: Creating custom aggregators, specialized connectors

_Core Developers:_

-   Framework infrastructure: Implementing new connector types, agent architectures
-   Low-level integration: Creating new communication patterns, message types

_LLM agents:_

-   Runtime decisions: "The Agent was also responsible for deciding when to report completion or failure"
-   Tool invocation: Using open-set detection, navigation planning, manipulation
-   Task execution: "Navigate to the chair", object sorting, stacking

### Research Opportunities for RAI

RAI's multi-audience architecture and tiered API design present novel research opportunities that have not been formally studied:

1. _Tiered APIs for dual human/LLM audiences_: No formal research exists on tiered APIs serving both human developers and LLM agents simultaneously. Current work focuses on either human-focused OR LLM-focused APIs, not both.

2. _Empirical studies of LLM agent API usage_: Limited empirical work exists on how LLM agents interact with APIs. Most guidance is anecdotal or practitioner advice rather than rigorous user studies.

3. _Progressive disclosure patterns in APIs_: While progressive disclosure is established in UI design, it hasn't been formalized for programmatic interfaces. RAI could establish patterns for progressive disclosure in API design.

4. _Frameworks for evaluating multi-audience robotics APIs_: No established frameworks exist for evaluating robotics APIs that serve multiple audiences (Application Developers, Extension Developers, Core Developers, and LLM agents) with different needs that must be balanced.

_Research opportunity:_ RAI could be among the first to empirically study how LLM agents navigate tiered APIs vs. how humans do, and whether a single design can serve both well across robotics domains. The framework's unique position at the intersection of classical API usability research, emerging LLM-optimized design, and robotics domain requirements provides a unique opportunity to validate tiered API patterns empirically.

## Tiered API Design for RAI

A tiered API structure could organize code into three abstraction levels to support progressive disclosure and reduce cognitive load. This pattern might apply across RAI extensions (perception, navigation, manipulation, etc.).

### Three-Tier Structure

_High-level layer (Agent-friendly):_ Simple, intent-based tools with minimal required arguments. Tools hide pipeline complexity, algorithm selection, and configuration details. For example, `GetObjectGrippingPointsTool(object_name="cup")` only requires the object name—camera topics, filter configs, and estimation strategies are handled via ROS2 parameters or sensible defaults. Agents don't need to understand the pipeline or choose between `isolation_forest` vs `dbscan` strategies.

_Mid-level layer (Configurable):_ Configurable components that expose key parameters for tuning behavior. Components allow users to configure strategies and methods without implementing algorithms from scratch. For example, `PointCloudFilter` and `GrippingPointEstimator` with their Config classes (`PointCloudFilterConfig`, `GrippingPointEstimatorConfig`) allow configuring filtering strategies and estimation methods without understanding DBSCAN or Isolation Forest internals.

_Low-level layer (Expert control):_ Core algorithms providing direct access to model inference and processing stages. For example, detection algorithms (e.g., `GDBoxer`), segmentation algorithms (e.g., `GDSegmenter`), and processing functions (e.g., `depth_to_point_cloud`) for users who need full control over every parameter.

### Key Design Considerations

1. _Named presets over raw parameters_: High-level tools could support semantic presets (e.g., `quality="high"`, `approach="top_down"`) that internally map to appropriate component configurations, rather than exposing all algorithm parameters.

2. _Semantic parameter names_: Mid-level components might expose parameters that describe outcomes (e.g., `noise_handling="aggressive"`) rather than algorithm names (e.g., `strategy="isolation_forest"`).

3. _Rich result metadata_: Tools could return confidence scores, strategy used, and alternative options to help LLM agents make better decisions about retrying or adjusting approaches.

4. _Progressive evaluation support_: Users might benefit from being able to test individual pipeline stages without running the full pipeline, enabling incremental debugging and validation.

### Common Usability Concerns in RAI

Based on analysis of RAI extensions, common usability issues include:

1. _Configuration complexity_: Tools requiring 6+ ROS2 parameters plus multiple configuration objects with 10+ parameters each create high cognitive load.

2. _Algorithm knowledge requirement_: Mid-level users must understand domain-specific concepts (DBSCAN, RANSAC, path planning algorithms) to configure effectively.

3. _Hidden dependencies_: Tool initialization depends on ROS2 parameters that must be set before tool creation—no clear error if missing.

4. _Pipeline complexity_: Tools orchestrate multi-stage pipelines (detection → processing → filtering → estimation) with no visibility into intermediate stages.

5. _Progressive evaluation difficulty_: Cannot test individual pipeline stages—must run full pipeline to see results.

6. _Parameter discovery_: Configuration options scattered across multiple config classes—no single source of truth for all parameters.

7. _Error messages_: Algorithm-specific errors may not provide actionable guidance for LLM agents or application developers.

8. _Domain correspondence gap_: Parameter names like `if_contamination`, `lof_n_neighbors` don't clearly map to domain concepts.

9. _Code duplication_: Infrastructure-level duplication (service client creation, message retrieval, parameter handling) creates maintenance burden and inconsistent error handling.

### Validation Approaches

Tiered API structures could be validated through the following scenarios:

_High-Level Tier:_

-   Zero-shot agent usage: LLM agents can use tools (e.g., `GetObjectGrippingPointsTool(object_name="cup")`) without examples or parameter tuning
-   Error recovery: Tools return actionable error messages with suggestions (e.g., "Try with quality='high'")
-   Common task success: 80%+ of tasks solvable with high-level API alone, 1-2 tool calls per task

_Mid-Level Tier:_

-   Edge case handling: Components handle non-standard scenarios with semantic parameters (e.g., `noise_handling="aggressive"`)
-   Strategy comparison: Users can experiment with different approaches without understanding low-level algorithms
-   Environment adaptation: Same API works across different robot/sensor setups via configuration

_Low-Level Tier:_

-   Custom pipelines: Users can inject custom algorithms and compose pipeline stages independently
-   Algorithm reproduction: Every stage is accessible and configurable for research use cases

_Tier Transitions:_

-   Gradual complexity: Users can progress from simple to advanced without rewriting code
-   API discoverability: Docstrings and type hints guide users to appropriate tiers

_Performance:_

-   Abstraction overhead: High-level tools might aim for <5% overhead compared to direct algorithm usage

_Success Metrics (examples to consider):_

-   Target: 90% of new users succeed with high-level tier
-   Target: 15% of power users utilize low-level tier
-   Target: > 85% LLM agent success rate
-   Natural progression between tiers over time

---

## References: Detailed Research Findings

### Classic API Usability Research

#### Cognitive Dimensions Framework

The foundational work for evaluating API usability comes from the Cognitive Dimensions Framework, which provides a systematic way to assess API design trade-offs.

_Key Research Findings:_

-   Piccioni et al. (2013) - "An Empirical Study of API Usability" studied API abstraction levels and found that 48% of participants had to understand implementation details to use APIs, revealing abstraction deficiencies. The study involved 25 programmers (students, researchers, professionals) using a persistence library API.

-   Clarke & Becker (2003-2006) - Applied Cognitive Dimensions to characterize API design trade-offs and evaluate Microsoft .NET libraries.

-   Systematic mapping study (2019) - Reviewed 47 primary studies on API usability evaluation methods, showing 32 appeared from 2010-2018, indicating growing research interest.

_Key Cognitive Dimensions for RAI:_

-   Abstraction Level: Range of abstraction exposed and usable by developers
-   Progressive Evaluation: Ability to test partially completed code
-   Penetrability: Ease of exploring and understanding API components
-   Consistency: How much can be inferred once part of the API is learned
-   Domain Correspondence: How clearly API components map to the robotics domain
-   Role Expressiveness: How apparent the relationship between components and the program is

_Key Findings from Piccioni et al. (2013):_

Study of 25 programmers using a persistence library API revealed:

-   Naming is critical: Finding descriptive, non-ambiguous names is challenging (e.g., "CRUD" unfamiliar to 44%)
-   Type relationships are unclear: Discovering connections between API classes requires significant effort
-   Documentation is critical: ALL major usability flaws traced to incomplete/unclear documentation
-   Flexibility cuts both ways: Experienced programmers leverage it; novices get confused by choices
-   _Key insight:_ 48% of participants had to peek at implementation details to understand the API—a clear abstraction failure

_Key Findings from Stylos & Myers (2008):_

Study of 10 programmers testing method placement found:

-   Method placement dramatically affects API learnability (2-11x faster when methods are on expected classes)
-   Programmers gravitate to the same starting classes and discover other classes through method signatures
-   _Practical impact:_ Place methods on classes where developers naturally start exploring

### Abstraction Level Trade-offs in Robotics

_Diprose et al. (2016-2017)_ studied social robot programming APIs and found: high-level primitives with close domain mapping improved usability, BUT the main trade-off was that hiding implementation details made progressive evaluation difficult—programmers were suddenly exposed to low-level details when debugging.

This directly validates the concern about "better abstractions → reduced flexibility."

User evaluations showed that while high abstraction levels hide lower-level implementation details and provide close mapping to the problem domain, they can take away too much control from programmers.

_Key findings from Diprose et al. (2016):_

-   High-level primitives with close domain mapping improve usability but may need to preserve necessary control
-   The ideal abstraction level depends on task and audience—some need lower levels, others higher
-   Benefits: close domain mapping, hides implementation details, terse notation, good role-expressiveness
-   Main trade-off: Low remote visibility makes progressive evaluation difficult—programmers exposed to low-level details when debugging

### Progressive Disclosure

_Summary from Nielsen (2006):_ Progressive disclosure defers advanced or rarely used features to reduce cognitive load, making applications easier to learn and less error-prone. For RAI, this translates to tiered API design: start simple, reveal complexity as needed. Terminology such as "layered API design" or "tiered abstraction" is preferred over UX terminology for developer audiences.

### Designing APIs for LLM Agents

_Emerging Research (2023-2025):_ This is a very new area, mostly practitioner-focused, with limited formal academic research. APIs for LLMs serve two primary functions: grounding (fetching real-time, factual data to prevent hallucination) and action (executing tasks in external systems).

_What makes an API ready for AI?_ An API is "AI-ready" when designed for efficient interaction with LLMs, providing structured, unambiguous data that prioritizes explicit machine readability over human interpretation.

_Key Insight:_ The utility of large language models can be largely attributed to API designers creating abstractions that fit real use cases well (Myers & Stylos, 2016; Piccioni et al., 2013). Core principles and common mistakes are covered in the [LLM App Design Patterns](#llm-app-design-patterns-for-rai) section above.

_Practical Takeaways for RAI Core Developers (to consider):_

1. _Robotics API research suggests tiered approach_: The findings show high-level primitives work well but abstraction may need to preserve necessary control. This aligns with RAI's tiered API structure across extensions.

2. _Method placement may matter_: When designing tools and components, consider placing methods on the classes where developers naturally start exploring. Research suggests this can make APIs 2-11x faster to learn.

3. _Documentation appears critical_: Research shows all major usability flaws trace to incomplete/unclear documentation. Method signatures, comments, and contracts might benefit from being comprehensive.

4. _Abstraction gaps may indicate usability issues_: If 48% of users need to peek at implementation details, the abstraction level might need adjustment. RAI could consider providing clear paths between tiers without requiring implementation knowledge.

5. _Progressive evaluation may enable debugging_: High abstraction levels make progressive evaluation difficult. RAI's tiered structure might benefit from allowing testing of individual stages without running full pipelines.

6. _Semantic clarity for LLM agents_: Consider using self-descriptive names, structured responses, and actionable error messages. LLM agents may benefit from explicit machine-readable data, rather than ambiguous responses requiring complex parsing.

---

## Reading List

1. **Diprose et al. (2016)** - "Designing an API at an appropriate abstraction level for programming social robot applications" - https://www.cs.auckland.ac.nz/~beryl/publications/jvlc%202016%20Designing%20API.pdf  
   Abstraction/flexibility trade-off in robotics. High-level primitives work well but shouldn't remove necessary control.

2. **Piccioni, Furia & Meyer (2013)** - "An Empirical Study of API Usability" - https://bugcounting.net/pubs/esem13.pdf  
   Found 48% of participants had to understand implementation details to use APIs, revealing abstraction deficiencies.

3. **Stylos & Myers (2008)** - "The implications of method placement on API learnability" - https://www.cs.cmu.edu/~NatProg/papers/FSE2008-p105-stylos.pdf  
   Method placement dramatically affects learnability—developers were 2-11x faster when methods were on expected classes.

4. **Clarke & Becker (2003)** - "Using the Cognitive Dimensions Framework to evaluate the usability of a class library" - https://www.ppig.org/files/2003-PPIG-15th-clarke.pdf

5. **Gravitee (2025)** - "Designing APIs for LLM Apps" - https://www.gravitee.io/blog/designing-apis-for-llm-apps  
   AI-ready APIs require structured, semantically clear, and context-aware designs.

6. **Jakob Nielsen (2006)** - "Progressive disclosure" - https://www.nngroup.com/articles/progressive-disclosure/  
   Defers advanced features to reduce cognitive load—applies to tiered API design.
