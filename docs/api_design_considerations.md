# API Usability Considerations for RAI Framework

This document synthesizes research on API usability and LLM-compatible API design, providing information and perspectives to RAI core developers and maintainers as they explore API designs that serve multiple audiences: Application Developers, Extension Developers, Core Developers, and LLM agents. This is an active exploration, not a finalized set of rules.

## Table of Contents

-   [Classic Usability Dimensions](#classic-usability-dimensions)
-   [LLM App Design Patterns for RAI](#llm-app-design-patterns-for-rai)
-   [RAI Framework Audience Analysis](#rai-framework-audience-analysis)
-   [Tiered API Design for RAI](#tiered-api-design-for-rai)
    -   [Three-Tier Structure](#three-tier-structure)
    -   [Key Design Considerations](#key-design-considerations)
    -   [Case Study: rai_perception Implementation](#case-study-rai_perception-implementation)
    -   [Common Usability Concerns in RAI extension](#common-usability-concerns-in-rai-extension)
-   [Research Findings and Practical Takeaways](#research-findings-and-practical-takeaways)
-   [Reading List](#reading-list)

---

## Classic Usability Dimensions

The Cognitive Dimensions Framework provides a systematic way to evaluate API design. Key dimensions relevant to RAI API evaluation:

-   **Abstraction Level**: Range of abstraction exposed and usable by target developers
-   **Progressive Evaluation**: Ability to test partially completed code
-   **Penetrability**: Ease of exploring and understanding API components
-   **Consistency**: How much can be inferred once part of the API is learned
-   **Domain Correspondence**: How clearly API components map to the robotics domain
-   **Role Expressiveness**: How apparent the relationship between components and the program is

**Key findings from empirical studies:**

-   48% of developers had to understand implementation details to use APIs (Piccioni et al., 2013)
-   Method placement dramatically affects learnability—developers were 2-11x faster when methods were on expected classes (Stylos & Myers, 2008)
-   High abstraction levels improve usability but reduce control—the ideal level depends on task and audience (Diprose et al., 2016)

**Progressive Disclosure Pattern**: Progressive disclosure (Nielsen, 1995) defers advanced features to reduce cognitive load. For APIs, this translates to tiered or layered API design: start with simple, intent-based methods, reveal complexity progressively as needed, and maintain consistent patterns across tiers. For technical contexts, "layered API design" or "tiered abstraction" is preferred over UX terminology.

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

The [RAI framework paper](https://arxiv.org/abs/2505.07532}) (Rachwał et al., 2025) states Tools are "compatible with langchain, enabling seamless integration with tool-calling-enabled LLMs" while also being "used by Agents utilizing other decision-making mechanisms.". Based on this paper, RAI serves four distinct roles at different architectural levels:

-   Application Developers: Design and configure the system (choose Agents, Connectors, Tools)
-   Extension Developers: Extend RAI by creating custom tools and components
-   Core Developers: Implement new framework components (connectors, agents)
-   LLM agents: Consume Tools at runtime via tool-calling mechanisms

### API Similarities & Differences

All roles interact with the Tools abstraction and access the same underlying capabilities (ROS 2 services, perception, navigation). However, their interaction patterns differ:

-   **Application Developers**: Configure Agents (Conversational, StateBased), Connectors (ROS2Connector), and system architecture
-   **Extension Developers**: Create custom tools extending `BaseROS2Tool`, implement domain-specific functionality
-   **Core Developers**: Implement new connector types, agent types, framework infrastructure
-   **LLM agents**: Invoke Tools at runtime (CallROS2Service, GetDistanceToObjects)

Application Developers architect the system; LLM agents execute within it. Extension and Core Developers extend the framework itself. RAI maintains clear separation: Application Developers configure, LLM agents operate, Extension/Core Developers extend. However, APIs need to serve multiple audiences simultaneously, which benefits from tiered design that supports progressive disclosure.

### Use Case Differences

_Application Developers:_ System design and configuration, debugging, multi-agent setup decisions

_Extension Developers:_ Tool creation (extending `BaseROS2Tool`), component extension, custom aggregators

_Core Developers:_ Framework infrastructure, new connector/agent types, low-level integration

_LLM agents:_ Runtime tool invocation, task execution, decision-making during task performance

### Research Opportunities for RAI

RAI's multi-audience architecture and tiered API design present novel research opportunities:

1. **Tiered APIs for dual human/LLM audiences**: No formal research exists on tiered APIs serving both human developers and LLM agents simultaneously
2. **Empirical studies of LLM agent API usage**: Limited empirical work exists on how LLM agents interact with APIs
3. **Progressive disclosure patterns in APIs**: While established in UI design, it hasn't been formalized for programmatic interfaces
4. **Frameworks for evaluating multi-audience robotics APIs**: No established frameworks exist for evaluating robotics APIs that serve multiple audiences with different needs

RAI could be among the first to empirically study how LLM agents navigate tiered APIs vs. how humans do, and whether a single design can serve both well across robotics domains.

## Tiered API Design for RAI

A tiered API structure could organize code into three abstraction levels to support progressive disclosure and reduce cognitive load. This pattern might apply across RAI extensions (perception, navigation, manipulation, etc.).

_Note:_ The `rai_perception` extension has implemented a tiered API structure as an exploration of these design principles. See [`rethinking_usability.md`](extensions/rethinking_usability.md) for detailed analysis and implementation examples.

### Three-Tier Structure

_High-level layer (Agent-friendly):_ Simple, intent-based tools with minimal required arguments. Tools hide pipeline complexity, algorithm selection, and configuration details. For example, `GetObjectGrippingPointsTool(object_name="cup")` only requires the object name—camera topics, filter configs, and estimation strategies are handled via ROS2 parameters or sensible defaults. Agents don't need to understand the pipeline or choose between `isolation_forest` vs `dbscan` strategies.

_Mid-level layer (Configurable):_ Configurable components that expose key parameters for tuning behavior. Components allow users to configure strategies and methods without implementing algorithms from scratch. For example, `PointCloudFilter` and `GrippingPointEstimator` with their Config classes (`PointCloudFilterConfig`, `GrippingPointEstimatorConfig`) allow configuring filtering strategies and estimation methods without understanding DBSCAN or Isolation Forest internals.

_Low-level layer (Expert control):_ Core algorithms providing direct access to model inference and processing stages. For example, detection algorithms (e.g., `GDBoxer`), segmentation algorithms (e.g., `GDSegmenter`), and processing functions (e.g., `depth_to_point_cloud`) for users who need full control over every parameter.

### Key Design Considerations

1. _Named presets over raw parameters_: High-level tools could support semantic presets (e.g., `quality="high"`, `approach="top_down"`) that internally map to appropriate component configurations, rather than exposing all algorithm parameters.

2. _Semantic parameter names_: Mid-level components might expose parameters that describe outcomes (e.g., `noise_handling="aggressive"`) rather than algorithm names (e.g., `strategy="isolation_forest"`).

3. _Rich result metadata_: Tools could return confidence scores, strategy used, and alternative options to help LLM agents make better decisions about retrying or adjusting approaches.

4. _Progressive evaluation support_: Users might benefit from being able to test individual pipeline stages without running the full pipeline, enabling incremental debugging and validation.

### Case Study: rai_perception Implementation

The refactored `rai_perception` extension provides a concrete implementation of tiered API design principles, demonstrating how theoretical considerations translate to practice. Key implementations include:

_Progressive Disclosure:_

-   Three-tier structure: Tools (`tools/`), Components (`components/`), Algorithms (`algorithms/`)
-   Semantic presets (`perception_presets.py`) with self-documenting names: `"default_grasp"`, `"precise_grasp"`, `"top_grasp"`

_Configuration Management:_

-   Multi-tier configuration: algorithm configs (model registry), ROS2 parameters (deployment), component configs (Pydantic classes), presets (semantic mappings)
-   Helper function `rai.communication.ros2.get_param_value()` for consistent parameter extraction with automatic type conversion

_Domain Correspondence:_

-   Semantic parameter names: `outlier_fraction` instead of `if_contamination`, `neighborhood_size` instead of `lof_n_neighbors`
-   Domain-oriented strategy names: `"aggressive_outlier_removal"` instead of `"isolation_forest"`, `"density_based"` instead of `"dbscan"`

_Consistency:_

-   Input schema naming pattern: `{ToolName}Input` (e.g., `GetObjectGrippingPointsToolInput`)
-   Standardized parameter handling: `_load_parameters()` method with consistent prefixes (`perception.gripping_points.*`)

_Role Expressiveness:_

-   Pipeline visibility: Tools expose `pipeline_stages` attributes and `get_pipeline_info()` methods
-   Service dependency clarity: `required_services` attributes and `check_service_dependencies()` methods

_Progressive Evaluation:_

-   Debug mode (`debug=True`) publishes intermediate results to ROS2 topics for RVIZ visualization
-   Enables inspection of pipeline stages without code modification

_Model Registry Pattern:_

-   Enables switching between detection models (e.g., GroundingDINO, YOLO) via ROS2 parameters
-   No code changes required—service reads `model_name` parameter and queries registry

These implementations demonstrate how tiered API design addresses usability concerns identified in research while maintaining flexibility for expert users. The `rai_perception` exploration serves as a reference implementation for other RAI extensions considering similar refactoring.

### Common Usability Concerns in RAI extension

Based on analysis of RAI extensions, common usability issues may include:

1. _Configuration complexity_: Tools requiring 6+ ROS2 parameters plus multiple configuration objects with 10+ parameters each create high cognitive load.

2. _Algorithm knowledge requirement_: Mid-level users must understand domain-specific concepts (DBSCAN, RANSAC, path planning algorithms) to configure effectively.

3. _Hidden dependencies_: Tool initialization depends on ROS2 parameters that must be set before tool creation—no clear error if missing.

4. _Pipeline complexity_: Tools orchestrate multi-stage pipelines (detection → processing → filtering → estimation) with no visibility into intermediate stages.

5. _Progressive evaluation difficulty_: Cannot test individual pipeline stages—must run full pipeline to see results.

6. _Parameter discovery_: Configuration options scattered across multiple config classes—no single source of truth for all parameters.

7. _Error messages_: Algorithm-specific errors may not provide actionable guidance for LLM agents or application developers.

8. _Domain correspondence gap_: Parameter names like `if_contamination`, `lof_n_neighbors` don't clearly map to domain concepts.

9. _Code duplication_: Infrastructure-level duplication (service client creation, message retrieval, parameter handling) creates maintenance burden and inconsistent error handling.

## Research Findings and Practical Takeaways

### Practical Takeaways for RAI Core Developers

1. **Tiered approach aligns with research**: High-level primitives work well but abstraction may need to preserve necessary control. This aligns with RAI's tiered API structure across extensions.

2. **Method placement matters**: Place methods on classes where developers naturally start exploring. Research suggests this can make APIs 2-11x faster to learn.

3. **Documentation is critical**: All major usability flaws trace to incomplete/unclear documentation. Method signatures, comments, and contracts should be comprehensive.

4. **Address abstraction gaps**: If 48% of users need to peek at implementation details, the abstraction level needs adjustment. Provide clear paths between tiers without requiring implementation knowledge. The `rai_perception` implementation addresses this through semantic presets and discoverable component relationships.

5. **Enable progressive evaluation**: High abstraction levels make progressive evaluation difficult. Allow testing of individual stages without running full pipelines. `rai_perception` implements debug mode that publishes intermediate results to ROS2 topics, enabling visualization without code changes.

6. **Semantic clarity for LLM agents**: Use self-descriptive names, structured responses, and actionable error messages. `rai_perception` demonstrates this through semantic parameter names (`outlier_fraction` vs `if_contamination`) and preset names that are self-documenting.

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

7. **Rachwał et al. (2025)** - "RAI: Flexible Agent Framework for Embodied AI" - https://arxiv.org/abs/2505.07532  
   Introduces the RAI framework for creating embodied Multi Agent Systems for robotics, with integration for ROS 2, Large Language Models, and simulations. Describes the framework's architecture, tools, and mechanisms for agent embodiment.
