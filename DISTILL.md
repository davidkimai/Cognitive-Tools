# Eliciting Reasoning in Language Models with Cognitive Tools

## Title
Eliciting Reasoning in Language Models with Cognitive Tools

## Hypotheses
1. **Modularity Hypothesis**: Modular cognitive operations implemented as discrete tools reduce interference between reasoning steps compared to monolithic prompting approaches
2. **Latent Capability Hypothesis**: Base LLMs possess inherent reasoning capabilities that can be elicited through structured cognitive workflows without requiring reinforcement learning
3. **Cognitive Architecture Hypothesis**: Human-inspired cognitive operations (goal management, memory recall, self-reflection, backtracking) can be effectively translated to LLM reasoning frameworks

## Executive Summary
This research introduces an alternative to reinforcement learning for eliciting reasoning in Large Language Models by implementing modular "cognitive tools" based on cognitive psychology principles. The approach implements four core cognitive operations as discrete tools within a modern agentic framework: understand question, recall related examples, examine current reasoning, and backtracking. Results demonstrate significant performance improvements on mathematical reasoning benchmarks, with GPT-4.1 enhanced with cognitive tools achieving 43.3% accuracy on AIME2024 compared to 26.7% baseline, approaching o1-preview performance (44.6%).

## Methods
### Cognitive Tools Framework
- **Architecture**: Tool-calling framework where each cognitive operation is encapsulated as a self-contained function
- **Implementation**: Each tool executed by the same LLM with specialized prompts in sandboxed contexts
- **Modularity**: Operations isolated to prevent interference, allowing flexible orchestration

### Four Core Cognitive Tools
1. **Understand Question**: Breaks down problems by identifying key concepts, extracting relevant information, and highlighting applicable techniques
2. **Recall Related**: Provides analogous solved problems to guide reasoning through similar examples
3. **Examine Answer**: Implements self-reflection to check reasoning traces for flaws, assumptions, and missed constraints
4. **Backtracking**: Enables exploration of alternative reasoning paths when current approaches fail

### Experimental Design
- **Models**: Open-weight (Qwen2.5-7B/32B, Llama3.1-8B, Llama3.3-70B) and closed (GPT-4.1, o1-preview)
- **Benchmarks**: AIME 2024 (30 samples), MATH500 (500 problems), AMC (83 problems), Smolbenchmark (50 samples)
- **Evaluation**: Pass@1 accuracy with parsing for numerical answers, LLM-as-judge (GPT-4.1) for complex expressions
- **Baselines**: Vanilla models, cognitive prompting comparison

## Results
### Individual Tool Performance
- All cognitive tools consistently improve performance over baseline across models
- Backtracking tool showed largest improvement on Llama3.3-70B (+26.7% on Smolbenchmark)
- Different tools achieve optimal performance for different model architectures

### Cognitive Tools vs Cognitive Prompting
- Modular cognitive tools consistently outperform monolithic cognitive prompting
- Performance increases range from +4.2% (Qwen2.5-7B) to +27.2% (Llama3.3-70B)

### Main Benchmark Results
- **AIME 2024**: Qwen2.5-32B improved from 17.2% to 32.1% (+14.9%)
- **MATH500**: Significant improvements across all models (7-11% average gain)
- **AMC**: Consistent improvements ranging from 7-10% across model sizes

### Closed Model Comparison
- GPT-4.1 baseline: 26.7% on AIME 2024
- GPT-4.1 + cognitive tools: 43.3% (+16.6%)
- o1-preview (RL-trained): 44.6%
- Cognitive tools approach within 1.3% of state-of-the-art reasoning model

## Statistical Summaries
### Performance Improvements by Model Class
- **Small Models (7-8B)**: Average improvement of 6.1% across benchmarks
- **Medium Models (32B)**: Average improvement of 10.9% across benchmarks  
- **Large Models (70B)**: Average improvement of 17.4% across benchmarks
- **Closed Models (GPT-4.1)**: 16.6% improvement on AIME 2024

### Tool Usage Patterns
- understand_question: Most effective for complex problem decomposition
- recall_related: Strongest performance on pattern-matching tasks
- examine_answer: Critical for error detection and correction
- backtracking: Essential for exploring alternative solution paths

## Code Implementations or Validations
Complete implementations provided for:
- Modular cognitive tools as Python classes
- Tool-calling orchestration framework
- Evaluation pipeline matching paper benchmarks
- Baseline comparison implementations
- Cognitive prompting baseline implementation

## Discussion
The research provides compelling evidence that modular cognitive tools can elicit reasoning capabilities comparable to reinforcement learning approaches. Key insights include:

1. **Modularity Benefits**: Isolated cognitive operations reduce interference and enable flexible reasoning strategies
2. **Base Model Capabilities**: Results support the hypothesis that base models possess latent reasoning abilities that structured approaches can unlock
3. **Practical Implications**: Method requires no additional training, making it immediately applicable to existing models

## Limitations
- **Domain Scope**: Evaluation limited primarily to mathematical reasoning benchmarks
- **Manual Design**: Cognitive tools manually crafted, lacking automated discovery mechanisms
- **Model Specificity**: Prompts tailored to tested model families, may require adaptation for other architectures
- **Scale Evaluation**: Limited testing on models beyond 70B parameters

## Implications
### Theoretical
- Supports theories of latent reasoning capabilities in base language models
- Provides evidence for cognitive architecture principles in artificial systems
- Challenges necessity of reinforcement learning for reasoning tasks

### Practical
- Immediate applicability to existing deployed models without retraining
- Enhanced interpretability through explicit reasoning step identification
- Democratized access to advanced reasoning without computational overhead of RL

## Future Directions
1. **Automated Tool Discovery**: Develop methods for automatically identifying effective cognitive operations
2. **Domain Extension**: Apply framework to reasoning tasks beyond mathematics (scientific reasoning, logical inference, planning)
3. **Integration with RL**: Explore hybrid approaches combining cognitive tools with reinforcement learning
4. **Scale Studies**: Evaluate effectiveness on larger model architectures and parameter counts
5. **Tool Composition**: Investigate optimal combinations and sequences of cognitive tools

## Data Availability
- AIME 2024: Publicly available through Mathematical Association of America
- MATH500: Available through original Hendrycks et al. paper
- AMC: Curated collection available through AI-MO project
- Smolbenchmark: Available through HuggingFace datasets

## References
Primary citations include Anderson et al. (1997) for cognitive architecture foundations, Kramer and Baumann (2024) for cognitive prompting, and extensive recent work on reasoning in language models including OpenAI's o1 and DeepSeek-R1 developments.

## Reproducibility Checklist
- [x] Complete methodology description provided
- [x] Hyperparameters and experimental settings specified
- [x] Evaluation metrics and procedures detailed
- [x] Statistical significance testing methodology described
- [x] Code implementation details provided in appendix
- [x] Tool prompts fully specified for replication
- [x] Baseline comparison methods documented
