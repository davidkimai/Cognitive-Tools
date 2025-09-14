# Cognitive Tools for Large Language Model Reasoning

Minimal practical implementation of "Eliciting Reasoning in Language Models with Cognitive Tools" (Ebouky et al., 2025) - a modular framework that enhances LLM reasoning through cognitive psychology-inspired tools without requiring additional training.

## Core Contribution

This framework demonstrates that **modular cognitive operations** can unlock reasoning capabilities in base language models, achieving near-reasoning-model performance without reinforcement learning:

- **GPT-4.1 baseline**: 26.7% on AIME 2024
- **GPT-4.1 + cognitive tools**: 43.3% on AIME 2024  
- **o1-preview (RL-trained)**: 44.6% on AIME 2024

The approach closes 94% of the performance gap to state-of-the-art reasoning models using only structured prompting.

## Repository Structure

```
cognitive-tools-llm/
├── README.md              # Complete guide and documentation
├── cognitive_tools.py     # Core framework: all tools + orchestrator  
├── reproduce_paper.py     # Exact experimental reproduction
├── demo.py               # Interactive examples (jupyter-style)
├── evaluate.py           # Evaluation framework and metrics
├── benchmarks.json       # Benchmark datasets
└── requirements.txt      # Dependencies
```

This minimal structure maximizes information density while ensuring immediate accessibility for both novice learners and experienced researchers.

## Quick Start

### Installation
```bash
git clone https://github.com/davidkimai/Cognitive-Tools.git
cd Cognitive-Tools
pip install -r requirements.txt
```

### Basic Usage
```python
from cognitive_tools import CognitiveToolsOrchestrator

# Initialize with your LLM interface
orchestrator = CognitiveToolsOrchestrator(llm_interface)

# Solve problems with cognitive tools
result = orchestrator.solve_problem("Find the GCD of 3339, 2961, and 1491.")
print(result)  # Uses understand_question, examine_answer, backtracking tools
```

### Interactive Exploration
```python
# Run jupyter-style demonstrations
python demo.py

# Reproduce paper experiments  
python reproduce_paper.py

# Custom evaluation
from evaluate import CognitiveToolsEvaluator
evaluator = CognitiveToolsEvaluator(llm_interfaces)
result = await evaluator.evaluate_configuration("model", "benchmark", "cognitive_tools")
```

## Cognitive Tools Framework

### Four Core Tools

1. **understand_question**: Problem decomposition and goal management
   - Identifies mathematical concepts and solution approaches
   - Extracts relevant information and constraints
   - Highlights applicable theorems and techniques

2. **recall_related**: Analogical reasoning through examples  
   - Retrieves similar solved problems from knowledge
   - Provides step-by-step solution patterns
   - Guides reasoning through structural similarities

3. **examine_answer**: Self-reflection and verification
   - Checks reasoning traces for logical consistency
   - Identifies miscalculations and wrong assumptions
   - Validates solutions against problem constraints

4. **backtracking**: Alternative path exploration
   - Detects flawed reasoning steps
   - Suggests alternative solution approaches  
   - Enables systematic exploration of solution space

### Architecture Principles

- **Modularity**: Each tool operates independently to prevent interference
- **Flexibility**: LLM decides which tools to use and when
- **Transparency**: Each reasoning step is explicit and traceable
- **Composability**: Tools can be combined in any sequence

## Key Results

### Individual Tool Performance (Smolbenchmark)
| Model | Baseline | Best Tool | Improvement |
|-------|----------|-----------|-------------|
| Qwen2.5-32B | 79.6% | 84.2% | +4.6% |
| Llama3.3-70B | 52.8% | 79.5% | +26.7% |

### Cognitive Tools vs Cognitive Prompting
| Model | Cognitive Prompting | Cognitive Tools | Advantage |
|-------|-------------------|-----------------|-----------|
| Qwen2.5-32B | 82.0% | 88.0% | +6.0% |  
| Llama3.3-70B | 66.0% | 80.0% | +14.0% |

### Main Benchmarks
| Model | Benchmark | Baseline | With Tools | Improvement |
|-------|-----------|----------|------------|-------------|
| Qwen2.5-32B | AIME 2024 | 17.2% | 32.1% | +14.9% |
| Llama3.3-70B | MATH500 | 57.0% | 74.7% | +17.7% |

## Theoretical Foundation

The framework implements cognitive architecture principles from Anderson et al. (1997), translating human reasoning patterns into modular LLM operations:

1. **Goal Management**: Systematic problem decomposition
2. **Memory Retrieval**: Analogical reasoning through examples
3. **Self-Monitoring**: Metacognitive reflection and error detection  
4. **Strategy Selection**: Flexible exploration of solution paths

This approach provides evidence that base language models possess latent reasoning capabilities that structured cognitive workflows can unlock without requiring reinforcement learning.

## Implementation Details

### Tool Execution Flow
```python
# Simplified orchestration logic
def solve_problem(question):
    conversation = [system_prompt, question]
    
    while not finished:
        response = llm.generate(conversation)
        
        if "ANSWER:" in response:
            return response
            
        tool_calls = extract_tool_calls(response)
        for call in tool_calls:
            result = execute_tool(call)
            conversation.append(tool_result)
```

### System Prompt Structure
The orchestrator uses a carefully designed system prompt that:
- Describes available cognitive tools and their functions
- Provides flexibility in tool usage decisions
- Enforces structured output with "ANSWER:" format
- Includes monetary reward incentive for motivation

### Tool Implementation
Each cognitive tool is implemented as:
- **Prompt Template**: Specialized prompts for specific cognitive operations
- **Input Parameters**: Question, context, and reasoning state
- **Output Processing**: Structured responses fed back to main reasoning loop

## Evaluation Methodology

### Metrics
- **Pass@1 Accuracy**: Percentage of problems solved correctly on first attempt
- **Statistical Significance**: Paired t-tests with multiple runs
- **Error Analysis**: Systematic categorization of reasoning failures

### Benchmarks
- **AIME 2024**: 30 problems, high-difficulty mathematical reasoning
- **MATH500**: 500 problems, varied difficulty with complex expressions
- **AMC**: 83 problems, medium-difficulty competition mathematics
- **Smolbenchmark**: 50 problems, diverse mathematical tasks

### Evaluation Pipeline
```python
# Statistical evaluation with confidence intervals
result = evaluator.evaluate_configuration(
    model_name="GPT-4.1",
    benchmark_name="AIME2024", 
    configuration="cognitive_tools",
    num_runs=8
)
print(f"Accuracy: {result.accuracy:.1f}% ± {result.std_error:.1f}")
```

## Research Applications

### Immediate Applications
- **Mathematical Reasoning**: Enhanced problem-solving on competition mathematics
- **Educational Tools**: Transparent reasoning for learning applications  
- **Research Analysis**: Systematic exploration of reasoning capabilities

### Extension Possibilities
- **Domain-Specific Tools**: Custom cognitive operations for specialized fields
- **Multi-Modal Reasoning**: Integration with visual and symbolic reasoning
- **Hybrid Approaches**: Combination with reinforcement learning methods

## Limitations and Future Work

### Current Limitations
- **Domain Scope**: Primarily evaluated on mathematical reasoning tasks
- **Manual Design**: Tools are hand-crafted rather than automatically discovered
- **Model Dependence**: Prompts optimized for specific model families

### Future Directions
- **Automated Tool Discovery**: Machine learning approaches to identify effective cognitive operations
- **Broader Domains**: Extension to scientific reasoning, logical inference, and planning tasks
- **Scale Studies**: Evaluation on larger model architectures and parameter counts

## Citation

```bibtex
@article{ebouky2025cognitive,
  title={Eliciting Reasoning in Language Models with Cognitive Tools},
  author={Ebouky, Brown and Bartezzaghi, Andrea and Rigotti, Mattia},
  journal={arXiv preprint arXiv:2506.12115},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

This repository prioritizes **accessible minimalism** and **high information density**. Contributions should:

1. Maintain the flat, simple repository structure
2. Follow the jupyter notebook style for new demonstrations  
3. Include comprehensive documentation within code files
4. Provide quantitative validation against paper results
5. Ensure accessibility for both novice and expert users

For questions or contributions, please open an issue or pull request.
