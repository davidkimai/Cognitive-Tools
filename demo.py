"""
Interactive Demonstration: Cognitive Tools for LLM Reasoning

This notebook-style demonstration shows how cognitive psychology-inspired tools
enhance mathematical reasoning in language models without requiring training.

JUPYTER/COLAB USAGE:
    from demo import run_comparison_demo, demonstrate_individual_tools, show_paper_results
    await run_comparison_demo()  # Compare baseline vs cognitive tools
    demonstrate_individual_tools()  # Show each tool individually
    show_paper_results()  # Display key quantitative results
    
Execute functions individually in cells to see progressive demonstrations.
"""

import json
import asyncio
from typing import Dict, List, Tuple
from cognitive_tools import (
    CognitiveToolsOrchestrator, 
    UnderstandQuestionTool,
    RecallRelatedTool, 
    ExamineAnswerTool,
    BacktrackingTool,
    parse_answer
)


# Core Theory and Setup Section
def show_framework_overview():
    """
    Display framework overview and core theory
    
    JUPYTER USAGE:
        from demo import show_framework_overview
        show_framework_overview()
    """
    print("COGNITIVE TOOLS FRAMEWORK")
    print("Paper: Eliciting Reasoning in Language Models with Cognitive Tools (Ebouky et al., 2025)")
    print("=" * 70)
    print()
    print("Core Theory:")
    print("Cognitive tools implement four key reasoning operations based on cognitive psychology:")
    print("- understand_question: Problem decomposition and goal management")
    print("- recall_related: Analogical reasoning through example retrieval")
    print("- examine_answer: Self-reflection and error detection") 
    print("- backtracking: Alternative path exploration when reasoning fails")
    print()
    print("Architecture Principles:")
    print("- Modularity: Each tool operates independently to prevent interference")
    print("- Flexibility: LLM decides which tools to use and when")
    print("- Transparency: Each reasoning step is explicit and traceable")
    print("- Composability: Tools can be combined in any sequence")


class DemoLLM:
    """
    Simulated LLM responses showing baseline vs cognitive tools differences
    
    JUPYTER USAGE:
        demo_llm = DemoLLM("Demo-GPT-4")
        response = await demo_llm.generate([{"role": "user", "content": "What is 2+2?"}])
        print(response)
    """
    
    def __init__(self, model_name: str = "Demo-GPT-4"):
        self.model_name = model_name
        
        # Realistic responses based on paper examples
        self.problem_responses = {
            "gcd_3339_2961_1491": {
                "baseline": """To find the GCD of 3339, 2961, and 1491, I'll use the Euclidean algorithm.

GCD(3339, 2961):
3339 = 2961 × 1 + 378
2961 = 378 × 7 + 375  
378 = 375 × 1 + 3
375 = 3 × 125 + 0

So GCD(3339, 2961) = 3

GCD(3, 1491):
1491 = 3 × 497 + 0

Therefore GCD(3339, 2961, 1491) = 3

ANSWER: 3""",
                
                "with_tools": """I'll approach this systematically using cognitive tools.

understand_question()

This requires finding the greatest common divisor of three integers. I should consider multiple approaches: Euclidean algorithm, prime factorization, or systematic division. The problem involves number theory concepts.

Let me work through this step by step:
GCD(3339, 2961):
3339 = 2961 × 1 + 378
2961 = 378 × 7 + 375
378 = 375 × 1 + 3  
375 = 3 × 125 + 0

This gives GCD(3339, 2961) = 3

examine_answer()

Wait, let me verify by checking if these numbers have larger common factors. Let me try prime factorization:

3339 = 3 × 1113 = 3 × 3 × 371 = 3² × 371
Let me check: 371 = 7 × 53

2961: Let me divide by small primes
2961 ÷ 3 = 987
987 ÷ 3 = 329  
329 ÷ 7 = 47

So 2961 = 3² × 7 × 47

1491 ÷ 3 = 497
497 ÷ 7 = 71

So 1491 = 3 × 7 × 71

backtracking()

My initial calculation was wrong. Looking at the prime factorizations:
3339 = 3² × 7 × 53
2961 = 3² × 7 × 47  
1491 = 3 × 7 × 71

The common factors are 3 × 7 = 21

ANSWER: 21"""
            },
            
            "combinatorics_circular": {
                "baseline": """For circular arrangements, we fix one person and arrange the rest.
With 5 people, fix one person, then arrange the remaining 4 people.
This gives us 4! = 24 arrangements.

ANSWER: 24""",
                
                "with_tools": """understand_question()

This is a circular permutation problem. In circular arrangements, rotations are considered identical, so we fix one position to eliminate rotational symmetry.

recall_related()

Similar problems: arranging n distinct objects in a circle gives (n-1)! arrangements because we can fix one object and arrange the remaining (n-1) objects linearly.

With 5 people in a circle:
Fix person A in one position
Arrange remaining 4 people: 4! = 4 × 3 × 2 × 1 = 24

examine_answer()

This is correct for circular permutations where people are distinguishable and we only consider rotations as identical (not reflections).

ANSWER: 24"""
            },
            
            "basic_arithmetic": {
                "baseline": "2 + 2 = 4\n\nANSWER: 4",
                "with_tools": "This is a simple arithmetic problem. 2 + 2 = 4\n\nANSWER: 4"
            }
        }
    
    async def generate(self, messages: List[Dict], **kwargs) -> str:
        """Generate appropriate response based on context"""
        conversation_text = " ".join([msg["content"].lower() for msg in messages])
        
        # Detect problem type and approach
        if "3339" in conversation_text and "2961" in conversation_text:
            problem_key = "gcd_3339_2961_1491"
        elif "circular table" in conversation_text or "circular" in conversation_text:
            problem_key = "combinatorics_circular"
        elif "2+2" in conversation_text or "what is 2+2" in conversation_text:
            problem_key = "basic_arithmetic"
        else:
            return "ANSWER: [Demo response - problem type not recognized]"
        
        # Determine if using tools based on system prompt presence
        use_tools = any("cognitive tools" in msg["content"].lower() or 
                       "understand_question" in msg["content"].lower()
                       for msg in messages)
        
        if problem_key in self.problem_responses:
            approach = "with_tools" if use_tools else "baseline"
            return self.problem_responses[problem_key][approach]
        
        return "ANSWER: [Demo response]"


# Sample Problems for Analysis
SAMPLE_PROBLEMS = {
    "gcd_hard": {
        "question": "Find the greatest common divisor of 3339, 2961, and 1491.",
        "correct_answer": "21",
        "difficulty": "Hard",
        "concepts": ["Number theory", "Euclidean algorithm", "Prime factorization"],
        "common_errors": ["Stopping at GCD=3", "Calculation mistakes in algorithm"]
    },
    
    "circular_permutation": {
        "question": "In how many ways can 5 people sit around a circular table?",
        "correct_answer": "24", 
        "difficulty": "Medium",
        "concepts": ["Combinatorics", "Circular permutations", "Symmetry"],
        "common_errors": ["Using 5! instead of 4!", "Confusing with linear arrangements"]
    },
    
    "quadratic_factoring": {
        "question": "Solve x² - 7x + 12 = 0",
        "correct_answer": "x = 3, 4",
        "difficulty": "Easy",
        "concepts": ["Algebra", "Quadratic equations", "Factoring"],
        "common_errors": ["Sign errors", "Arithmetic mistakes"]
    }
}


def display_problem_info(problem_key: str):
    """
    Display comprehensive problem information
    
    JUPYTER USAGE:
        from demo import display_problem_info
        display_problem_info("gcd_hard")
    """
    problem = SAMPLE_PROBLEMS[problem_key]
    print(f"Problem: {problem['question']}")
    print(f"Difficulty: {problem['difficulty']}")
    print(f"Concepts: {', '.join(problem['concepts'])}")
    print(f"Correct Answer: {problem['correct_answer']}")
    print(f"Common Errors: {', '.join(problem['common_errors'])}")


# Baseline vs Cognitive Tools Comparison
async def run_comparison_demo(problem_key: str = "gcd_hard"):
    """
    Compare baseline vs cognitive tools on specific problem
    
    JUPYTER USAGE:
        await run_comparison_demo("gcd_hard")
        await run_comparison_demo("circular_permutation")
    """
    
    demo_llm = DemoLLM()
    orchestrator = CognitiveToolsOrchestrator(demo_llm)
    
    problem = SAMPLE_PROBLEMS[problem_key]
    question = problem["question"]
    
    print("REASONING APPROACH COMPARISON")
    print("=" * 60)
    
    display_problem_info(problem_key)
    print()
    
    # Baseline Approach
    print("BASELINE APPROACH:")
    print("-" * 25)
    
    baseline_messages = [{"role": "user", "content": f"Solve: {question}"}]
    baseline_response = await demo_llm.generate(baseline_messages)
    print(baseline_response)
    
    baseline_answer = parse_answer(baseline_response)
    baseline_correct = baseline_answer == problem['correct_answer']
    
    print(f"\nExtracted Answer: {baseline_answer}")
    print(f"Result: {'CORRECT' if baseline_correct else 'INCORRECT'}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Cognitive Tools Approach  
    print("COGNITIVE TOOLS APPROACH:")
    print("-" * 30)
    
    tools_messages = [
        {"role": "system", "content": orchestrator.get_system_prompt()},
        {"role": "user", "content": f"Solve: {question}"}
    ]
    tools_response = await demo_llm.generate(tools_messages)
    print(tools_response)
    
    tools_answer = parse_answer(tools_response)
    tools_correct = tools_answer == problem['correct_answer']
    
    print(f"\nExtracted Answer: {tools_answer}")
    print(f"Result: {'CORRECT' if tools_correct else 'INCORRECT'}")
    
    # Analysis
    print("\n" + "=" * 40)
    print("COMPARISON ANALYSIS:")
    print("=" * 40)
    print(f"Baseline: {'CORRECT' if baseline_correct else 'INCORRECT'}")
    print(f"Cognitive Tools: {'CORRECT' if tools_correct else 'INCORRECT'}")
    
    if tools_correct and not baseline_correct:
        print("SUCCESS: Cognitive tools corrected baseline error")
    elif tools_correct and baseline_correct:
        print("BOTH CORRECT: Cognitive tools provided more thorough reasoning")
    else:
        print("Both approaches need refinement")
    
    return baseline_correct, tools_correct


# Individual Tool Analysis
def analyze_individual_tool(tool_name: str, tool_instance, question: str):
    """
    Analyze what each cognitive tool contributes
    
    JUPYTER USAGE:
        from demo import analyze_individual_tool
        from cognitive_tools import UnderstandQuestionTool
        analyze_individual_tool("understand_question", UnderstandQuestionTool(), "Find GCD of 48 and 18")
    """
    
    print(f"COGNITIVE TOOL: {tool_name.upper().replace('_', ' ')}")
    print("-" * 40)
    print(f"Function: {tool_instance.description}")
    print()
    
    # Execute tool with appropriate parameters
    if tool_name == "examine_answer":
        response = tool_instance.execute(
            question=question,
            current_reasoning="Used Euclidean algorithm to find GCD = 3"
        )
    elif tool_name == "backtracking":
        response = tool_instance.execute(
            question=question,
            current_reasoning="Current approach yields GCD = 3, but this may be incomplete"
        )
    else:
        response = tool_instance.execute(question=question)
    
    # Display truncated response for readability
    content = response.content
    if "TOOL_PROMPT:" in content:
        # Extract just the prompt structure for demonstration
        lines = content.split('\n')
        relevant_lines = [line for line in lines if not line.startswith("TOOL_PROMPT:")][:15]
        content = '\n'.join(relevant_lines)
        if len(relevant_lines) >= 15:
            content += "\n[Prompt continues...]"
    
    print("Tool Output Structure:")
    print(content[:500] + "..." if len(content) > 500 else content)
    print()


def demonstrate_individual_tools(question: str = None):
    """
    Demonstrate each cognitive tool individually
    
    JUPYTER USAGE:
        from demo import demonstrate_individual_tools
        demonstrate_individual_tools()  # Uses default GCD problem
        demonstrate_individual_tools("Solve x² - 5x + 6 = 0")  # Custom problem
    """
    
    if question is None:
        question = SAMPLE_PROBLEMS["gcd_hard"]["question"]
    
    print("INDIVIDUAL TOOL CONTRIBUTIONS")
    print("=" * 50)
    print(f"Problem: {question}")
    print()

    tools_to_analyze = [
        ("understand_question", UnderstandQuestionTool()),
        ("recall_related", RecallRelatedTool()),
        ("examine_answer", ExamineAnswerTool()),
        ("backtracking", BacktrackingTool())
    ]

    for tool_name, tool_instance in tools_to_analyze:
        analyze_individual_tool(tool_name, tool_instance, question)
        print()


# Paper Results Validation
PAPER_RESULTS = {
    "individual_tools_smolbenchmark": {
        "Qwen2.5-32B": {
            "baseline": 79.6,
            "understand_question": 82.5,
            "recall_related": 84.2,
            "examine_answer": 84.0,
            "backtracking": 82.9
        }
    },
    "main_benchmarks": {
        "GPT-4.1_AIME2024": {
            "baseline": 26.7,
            "cognitive_tools": 43.3,
            "o1_preview": 44.6
        }
    }
}


def show_paper_results():
    """
    Display key quantitative results from the paper
    
    JUPYTER USAGE:
        from demo import show_paper_results
        show_paper_results()
    """
    
    print("PAPER RESULTS SUMMARY")
    print("=" * 40)
    
    print("Individual Tool Performance (Smolbenchmark):")
    print("Model: Qwen2.5-32B Instruct")
    
    smol_results = PAPER_RESULTS["individual_tools_smolbenchmark"]["Qwen2.5-32B"]
    baseline = smol_results["baseline"]
    
    for tool, accuracy in smol_results.items():
        if tool != "baseline":
            improvement = accuracy - baseline
            print(f"  {tool.replace('_', ' ').title()}: {accuracy:.1f}% (+{improvement:.1f}%)")
    
    print(f"  Baseline: {baseline:.1f}%")
    print()
    
    print("Main Result (AIME 2024):")
    aime_results = PAPER_RESULTS["main_benchmarks"]["GPT-4.1_AIME2024"]
    print(f"  GPT-4.1 Baseline: {aime_results['baseline']:.1f}%")
    print(f"  GPT-4.1 + Cognitive Tools: {aime_results['cognitive_tools']:.1f}%")
    print(f"  o1-preview (RL-trained): {aime_results['o1_preview']:.1f}%")
    
    improvement = aime_results['cognitive_tools'] - aime_results['baseline'] 
    print(f"  Improvement: +{improvement:.1f} percentage points")
    print(f"  Gap to o1-preview: {aime_results['o1_preview'] - aime_results['cognitive_tools']:.1f}%")


# Interactive Problem Solving
class InteractiveSolver:
    """
    Interactive interface for testing cognitive tools on custom problems
    
    JUPYTER USAGE:
        solver = InteractiveSolver()
        answer = await solver.solve_custom_problem("What is 15% of 80?", use_tools=True)
        print(f"Answer: {answer}")
    """
    
    def __init__(self):
        self.demo_llm = DemoLLM()
        self.orchestrator = CognitiveToolsOrchestrator(self.demo_llm)
    
    async def solve_custom_problem(self, question: str, use_tools: bool = True):
        """Solve a custom problem with or without cognitive tools"""
        
        print(f"SOLVING: {question}")
        print(f"METHOD: {'Cognitive Tools' if use_tools else 'Baseline'}")
        print("-" * 50)
        
        if use_tools:
            messages = [
                {"role": "system", "content": self.orchestrator.get_system_prompt()},
                {"role": "user", "content": f"Solve: {question}"}
            ]
        else:
            messages = [{"role": "user", "content": f"Solve: {question}"}]
        
        response = await self.demo_llm.generate(messages)
        print(response)
        
        answer = parse_answer(response)
        print(f"\nExtracted Answer: {answer}")
        
        return answer


# Framework Architecture Explanation
def explain_framework_architecture():
    """
    Explain the cognitive tools framework structure
    
    JUPYTER USAGE:
        from demo import explain_framework_architecture
        explain_framework_architecture()
    """
    
    print("COGNITIVE TOOLS FRAMEWORK ARCHITECTURE")
    print("=" * 50)
    print()
    
    print("Core Components:")
    print("1. CognitiveTool (Abstract Base Class)")
    print("   - Defines interface for all cognitive operations")
    print("   - Ensures consistent tool behavior and integration")
    print()
    
    print("2. Individual Tool Implementations:")
    print("   - UnderstandQuestionTool: Problem decomposition")
    print("   - RecallRelatedTool: Analogical reasoning")  
    print("   - ExamineAnswerTool: Self-reflection")
    print("   - BacktrackingTool: Alternative exploration")
    print()
    
    print("3. CognitiveToolsOrchestrator:")
    print("   - Manages tool calling and execution")
    print("   - Integrates with LLM interfaces")
    print("   - Handles conversation flow and tool responses")
    print()
    
    print("Key Design Principles:")
    print("- Modularity: Each tool is independent and composable")
    print("- Flexibility: LLM decides which tools to use when")
    print("- Transparency: Each reasoning step is explicit")
    print("- Compatibility: Works with existing LLMs without training")


# Performance Analysis
def analyze_performance_patterns():
    """
    Analyze performance improvement patterns from paper
    
    JUPYTER USAGE:
        from demo import analyze_performance_patterns
        analyze_performance_patterns()
    """
    
    print("PERFORMANCE IMPROVEMENT ANALYSIS")
    print("=" * 45)
    print()
    
    # Model size vs improvement correlation
    model_improvements = {
        "Llama3.1-8B": {"baseline": 23.1, "tools": 29.2, "improvement": 6.1},
        "Qwen2.5-32B": {"baseline": 48.0, "tools": 58.9, "improvement": 10.9},
        "Llama3.3-70B": {"baseline": 34.4, "tools": 51.8, "improvement": 17.4}
    }
    
    print("Model Size vs Improvement Correlation:")
    for model, results in model_improvements.items():
        size = model.split('-')[1]  # Extract size
        print(f"  {model} ({size}): +{results['improvement']:.1f}% average improvement")
    
    print("\nKey Observations:")
    print("- Larger models show greater improvements with cognitive tools")
    print("- Improvements range from 6-17% across different model sizes")
    print("- All model sizes benefit consistently from cognitive tools")
    print()
    
    # Tool-specific contributions
    print("Individual Tool Contributions (Qwen2.5-32B on Smolbenchmark):")
    tools_data = {
        "understand_question": 2.9,
        "recall_related": 4.6,
        "examine_answer": 4.4,
        "backtracking": 3.3
    }
    
    for tool, improvement in tools_data.items():
        print(f"  {tool.replace('_', ' ').title()}: +{improvement:.1f}%")


# Complete Demo Runner
async def run_complete_demo():
    """
    Run the complete demonstration sequence
    
    JUPYTER USAGE:
        from demo import run_complete_demo
        await run_complete_demo()
    """
    
    print("COGNITIVE TOOLS COMPLETE DEMONSTRATION")
    print("=" * 60)
    print()
    
    # 1. Framework Overview
    show_framework_overview()
    print("\n" + "="*50 + "\n")
    
    # 2. Individual Tools
    print("SECTION 1: INDIVIDUAL TOOLS")
    demonstrate_individual_tools()
    print("\n" + "="*50 + "\n")
    
    # 3. Comparison Demo
    print("SECTION 2: BASELINE VS COGNITIVE TOOLS")
    await run_comparison_demo("gcd_hard")
    print("\n" + "="*50 + "\n")
    
    # 4. Paper Results
    print("SECTION 3: PAPER RESULTS")
    show_paper_results()
    print("\n" + "="*50 + "\n")
    
    # 5. Performance Analysis
    print("SECTION 4: PERFORMANCE ANALYSIS")
    analyze_performance_patterns()
    print("\n" + "="*50 + "\n")
    
    # 6. Architecture
    print("SECTION 5: FRAMEWORK ARCHITECTURE")
    explain_framework_architecture()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print()
    print("Key Takeaways:")
    print("1. Cognitive tools consistently improve reasoning accuracy")
    print("2. Modular design allows flexible reasoning strategies")
    print("3. No training required - works with existing models")
    print("4. Significant performance gains approaching RL-trained models")
    print()
    print("Next Steps:")
    print("- Run reproduce_paper.py for full experimental replication")
    print("- Use evaluate.py for custom benchmark evaluation") 
    print("- Extend cognitive_tools.py with domain-specific tools")


# Make key components easily accessible for imports
__all__ = [
    'show_framework_overview',
    'run_comparison_demo',
    'demonstrate_individual_tools', 
    'show_paper_results',
    'analyze_performance_patterns',
    'explain_framework_architecture',
    'run_complete_demo',
    'InteractiveSolver',
    'DemoLLM',
    'SAMPLE_PROBLEMS',
    'display_problem_info'
]
