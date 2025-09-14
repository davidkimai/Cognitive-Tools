"""
Cognitive Tools for Large Language Model Reasoning

Implementation of modular cognitive operations based on cognitive psychology
principles for enhancing LLM reasoning without reinforcement learning.

Based on: "Eliciting Reasoning in Language Models with Cognitive Tools" 
(Ebouky et al., 2025)
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ToolCall:
    """Represents a cognitive tool function call"""
    name: str
    parameters: Dict[str, Any]
    

@dataclass 
class ToolResponse:
    """Response from executing a cognitive tool"""
    content: str
    success: bool = True
    metadata: Dict[str, Any] = None


class CognitiveTool(ABC):
    """Abstract base class for cognitive tools"""
    
    @abstractmethod
    def execute(self, question: str, context: str = "", **kwargs) -> ToolResponse:
        """Execute the cognitive tool with given inputs"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for function calling"""
        pass
    
    @property 
    @abstractmethod
    def description(self) -> str:
        """Tool description and usage guidelines"""
        pass


class UnderstandQuestionTool(CognitiveTool):
    """
    Goal management and problem decomposition tool.
    Breaks down problems by identifying key concepts and solution approaches.
    """
    
    @property
    def name(self) -> str:
        return "understand_question"
    
    @property
    def description(self) -> str:
        return "Analyze and break down complex problems into structured steps"
    
    def execute(self, question: str, context: str = "", **kwargs) -> ToolResponse:
        prompt = f"""You are a mathematical reasoning assistant designed to analyze and break down complex mathematical problems into structured steps to help the system that actually solves problems. Your goal is to:

1. Identify the core mathematical concepts involved (e.g., algebra, calculus, linear algebra).
2. Extract and categorize relevant symbols, variables, and functions.
3. Rephrase the problem into a step-by-step sequence that makes solving easier.
4. Highlight any known theorems or techniques that might be useful in solving the problem.
5. DO NOT provide any answer to the question, only provide instructions which will guide the upstream system.

Question: {question}

{f'Context: {context}' if context else ''}

Provide a structured analysis following the guidelines above."""

        # In practice, this would call the LLM with the prompt
        # For now, return the prompt structure
        return ToolResponse(
            content=f"TOOL_PROMPT: understand_question\n{prompt}",
            metadata={"tool": "understand_question", "prompt": prompt}
        )


class RecallRelatedTool(CognitiveTool):
    """
    Knowledge retrieval tool providing analogous solved problems.
    Guides reasoning through similar examples and solution patterns.
    """
    
    @property
    def name(self) -> str:
        return "recall_related"
    
    @property
    def description(self) -> str:
        return "Provide solved examples of analogous problems to guide reasoning"
    
    def execute(self, question: str, context: str = "", **kwargs) -> ToolResponse:
        prompt = f"""You are a retrieval assistant whose purpose is to help solve new mathematical problems by providing solved examples of analogous problems.

Given a new math problem, your task is to:
1. Identify 2 or 3 **similar problems** from your knowledge or training set that require **comparable mathematical concepts or reasoning steps**.
2. For each similar problem:
   - Provide the **full problem statement**.
   - Provide a **complete step-by-step solution**, including relevant formulas, simplifications, or code.
   - Highlight the **final answer**, preferably using LaTeX formatting (e.g., $42$).

Do **not** solve the current problem. Instead, present only useful analogous examples that could help someone reason through it.

Output Format:
Analogous Example 1:
Q: [Similar Problem 1]
A: [Step-by-step solution...]
Final Answer: ...

Analogous Example 2:
Q: [Similar Problem 2] 
A: [Step-by-step solution...]
Final Answer: ...

Analogous Example 3:
Q: [Similar Problem 3]
A: [Step-by-step solution...]
Final Answer: ...

Some important notes to keep in mind:
- Select examples with strong structural or conceptual similarity, not just keyword overlap.
- Variation in surface details (numbers, variable names) is acceptable as long as the mathematical logic aligns.

Question: {question}

{f'Context: {context}' if context else ''}"""

        return ToolResponse(
            content=f"TOOL_PROMPT: recall_related\n{prompt}",
            metadata={"tool": "recall_related", "prompt": prompt}
        )


class ExamineAnswerTool(CognitiveTool):
    """
    Self-reflection and verification tool.
    Examines reasoning traces for errors, inconsistencies, and missed constraints.
    """
    
    @property
    def name(self) -> str:
        return "examine_answer"
    
    @property
    def description(self) -> str:
        return "Examine current reasoning trace for errors and improvements"
    
    def execute(self, question: str, context: str = "", current_reasoning: str = "", **kwargs) -> ToolResponse:
        prompt = f"""You are an expert mathematical assistant tasked with **verifying and improving** solutions to complex mathematical problems. Your role is **not to solve the problem** but to critically analyze the provided solution for correctness, clarity, and completeness. You will be given a problem/question and the current reasoning that has been produced so far.

### **Your Task:**
Follow a structured **verification process**:

### **1. Understanding the Problem**
- Ensure the proposed solution correctly interprets the given problem.
- Identify the core mathematical concepts involved (e.g., algebra, calculus, number theory).
- Extract and categorize relevant symbols, variables, and functions.
- Identify any implicit assumptions or missing constraints.

### **2. Verifying the Given Solution**
- Clearly state what is the current answer of the problem.
- Break the provided solution down into distinct logical steps.
- Check for **logical consistency**, **mathematical correctness**, and **proper justification**.
- Identify any **miscalculations, incorrect assumptions, or unjustified leaps** in reasoning.
- Analyze the **edge cases** or conditions where the solution may fail.
- Evaluate whether all necessary steps and justifications are present.

#### **2.a) Testing and Validation (Problem-Derived Checks)**
- Examine the original problem statement and extract any **constraints, conditions, identities, or testable properties** that a correct answer must satisfy.
- Derive **test cases or evaluation criteria** based on those constraints.

**If the proposed solution is a numerical answer:**
- Plug the number into the original equation(s), inequality, or scenario to verify it satisfies all conditions.
- Check whether it meets qualitative criteria (e.g., smallest, largest, integer, range bounds).

**If the proposed solution is an expression or formula:**
- **Symbolically substitute** the expression into the original problem statement or equations, and confirm that it satisfies all requirements.
- Simplify or manipulate the expression to check **equivalence**, **domain correctness**, and **edge cases**.
- Where applicable, test the expression against representative sample inputs derived from the problem.

**For both cases:**
- Clearly describe each test performed and the outcome.
- State whether the provided answer (number or expression) **passes all derived problem-based tests**.

### **3. Suggesting Improvements**
- If an error is found, explain **precisely what is wrong** and **why**.
- Suggest possible fixes or improvements **without directly solving the problem**.
- Propose alternative methods to solve the problem where relevant (e.g., algebraic vs. numerical, direct proof vs. counterexample).

### **4. Providing a Judgment**
- Clearly state whether the proposed solution is **correct or incorrect**.
- Justify your judgment with a concise explanation.
- If incorrect, **recommend corrections** without providing a direct answer.

### **Guidelines to Follow:**
- DO NOT provide the actual answer to the problem.
- Focus only on verifying and critiquing the given solution.
- Be rigorous in checking correctness but also constructive in suggesting improvements.
- Explicitly say whether the answer is correct or incorrect

Question: {question}

{f'Context: {context}' if context else ''}

Current Reasoning Trace:
{current_reasoning}

Now, **critically analyze the solution**, highlight any mistakes, and suggest improvements where necessary."""

        return ToolResponse(
            content=f"TOOL_PROMPT: examine_answer\n{prompt}",
            metadata={"tool": "examine_answer", "prompt": prompt}
        )


class BacktrackingTool(CognitiveTool):
    """
    Alternative path exploration tool.
    Identifies reasoning errors and suggests alternative solution approaches.
    """
    
    @property
    def name(self) -> str:
        return "backtracking"
    
    @property
    def description(self) -> str:
        return "Backtrack from flawed reasoning and explore alternative solution paths"
    
    def execute(self, question: str, context: str = "", current_reasoning: str = "", **kwargs) -> ToolResponse:
        prompt = f"""You are a careful problem-solving assistant with the ability to backtrack from flawed logic.

You will be given a math or logic problem and a reasoning trace. Your task is to:
1. Analyze the reasoning and summarize it into different steps.
2. Identify where the first error, bad assumption, or confusion occurs (if any).
3. Propose how to revise the approach from that point onward, using the steps that you have defined.
4. If the entire approach was invalid, suggest a better strategy from scratch.

Use the following format for your response:

**Identified Issues:**
- Step X: Explain what is incorrect or suboptimal.
- (Repeat for any additional steps if needed.)

**Backtrack Point:**
- Indicate the step where reasoning was still valid and you can continue from.

**Revised Strategy (from backtrack point or new):**
- Present a step-by-step strategy to solve the problem correctly from this point.

Be precise and critical. Avoid vague judgments. Always backtrack to the most recent correct step, unless no step is valid.

Question: {question}

{f'Context: {context}' if context else ''}

Current Reasoning Trace:
{current_reasoning}

Analyze the reasoning trace and provide guidance for backtracking and alternative approaches."""

        return ToolResponse(
            content=f"TOOL_PROMPT: backtracking\n{prompt}",
            metadata={"tool": "backtracking", "prompt": prompt}
        )


class CognitiveToolsOrchestrator:
    """
    Main orchestrator for cognitive tools reasoning framework.
    Manages tool calling, execution flow, and response integration.
    """
    
    def __init__(self, llm_interface=None):
        self.tools = {
            "understand_question": UnderstandQuestionTool(),
            "recall_related": RecallRelatedTool(),
            "examine_answer": ExamineAnswerTool(),
            "backtracking": BacktrackingTool()
        }
        self.llm_interface = llm_interface
        
    def get_system_prompt(self) -> str:
        """Return the main cognitive tools system prompt"""
        tools_signature = self._generate_tools_signature()
        
        return f"""You are an expert assistant who solves problems thoughtfully and effectively. You have access to a list of tools — these are Python-based functions that you can call to help you reason through or solve the problem more efficiently.

You are encouraged to use tools when they make the task easier, clearer or more robust — especially for complex, elaborated or ambiguous questions.

Use your best judgment to decide when to call tools.

You may call tools at any point in your reasoning process. Only use the tools listed below. If you choose to use a tool, describe your reasoning and clearly call it using their name.

You can solve problems however you find most appropriate.

When you are ready to provide the final answer to the problem or the question always follow the syntax: 'ANSWER: answer'.

You only have access to these tools, do not use any others:
{tools_signature}

Here are the rules you should always follow to solve your task:
1. **Call a tool when needed.** If you call a tool, only use the available ones and use its full name to do so.
2. ONLY USE Python to call an available tool and not for something else.
3. Don't give up! You're in charge of solving the problem.
4. Do not give an answer without reasoning about it.
5. **Never hallucinate results.** Wait for tool responses before continuing.
6. **Only write your final answer** after you are confident, and always in the form: 'ANSWER: your final answer here'

If the question is already clear, you may skip the 'understand_question' step when the corresponding tool is available. But when unsure, it's good practice to use it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000."""

    def _generate_tools_signature(self) -> str:
        """Generate tool signatures for the system prompt"""
        signatures = []
        for tool_name, tool in self.tools.items():
            signatures.append(f"- {tool_name}: {tool.description}")
        return "\n".join(signatures)
    
    def extract_tool_calls(self, text: str) -> List[ToolCall]:
        """Extract tool calls from LLM response text"""
        tool_calls = []
        
        # Simple pattern matching for tool calls - in practice would be more sophisticated
        for tool_name in self.tools.keys():
            if tool_name in text.lower():
                # Extract parameters if present
                parameters = {"question": "", "context": text}
                tool_calls.append(ToolCall(name=tool_name, parameters=parameters))
                
        return tool_calls
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResponse:
        """Execute a cognitive tool"""
        if tool_call.name not in self.tools:
            return ToolResponse(
                content=f"Error: Unknown tool {tool_call.name}",
                success=False
            )
        
        tool = self.tools[tool_call.name]
        return tool.execute(**tool_call.parameters)
    
    def solve_problem(self, question: str, max_iterations: int = 10) -> str:
        """
        Main problem solving loop using cognitive tools.
        
        This is a simplified version - in practice would integrate with actual LLM APIs
        and handle the full conversation flow with tool calling.
        """
        system_prompt = self.get_system_prompt()
        conversation_history = []
        
        # Initial prompt
        user_prompt = f"Solve the math problem: '{question}'"
        conversation_history.append({"role": "system", "content": system_prompt})
        conversation_history.append({"role": "user", "content": user_prompt})
        
        for iteration in range(max_iterations):
            if self.llm_interface:
                # Get LLM response
                response = self.llm_interface.generate(conversation_history)
                conversation_history.append({"role": "assistant", "content": response})
                
                # Check for final answer
                if "ANSWER:" in response:
                    return response
                
                # Extract and execute tool calls
                tool_calls = self.extract_tool_calls(response)
                
                for tool_call in tool_calls:
                    tool_response = self.execute_tool(tool_call)
                    
                    # Add tool response to conversation
                    tool_message = f"Tool '{tool_call.name}' executed:\n{tool_response.content}"
                    conversation_history.append({"role": "system", "content": tool_message})
            else:
                # For demonstration without actual LLM
                break
                
        return "Problem solving completed (no LLM interface provided)"


# Utility functions for evaluation and benchmarking

def parse_answer(response: str) -> Optional[str]:
    """Extract final answer from LLM response"""
    match = re.search(r'ANSWER:\s*(.+)', response, re.IGNORECASE)
    return match.group(1).strip() if match else None


def evaluate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Calculate pass@1 accuracy"""
    correct = sum(1 for pred, gt in zip(predictions, ground_truth) 
                  if pred.strip() == gt.strip())
    return correct / len(predictions) if predictions else 0.0


class CognitiveToolsConfig:
    """Configuration for cognitive tools framework"""
    
    def __init__(self):
        self.max_iterations = 10
        self.temperature = 0.7
        self.enable_tools = ["understand_question", "recall_related", "examine_answer", "backtracking"]
        self.require_final_answer = True


# Example usage and demonstration

def demo_cognitive_tools():
    """Demonstrate cognitive tools on a sample problem"""
    orchestrator = CognitiveToolsOrchestrator()
    
    # Sample problem
    question = "Find the greatest common divisor of 3339, 2961, and 1491."
    
    print("=== Cognitive Tools Demo ===")
    print(f"Problem: {question}\n")
    
    # Show system prompt
    print("System Prompt:")
    print(orchestrator.get_system_prompt())
    print("\n" + "="*50 + "\n")
    
    # Demonstrate individual tools
    for tool_name, tool in orchestrator.tools.items():
        print(f"Tool: {tool.name}")
        print(f"Description: {tool.description}")
        
        response = tool.execute(question=question)
        print(f"Response: {response.content[:200]}...")
        print("\n" + "-"*30 + "\n")


if __name__ == "__main__":
    demo_cognitive_tools()
