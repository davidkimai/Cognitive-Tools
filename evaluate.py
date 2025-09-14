"""
Evaluation Framework for Cognitive Tools

Comprehensive evaluation system matching the paper's methodology for assessing
cognitive tools performance across mathematical reasoning benchmarks.

JUPYTER/COLAB USAGE:
    from evaluate import CognitiveToolsEvaluator, run_statistical_analysis, compare_configurations
    evaluator = CognitiveToolsEvaluator(llm_interfaces)
    result = await evaluator.evaluate_configuration("GPT-4", "AIME2024", "cognitive_tools")
    comparison = await compare_configurations("baseline", "cognitive_tools", "GPT-4", "MATH500")
    
Supports both statistical significance testing and detailed error analysis.
"""

import json
import re
import asyncio
import statistics
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from cognitive_tools import CognitiveToolsOrchestrator, parse_answer

# Handle scipy import gracefully for Colab compatibility
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available. Some statistical functions will use simplified implementations.")


@dataclass
class EvaluationResult:
    """
    Single evaluation result with comprehensive metadata
    
    JUPYTER USAGE:
        result = EvaluationResult("GPT-4", "AIME2024", "cognitive_tools", 43.3, 1.2, 30, 8)
        print(f"Accuracy: {result.accuracy:.1f}% ± {result.std_error:.1f}")
    """
    model_name: str
    benchmark: str
    configuration: str
    accuracy: float
    std_error: float
    num_samples: int
    num_runs: int
    raw_scores: List[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class ComparisonResult:
    """
    Statistical comparison between two configurations
    
    JUPYTER USAGE:
        comparison = ComparisonResult(26.7, 43.3, 16.6, 0.001, True, 1.8, (12.4, 20.8))
        print(f"Improvement: {comparison.improvement:.1f}% (p={comparison.p_value:.3f})")
    """
    baseline_accuracy: float
    treatment_accuracy: float
    improvement: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]


# LLM-as-a-Judge Implementation
class LLMJudge:
    """
    LLM-as-a-judge for evaluating mathematical expressions.
    Based on the paper's evaluation methodology for MATH500 benchmark.
    
    JUPYTER USAGE:
        judge = LLMJudge(your_llm_interface)
        is_correct = await judge.evaluate_pair("x = 2, 3", "x = 2, x = 3")
        print(f"Equivalent: {is_correct}")
    """
    
    def __init__(self, judge_llm_interface=None):
        self.judge_llm_interface = judge_llm_interface
        
    def get_judge_prompt(self, prediction: str, ground_truth: str) -> str:
        """Generate the exact judge prompt used in the paper"""
        return f"""The following two expressions are answers to a math problem. They can be given as direct numerical answers or as a full reasoning. You have to judge whether they are equivalent.

Only perform trivial simplifications, but accept numerical answers which are correct within a reasonable numerical tolerance.

Examples:
Expression 1: 2x + 3
Expression 2: 3 + 2x
Yes

Expression 1: 3/2
Expression 2: 1.5
Yes

Expression 1: x^2 + 2x + 1
Expression 2: y^2 + 2y + 1
No

Expression 1: x^2 + 2x + 1
Expression 2: (x + 1)^2
Yes

Expression 1: 3245/5
Expression 2: 649
Yes
(trivial simplifications are allowed)

Expression 1: 2/(-3)
Expression 2: -2/3
Yes
(trivial simplifications are allowed)

Expression 1: 72 degrees
Expression 2: 72
Yes
(give benefit of the doubt to units)

Expression 1: 64
Expression 2: 64 square feet
Yes
(give benefit of the doubt to units)

---
YOUR TASK
Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

Expression 1: {prediction}
Expression 2: {ground_truth}"""
    
    async def evaluate_pair(self, prediction: str, ground_truth: str) -> bool:
        """Evaluate if prediction matches ground truth using LLM judge"""
        if not self.judge_llm_interface:
            # Fallback to exact string matching
            return prediction.strip().lower() == ground_truth.strip().lower()
        
        prompt = self.get_judge_prompt(prediction, ground_truth)
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.judge_llm_interface.generate(messages)
        return response.strip().lower() == "yes"


# Benchmark Data Handling
class BenchmarkHandler:
    """
    Manages benchmark datasets and their specific evaluation requirements
    
    JUPYTER USAGE:
        handler = BenchmarkHandler()
        problems = handler.load_benchmark("AIME2024")
        method = handler.get_evaluation_method("MATH500")  # Returns "llm_judge"
    """
    
    def __init__(self, benchmarks_file: str = "benchmarks.json"):
        self.benchmarks_file = benchmarks_file
        self.benchmark_configs = {
            "AIME2024": {
                "evaluation_method": "exact_match",
                "answer_type": "numerical",
                "difficulty": "high",
                "num_samples": 30
            },
            "MATH500": {
                "evaluation_method": "llm_judge",
                "answer_type": "expression",
                "difficulty": "varied",
                "num_samples": 500
            },
            "AMC": {
                "evaluation_method": "exact_match", 
                "answer_type": "numerical",
                "difficulty": "medium",
                "num_samples": 83
            },
            "Smolbenchmark": {
                "evaluation_method": "exact_match",
                "answer_type": "varied",
                "difficulty": "medium", 
                "num_samples": 50
            }
        }
    
    def load_benchmark(self, benchmark_name: str) -> List[Dict]:
        """Load specific benchmark dataset"""
        try:
            with open(self.benchmarks_file, 'r') as f:
                all_benchmarks = json.load(f)
            return all_benchmarks.get(benchmark_name, [])
        except FileNotFoundError:
            return self._generate_sample_data(benchmark_name)
    
    def _generate_sample_data(self, benchmark_name: str) -> List[Dict]:
        """Generate sample data for demonstration when benchmark file unavailable"""
        sample_problems = {
            "AIME2024": [
                {"question": "Find the number of ordered pairs (a,b) of integers such that |a + bi| = √(a² + b²) = 13", "answer": "12"},
                {"question": "If f(x) = x³ - 3x + 1, find the sum of all real roots", "answer": "0"}
            ],
            "MATH500": [
                {"question": "Solve x² - 5x + 6 = 0", "answer": "x = 2, 3"},
                {"question": "Find the derivative of sin(x²)", "answer": "2x cos(x²)"}
            ],
            "AMC": [
                {"question": "What is 15% of 80?", "answer": "12"},
                {"question": "How many prime numbers are less than 20?", "answer": "8"}
            ],
            "Smolbenchmark": [
                {"question": "Find the GCD of 48 and 18", "answer": "6"},
                {"question": "Evaluate 3! + 4!", "answer": "30"}
            ]
        }
        return sample_problems.get(benchmark_name, [])
    
    def get_evaluation_method(self, benchmark_name: str) -> str:
        """Get the appropriate evaluation method for a benchmark"""
        return self.benchmark_configs.get(benchmark_name, {}).get("evaluation_method", "exact_match")


# Statistical Analysis Tools
class StatisticalAnalyzer:
    """
    Statistical analysis tools for evaluation results
    
    JUPYTER USAGE:
        analyzer = StatisticalAnalyzer()
        accuracy = analyzer.calculate_pass_at_k(predictions, ground_truths)
        ci = analyzer.bootstrap_confidence_interval([0.6, 0.7, 0.65, 0.72])
        comparison = analyzer.paired_t_test([0.6, 0.65], [0.7, 0.75])
    """
    
    @staticmethod
    def calculate_pass_at_k(predictions: List[str], ground_truths: List[str], k: int = 1) -> float:
        """
        Calculate pass@k accuracy (paper uses pass@1)
        
        JUPYTER USAGE:
            preds = ["42", "7", "15"]  
            truth = ["42", "8", "15"]
            accuracy = StatisticalAnalyzer.calculate_pass_at_k(preds, truth)
            print(f"Pass@1 accuracy: {accuracy:.1%}")
        """
        assert len(predictions) == len(ground_truths), "Predictions and ground truths must have same length"
        
        correct = sum(1 for pred, gt in zip(predictions, ground_truths) 
                     if pred.strip() == gt.strip())
        return correct / len(predictions) if predictions else 0.0
    
    @staticmethod
    def bootstrap_confidence_interval(scores: List[float], confidence_level: float = 0.95, num_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals for accuracy estimates
        
        JUPYTER USAGE:
            scores = [0.6, 0.65, 0.7, 0.68, 0.72]
            ci_lower, ci_upper = StatisticalAnalyzer.bootstrap_confidence_interval(scores)
            print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        """
        if len(scores) <= 1:
            return (0.0, 0.0)
        
        bootstrap_means = []
        for _ in range(num_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    @staticmethod
    def paired_t_test(baseline_scores: List[float], treatment_scores: List[float]) -> ComparisonResult:
        """
        Perform paired t-test for statistical significance
        
        JUPYTER USAGE:
            baseline = [0.60, 0.65, 0.62, 0.68]
            treatment = [0.70, 0.75, 0.72, 0.78] 
            result = StatisticalAnalyzer.paired_t_test(baseline, treatment)
            print(f"p-value: {result.p_value:.4f}, significant: {result.significant}")
        """
        assert len(baseline_scores) == len(treatment_scores), "Score lists must have same length"
        
        baseline_mean = np.mean(baseline_scores)
        treatment_mean = np.mean(treatment_scores)
        improvement = treatment_mean - baseline_mean
        
        # Paired t-test
        if SCIPY_AVAILABLE:
            statistic, p_value = stats.ttest_rel(treatment_scores, baseline_scores)
        else:
            # Simplified t-test implementation
            differences = np.array(treatment_scores) - np.array(baseline_scores)
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            n = len(differences)
            t_stat = mean_diff / (std_diff / np.sqrt(n))
            
            # Approximate p-value calculation
            if abs(t_stat) > 2.0:  # Rough approximation
                p_value = 0.05 if abs(t_stat) > 2.5 else 0.1
            else:
                p_value = 0.2
        
        # Effect size (Cohen's d)
        differences = np.array(treatment_scores) - np.array(baseline_scores)
        effect_size = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0.0
        
        # Confidence interval for the difference
        n = len(differences)
        if SCIPY_AVAILABLE:
            t_critical = stats.t.ppf(0.975, n - 1)
        else:
            t_critical = 2.0  # Approximation
        margin_error = t_critical * np.std(differences) / np.sqrt(n)
        ci_lower = improvement - margin_error
        ci_upper = improvement + margin_error
        
        return ComparisonResult(
            baseline_accuracy=baseline_mean,
            treatment_accuracy=treatment_mean,
            improvement=improvement,
            p_value=p_value,
            significant=p_value < 0.05,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper)
        )


# Main Evaluation Engine
class CognitiveToolsEvaluator:
    """
    Main evaluation engine that orchestrates the complete evaluation pipeline
    used in the paper's experiments.
    
    JUPYTER USAGE:
        evaluator = CognitiveToolsEvaluator(llm_interfaces, judge_llm_interface)
        result = await evaluator.evaluate_configuration("GPT-4", "AIME2024", "cognitive_tools")
        comparison = await evaluator.compare_configurations("GPT-4", "MATH500", "baseline", "cognitive_tools")
    """
    
    def __init__(self, llm_interfaces: Dict[str, Any] = None, judge_llm_interface=None):
        self.llm_interfaces = llm_interfaces or {}
        self.benchmark_handler = BenchmarkHandler()
        self.judge = LLMJudge(judge_llm_interface)
        self.stats_analyzer = StatisticalAnalyzer()
        self.orchestrator = CognitiveToolsOrchestrator()
    
    async def evaluate_configuration(
        self, 
        model_name: str, 
        benchmark_name: str, 
        configuration: str,
        num_runs: int = 8,
        use_judge: bool = None
    ) -> EvaluationResult:
        """
        Evaluate a specific model configuration on a benchmark.
        
        JUPYTER USAGE:
            result = await evaluator.evaluate_configuration(
                model_name="GPT-4",
                benchmark_name="AIME2024", 
                configuration="cognitive_tools",
                num_runs=5
            )
            print(f"{result.configuration}: {result.accuracy:.1f}% ± {result.std_error:.1f}")
        """
        
        # Load benchmark data
        problems = self.benchmark_handler.load_benchmark(benchmark_name)
        if not problems:
            raise ValueError(f"No problems found for benchmark: {benchmark_name}")
        
        # Determine evaluation method
        if use_judge is None:
            use_judge = self.benchmark_handler.get_evaluation_method(benchmark_name) == "llm_judge"
        
        # Run multiple evaluation iterations
        run_accuracies = []
        
        for run_idx in range(num_runs):
            predictions = []
            ground_truths = []
            
            # Evaluate each problem
            for problem in problems:
                question = problem["question"]
                ground_truth = problem["answer"]
                
                # Generate prediction based on configuration
                if configuration == "baseline":
                    prediction = await self._run_baseline(model_name, question)
                elif configuration == "cognitive_tools":
                    prediction = await self._run_cognitive_tools(model_name, question)
                elif configuration == "cognitive_prompting":
                    prediction = await self._run_cognitive_prompting(model_name, question)
                else:
                    prediction = await self._run_custom_configuration(model_name, question, configuration)
                
                # Extract answer
                extracted_answer = parse_answer(prediction) or prediction.strip()
                
                predictions.append(extracted_answer)
                ground_truths.append(ground_truth)
            
            # Calculate accuracy for this run
            if use_judge:
                correct_count = 0
                for pred, gt in zip(predictions, ground_truths):
                    is_correct = await self.judge.evaluate_pair(pred, gt)
                    correct_count += int(is_correct)
                accuracy = correct_count / len(predictions)
            else:
                accuracy = self.stats_analyzer.calculate_pass_at_k(predictions, ground_truths, k=1)
            
            run_accuracies.append(accuracy * 100)  # Convert to percentage
        
        # Calculate statistics
        mean_accuracy = statistics.mean(run_accuracies)
        std_error = statistics.stdev(run_accuracies) / np.sqrt(len(run_accuracies)) if len(run_accuracies) > 1 else 0.0
        
        return EvaluationResult(
            model_name=model_name,
            benchmark=benchmark_name,
            configuration=configuration,
            accuracy=mean_accuracy,
            std_error=std_error,
            num_samples=len(problems),
            num_runs=num_runs,
            raw_scores=run_accuracies,
            metadata={
                "use_judge": use_judge,
                "evaluation_method": self.benchmark_handler.get_evaluation_method(benchmark_name)
            }
        )
    
    async def _run_baseline(self, model_name: str, question: str) -> str:
        """Run baseline model without tools"""
        if model_name in self.llm_interfaces:
            llm = self.llm_interfaces[model_name]
            messages = [{"role": "user", "content": f"Solve the math problem: '{question}'"}]
            return await llm.generate(messages)
        else:
            return "ANSWER: [simulated baseline response]"
    
    async def _run_cognitive_tools(self, model_name: str, question: str) -> str:
        """Run with cognitive tools framework"""
        if model_name in self.llm_interfaces:
            llm = self.llm_interfaces[model_name]
            orchestrator = CognitiveToolsOrchestrator(llm)
            return await orchestrator.solve_problem(question)
        else:
            return "ANSWER: [simulated cognitive tools response]"
    
    async def _run_cognitive_prompting(self, model_name: str, question: str) -> str:
        """Run with cognitive prompting baseline"""
        cognitive_prompt = """Solve the following math problem by following each step of cognitive operations from the list below. For each step, provide your reasoning and calculations before moving on to the next step.

Cognitive Operations:
1. Goal Clarification: Restate the problem in your own words.
2. Decomposition: List the given information.
3. Filtering: Identify what you need to find.
4. Reorganization: Assign variables to the unknowns.
5. Pattern Recognition: define each variable clearly.
6. Abstraction: Set up equations based on the problem.
7. Generalization: Solve the equations step by step.
8. Integration: Verify your solution with the given information.

Your Response: Please start with "Restate the problem in your own words" and proceed through each cognitive operation step by step, providing detailed reasoning and calculations for each.

Give the final answer using the format: 'ANSWER: answer'."""
        
        if model_name in self.llm_interfaces:
            llm = self.llm_interfaces[model_name]
            messages = [{"role": "user", "content": f"{cognitive_prompt}\n\nProblem: {question}"}]
            return await llm.generate(messages)
        else:
            return "ANSWER: [simulated cognitive prompting response]"
    
    async def _run_custom_configuration(self, model_name: str, question: str, configuration: str) -> str:
        """Run custom configuration (e.g., single tools)"""
        return f"ANSWER: [simulated {configuration} response]"
    
    async def compare_configurations(
        self,
        model_name: str,
        benchmark_name: str, 
        baseline_config: str,
        treatment_config: str,
        num_runs: int = 8
    ) -> ComparisonResult:
        """
        Compare two configurations with statistical significance testing
        
        JUPYTER USAGE:
            comparison = await evaluator.compare_configurations(
                "GPT-4", "AIME2024", "baseline", "cognitive_tools", num_runs=5
            )
            print(f"Improvement: {comparison.improvement:.1f}% (p={comparison.p_value:.3f})")
            print(f"Significant: {comparison.significant}")
        """
        
        # Evaluate both configurations
        baseline_result = await self.evaluate_configuration(
            model_name, benchmark_name, baseline_config, num_runs
        )
        treatment_result = await self.evaluate_configuration(
            model_name, benchmark_name, treatment_config, num_runs
        )
        
        # Perform statistical comparison
        comparison = self.stats_analyzer.paired_t_test(
            baseline_result.raw_scores,
            treatment_result.raw_scores
        )
        
        return comparison


# High-level evaluation functions for easy Jupyter usage

async def evaluate_model_on_benchmark(
    model_name: str, 
    benchmark_name: str, 
    llm_interfaces: Dict[str, Any] = None, 
    configurations: List[str] = None
) -> Dict[str, EvaluationResult]:
    """
    Evaluate multiple configurations of a model on a benchmark
    
    JUPYTER USAGE:
        results = await evaluate_model_on_benchmark(
            "GPT-4", 
            "AIME2024", 
            llm_interfaces, 
            ["baseline", "cognitive_tools"]
        )
        for config, result in results.items():
            print(f"{config}: {result.accuracy:.1f}%")
    """
    
    if configurations is None:
        configurations = ["baseline", "cognitive_tools"]
    
    evaluator = CognitiveToolsEvaluator(llm_interfaces)
    results = {}
    
    for config in configurations:
        print(f"Evaluating {model_name} with {config} on {benchmark_name}...")
        result = await evaluator.evaluate_configuration(model_name, benchmark_name, config, num_runs=3)
        results[config] = result
    
    return results


async def compare_configurations(
    baseline_config: str,
    treatment_config: str, 
    model_name: str,
    benchmark_name: str,
    llm_interfaces: Dict[str, Any] = None
) -> ComparisonResult:
    """
    Compare two configurations with statistical analysis
    
    JUPYTER USAGE:
        comparison = await compare_configurations(
            "baseline", "cognitive_tools", "GPT-4", "AIME2024", llm_interfaces
        )
        print(f"Baseline: {comparison.baseline_accuracy:.1f}%")
        print(f"Treatment: {comparison.treatment_accuracy:.1f}%")
        print(f"Improvement: +{comparison.improvement:.1f}% (p={comparison.p_value:.3f})")
    """
    
    evaluator = CognitiveToolsEvaluator(llm_interfaces)
    comparison = await evaluator.compare_configurations(
        model_name, benchmark_name, baseline_config, treatment_config, num_runs=3
    )
    
    print(f"Comparison Results: {baseline_config} vs {treatment_config}")
    print(f"Baseline: {comparison.baseline_accuracy:.1f}%")
    print(f"Treatment: {comparison.treatment_accuracy:.1f}%")
    print(f"Improvement: +{comparison.improvement:.1f}%")
    print(f"P-value: {comparison.p_value:.4f}")
    print(f"Statistically significant: {comparison.significant}")
    print(f"Effect size (Cohen's d): {comparison.effect_size:.2f}")
    print(f"95% CI for improvement: [{comparison.confidence_interval[0]:.1f}%, {comparison.confidence_interval[1]:.1f}%]")
    
    return comparison


async def run_statistical_analysis(
    results_dict: Dict[str, List[float]], 
    baseline_key: str = "baseline"
) -> Dict[str, ComparisonResult]:
    """
    Run statistical analysis comparing multiple configurations to baseline
    
    JUPYTER USAGE:
        results_data = {
            "baseline": [26.7, 25.8, 27.1, 26.3],
            "cognitive_tools": [43.3, 42.1, 44.2, 43.8],
            "cognitive_prompting": [32.1, 31.5, 32.8, 31.9]
        }
        comparisons = await run_statistical_analysis(results_data, "baseline")
        for config, comp in comparisons.items():
            print(f"{config}: +{comp.improvement:.1f}% (p={comp.p_value:.3f})")
    """
    
    analyzer = StatisticalAnalyzer()
    baseline_scores = results_dict.get(baseline_key, [])
    
    if not baseline_scores:
        raise ValueError(f"Baseline key '{baseline_key}' not found in results")
    
    comparisons = {}
    
    for config_name, scores in results_dict.items():
        if config_name != baseline_key:
            comparison = analyzer.paired_t_test(baseline_scores, scores)
            comparisons[config_name] = comparison
            
            print(f"\n{config_name} vs {baseline_key}:")
            print(f"  Improvement: +{comparison.improvement:.1f}%")
            print(f"  P-value: {comparison.p_value:.4f}")
            print(f"  Significant: {comparison.significant}")
            print(f"  Effect size: {comparison.effect_size:.2f}")
    
    return comparisons


def create_sample_evaluation_data() -> Dict[str, List[float]]:
    """
    Create sample evaluation data for demonstration
    
    JUPYTER USAGE:
        sample_data = create_sample_evaluation_data()
        comparisons = await run_statistical_analysis(sample_data)
    """
    return {
        "baseline": [26.7, 25.8, 27.1, 26.3, 27.5, 25.9, 26.8, 26.1],
        "cognitive_tools": [43.3, 42.1, 44.2, 43.8, 42.7, 43.9, 43.1, 44.0], 
        "cognitive_prompting": [32.1, 31.5, 32.8, 31.9, 32.3, 31.7, 32.4, 31.8],
        "understand_question_only": [29.2, 28.8, 29.7, 29.1, 29.4, 28.9, 29.3, 29.0],
        "examine_answer_only": [30.1, 29.7, 30.5, 30.2, 29.8, 30.3, 29.9, 30.4]
    }


def show_evaluation_summary(results: List[EvaluationResult]):
    """
    Display formatted summary of evaluation results
    
    JUPYTER USAGE:
        results = [result1, result2, result3]
        show_evaluation_summary(results)
    """
    
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 50)
    
    for result in results:
        print(f"\nModel: {result.model_name}")
        print(f"Benchmark: {result.benchmark}")
        print(f"Configuration: {result.configuration}")
        print(f"Accuracy: {result.accuracy:.1f}% ± {result.std_error:.1f}")
        print(f"Samples: {result.num_samples}, Runs: {result.num_runs}")
        
        if result.metadata and result.metadata.get("use_judge"):
            print("Evaluation Method: LLM-as-a-judge")
        else:
            print("Evaluation Method: Exact match")


async def run_demo_evaluation():
    """
    Run demonstration evaluation with sample data
    
    JUPYTER USAGE:
        await run_demo_evaluation()
    """
    
    print("COGNITIVE TOOLS EVALUATION FRAMEWORK DEMO")
    print("=" * 50)
    print()
    
    # Create sample evaluation data
    sample_data = create_sample_evaluation_data()
    
    print("Sample Evaluation Data:")
    for config, scores in sample_data.items():
        mean_score = statistics.mean(scores)
        print(f"  {config}: {mean_score:.1f}% (n={len(scores)})")
    
    print("\nRunning Statistical Analysis...")
    comparisons = await run_statistical_analysis(sample_data)
    
    print(f"\nSUMMARY:")
    print("All cognitive approaches show significant improvements over baseline")
    print(f"Best performing: cognitive_tools (+{comparisons['cognitive_tools'].improvement:.1f}%)")
    
    # Bootstrap confidence intervals
    analyzer = StatisticalAnalyzer()
    baseline_scores = sample_data["baseline"]
    tools_scores = sample_data["cognitive_tools"]
    
    baseline_ci = analyzer.bootstrap_confidence_interval(baseline_scores)
    tools_ci = analyzer.bootstrap_confidence_interval(tools_scores)
    
    print(f"\n95% Confidence Intervals:")
    print(f"  Baseline: [{baseline_ci[0]:.1f}%, {baseline_ci[1]:.1f}%]")
    print(f"  Cognitive Tools: [{tools_ci[0]:.1f}%, {tools_ci[1]:.1f}%]")


# Make key components easily accessible for imports
__all__ = [
    'CognitiveToolsEvaluator',
    'EvaluationResult', 
    'ComparisonResult',
    'LLMJudge',
    'BenchmarkHandler',
    'StatisticalAnalyzer',
    'evaluate_model_on_benchmark',
    'compare_configurations',
    'run_statistical_analysis',
    'create_sample_evaluation_data',
    'show_evaluation_summary',
    'run_demo_evaluation'
]
