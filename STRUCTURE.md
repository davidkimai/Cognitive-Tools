# Cognitive Tools for LLM Reasoning

```
Cognitive-Tools/
├── README.md                    # Complete guide: theory, usage, results
├── cognitive_tools.py           # Core implementation: all 4 tools + orchestrator
├── reproduce_paper.py           # Exact reproduction of paper experiments  
├── demo.py                     # Interactive examples and basic usage
├── evaluate.py                 # Evaluation framework and metrics
├── requirements.txt            # Dependencies only
└── benchmarks.json            # All benchmark data in single file
```

## Design Rationale

### Maximum Information Density
- **cognitive_tools.py**: Single file containing all tool implementations, prompts, and orchestration logic (~300-400 lines)
- **README.md**: Complete documentation serving both tutorial and reference needs
- **benchmarks.json**: All evaluation data consolidated instead of scattered across subdirectories

### Immediate Accessibility  
- Zero navigation complexity - everything visible at root level
- Core functionality in single importable module
- Working examples in demo.py executable immediately after clone

### Research Reproducibility
- **reproduce_paper.py**: Exact replication of paper methodology and results
- **evaluate.py**: Statistical analysis and benchmark evaluation matching paper
- All prompts and hyperparameters embedded in implementation, not separate config files

### First Principles Approach
- Eliminates artificial separation between "source" and "docs" and "data"
- Code serves as primary documentation through clear implementation
- Single source of truth for each concept instead of multiple redundant files

This structure maximizes signal-to-noise ratio while serving both novice learners (clear examples, comprehensive README) and experienced researchers (direct access to implementation, exact reproduction scripts).
