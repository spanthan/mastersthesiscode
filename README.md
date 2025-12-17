# DDL Legal Text Analysis Pipeline

A comprehensive, reproducible research pipeline for converting legal text into Defeasible Deontic Logic (DDL) and Answer Set Programming (ASP) representations.

## Overview

This pipeline processes legal text through five main phases:
1. **Atomic Phrase Extraction**: Extract subject-verb-object triples using spaCy
2. **Symbol Matching**: Match extracted atoms to existing symbols using embeddings
3. **Rule Segmentation**: Split legal text into distinct normative statements
4. **Formalization**: Convert rules into DDL format
5. **ASP Conversion**: Generate Clingo ASP programs for reasoning

## File Structure

- `main.py`: Main pipeline orchestrator with DDLChain class
- `llm_interface.py`: OpenAI API interface for LLM operations
- `clingo_interface.py`: Clingo ASP conversion and solving
- `symbol_table.py`: Symbol table management for consistent terminology
- `testing.py`: Evaluation and consistency testing functions

## Installation

```bash
# Install dependencies
pip install openai spacy clingo numpy

# Download spaCy model
python -m spacy download en_core_web_sm

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Pipeline Execution

```bash
# Run full pipeline on all examples
python main.py --input testing_texts.json --verbose

# Run with custom output directories
python main.py --input testing_texts.json --output-dir results --clingo-dir programs
```

### Consistency Testing

```bash
# Test consistency with 3 runs per example
python main.py --test-consistency --ids 1 2 3 4 5
```

### Evaluation

```bash
# Evaluate test results
python testing.py
```

## Configuration

### Input Format

The input JSON file should have the following structure:

```json
[
  {
    "id": 1,
    "text": "Legal text here...",
    "scenario": "Test scenario description..."
  }
]
```

### Output Structure

The pipeline generates:
- `testing_results/test_output_{id}.json`: Full pipeline results
- `clingo_programs/clingo_program_{id}.txt`: Clingo ASP programs
- `pipeline_clingo_test_results.json`: Consistency test results

## API Reference

### DDLChain Class

Main pipeline class that orchestrates the full analysis.

```python
from main import DDLChain

chain = DDLChain()
result = chain.ddl_chain_full_pipeline(
    text="Legal text...",
    verbose=True
)
```

### LLMInterface Class

Handles all LLM operations including rule rewriting, segmentation, and formalization.

```python
from llm_interface import LLMInterface

llm = LLMInterface(api_key="your-key")
segmentation = llm.extract_rule_segments(text)
```

### SymbolTable Class

Manages canonical symbol mappings for consistent terminology.

```python
from symbol_table import SymbolTable

table = SymbolTable()
symbols = table.extract_symbols_from_rules(formal_rules)
```

## Evaluation Metrics

The testing module computes:

- **Rule Consistency Score (RCS)**: Average similarity of rules across multiple runs (0-1)
- **Modality Stability**: Fraction of predicates with consistent modality across runs (0-1)

## Reproducibility

To ensure reproducibility:

1. Set random seeds where applicable
2. Use consistent API keys and model versions
3. Save all intermediate results
4. Document configuration parameters

## Citation

If you use this pipeline in your research, please cite appropriately.

