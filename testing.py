"""
Testing and Evaluation Module.

This module provides functions for evaluating pipeline consistency
and comparing results across multiple runs.
"""

import re
import json
from collections import defaultdict
from statistics import mean
from typing import List, Dict, Tuple


def extract_rules_string(rules_field) -> str:
    """
    Convert rules into a single string regardless of format.
    
    Args:
        rules_field: Rules as string, list, or other format
        
    Returns:
        Single string representation of rules
    """
    if isinstance(rules_field, str):
        return rules_field
    elif isinstance(rules_field, list):
        return "\n".join(rules_field)
    else:
        raise ValueError("Unexpected rules format")


def parse_rules_from_string(rules_str: str) -> List[Dict]:
    """
    Parse ASP-like rules into normalized dictionaries.
    
    Args:
        rules_str: String containing Clingo-style rules
        
    Returns:
        List of rule dictionaries with keys: id, modality, head, conditions
    """
    rules = {}
    conditions = defaultdict(list)

    # Standardized pattern: rule(ID, mod, head)
    rule_pat = r"rule\(\s*([A-Za-z0-9_]+)\s*,\s*([a-zA-Z_]+)\s*,\s*([^)]+)\)"
    cond_pat = r"condition\(\s*([A-Za-z0-9_]+)\s*,\s*([^)]+)\)"

    # Extract rules
    for rid, modality, head in re.findall(rule_pat, rules_str):
        rid = rid.lower()
        rules[rid] = {
            "id": rid,
            "modality": modality.lower(),
            "head": re.sub(r"\s+", "", head.lower()),
            "conditions": []
        }

    # Extract conditions
    for rid, cond in re.findall(cond_pat, rules_str):
        rid = rid.lower()
        cond = re.sub(r"\s+", "", cond.lower())
        conditions[rid].append(cond)

    # Attach conditions to rule entries
    for rid in rules:
        unique_conds = sorted(set(conditions[rid]))
        rules[rid]["conditions"] = unique_conds

    return list(rules.values())


def rule_similarity(a: Dict, b: Dict) -> float:
    """
    Compute similarity between two rules (exact structural match).
    
    Args:
        a: First rule dictionary
        b: Second rule dictionary
        
    Returns:
        1.0 if rules match exactly, 0.0 otherwise
    """
    if a["modality"] != b["modality"]:
        return 0.0
    if a["head"] != b["head"]:
        return 0.0
    if a["conditions"] != b["conditions"]:
        return 0.0
    return 1.0


def run_similarity_generic(rulesA_field, rulesB_field) -> float:
    """
    Compute similarity score between two sets of rules.
    
    Args:
        rulesA_field: First set of rules (string or list)
        rulesB_field: Second set of rules (string or list)
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Convert input field to a single string
    A_str = extract_rules_string(rulesA_field)
    B_str = extract_rules_string(rulesB_field)

    # Parse both formats into structured rules
    A = parse_rules_from_string(A_str)
    B = parse_rules_from_string(B_str)

    matched = 0
    used = set()

    # Greedy best matching
    for ra in A:
        for j, rb in enumerate(B):
            if j in used:
                continue
            if rule_similarity(ra, rb):
                matched += 1
                used.add(j)
                break

    denom = max(len(A), len(B))
    return matched / denom if denom > 0 else 0.0


def canonical_head_name(head: str) -> str:
    """
    Extract just the predicate name from a head.
    
    Examples:
        'enter(civilian,zone)' → 'enter'
        'edit_network_settings' → 'edit_network_settings'
        'attempt_edit_network(X,Y)' → 'attempt_edit_network'
    
    Args:
        head: Head predicate string
        
    Returns:
        Canonical predicate name
    """
    head = head.strip()
    # match functor(args)
    m = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", head)
    if m:
        return m.group(1).lower()
    return head.lower()


def compute_statistics_universal(filename: str) -> Tuple[List[Dict], float, float]:
    """
    Compute statistics from test results file.
    
    Computes:
    - Rule Consistency Score (RCS): Average similarity across 3 runs
    - Modality Stability: Fraction of predicates with consistent modality across runs
    
    Args:
        filename: Path to JSON file with test results
        
    Returns:
        Tuple of (example_stats, avg_RCS, avg_modality_stability)
    """
    with open(filename, "r") as f:
        data = json.load(f)

    example_stats = []

    for ex in data:
        if len(ex["result"]) < 3:
            print(f"Warning: Example {ex['id']} has fewer than 3 runs, skipping")
            continue
            
        run1 = ex["result"][0]["rules"]
        run2 = ex["result"][1]["rules"]
        run3 = ex["result"][2]["rules"]

        # Rule Consistency Score
        sim12 = run_similarity_generic(run1, run2)
        sim13 = run_similarity_generic(run1, run3)
        sim23 = run_similarity_generic(run2, run3)
        RCS = (sim12 + sim13 + sim23) / 3

        # Modality Stability
        modality_map = defaultdict(list)

        # For each of the 3 runs
        for run in ex["result"]:
            rules_field = run["rules"]  # may be string or list
            rules_str = extract_rules_string(rules_field)
            parsed = parse_rules_from_string(rules_str)

            for r in parsed:
                key = canonical_head_name(r["head"])
                modality_map[key].append(r["modality"])

        # Compute stability: all 3 modalities same → stable
        stable = sum(1 for mods in modality_map.values() if len(set(mods)) == 1)
        modality_stability = stable / len(modality_map) if modality_map else 0.0

        example_stats.append({
            "id": ex["id"],
            "rule_consistency": RCS,
            "modality_stability": modality_stability
        })

    if not example_stats:
        return [], 0.0, 0.0

    avg_RCS = mean(ex["rule_consistency"] for ex in example_stats)
    avg_mod = (
        mean(e["modality_stability"] for e in example_stats if e["modality_stability"] is not None)
        if any(e["modality_stability"] is not None for e in example_stats)
        else None
    )

    return example_stats, avg_RCS, avg_mod


def print_test_results(
    pipeline_file: str = "pipeline_clingo_test_results.json",
    zero_shot_file: str = "clingo_zero_shot_test_results.json"
):
    """
    Print formatted test results for both pipeline and zero-shot methods.
    
    Args:
        pipeline_file: Path to pipeline test results
        zero_shot_file: Path to zero-shot test results
    """
    print("\n" + "=" * 80)
    print("PIPELINE TEST RESULTS")
    print("=" * 80)
    
    try:
        stats, avg_rcs, avg_mod = compute_statistics_universal(pipeline_file)
        for stat in stats:
            print(f"Example {stat['id']}: RCS={stat['rule_consistency']:.3f}, "
                  f"Modality Stability={stat['modality_stability']:.3f}")
        print(f"\nAverage RCS: {avg_rcs:.3f}")
        print(f"Average Modality Stability: {avg_mod:.3f if avg_mod else 'N/A'}")
    except FileNotFoundError:
        print(f"File not found: {pipeline_file}")
    except Exception as e:
        print(f"Error processing {pipeline_file}: {e}")

    print("\n" + "=" * 80)
    print("ZERO-SHOT TEST RESULTS")
    print("=" * 80)
    
    try:
        stats, avg_rcs, avg_mod = compute_statistics_universal(zero_shot_file)
        for stat in stats:
            print(f"Example {stat['id']}: RCS={stat['rule_consistency']:.3f}, "
                  f"Modality Stability={stat['modality_stability']:.3f}")
        print(f"\nAverage RCS: {avg_rcs:.3f}")
        print(f"Average Modality Stability: {avg_mod:.3f if avg_mod else 'N/A'}")
    except FileNotFoundError:
        print(f"File not found: {zero_shot_file}")
    except Exception as e:
        print(f"Error processing {zero_shot_file}: {e}")


if __name__ == "__main__":
    # Run evaluation on default test result files
    print_test_results()

