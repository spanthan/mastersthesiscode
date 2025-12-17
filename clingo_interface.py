"""
Clingo ASP (Answer Set Programming) Interface.

This module handles conversion of formal DDL rules to Clingo ASP programs
and execution of Clingo solvers for reasoning.
"""

import re
import json
import clingo
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from openai import OpenAI
import os


############################################################
# Utility: ASP-safe naming
############################################################

def asp_name(x: str) -> str:
    """Convert any phrase to lowercase snake_case for ASP."""
    x = x.strip()
    x = re.sub(r'[^a-zA-Z0-9]+', ' ', x.lower())
    return "_".join([p for p in x.split() if p])


############################################################
# Flatten nested actors: A(B(C)) → a_b_c
############################################################

def flatten_actor(expr: str) -> str:
    """Flatten nested actor expressions into single identifier."""
    expr = expr.strip()
    m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\((.*)\)", expr)
    if m:
        outer = asp_name(m.group(1))
        inner = flatten_actor(m.group(2))
        return f"{outer}_{inner}"
    return asp_name(expr)


############################################################
# Parse head predicate: extract action + actor
############################################################
def parse_head_predicate(pred: str):
    """
    Handles 2-argument and multi-argument predicates.

    Cases:
    1) Action(Actor, Object)
       → action = action_object
         actor = actor

    2) Action(A, B, C)
       → A becomes a condition
         head becomes action_b
         C becomes a condition

    Returns:
        action_atom, actor_atom, extra_conditions
    """
    pred = pred.strip()
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", pred)
    if not m:
        return asp_name(pred), None, []

    functor = asp_name(m.group(1))
    args = [a.strip() for a in m.group(2).split(",")]

    # -----------------------------
    # Case 1: Only one argument — treat as simple action
    # -----------------------------
    if len(args) == 1:
        return functor, None, []

    # -----------------------------
    # Case 2: Standard 2-argument case
    # -----------------------------
    if len(args) == 2:
        actor_raw, obj_raw = args
        actor_atom = flatten_actor(actor_raw)
        object_atom = flatten_actor(obj_raw)
        action_atom = f"{functor}_{object_atom}"
        return action_atom, actor_atom, []

    # -----------------------------
    # Case 3: 3+ arguments
    # -----------------------------
    first = flatten_actor(args[0])            # becomes condition
    second = flatten_actor(args[1])           # becomes part of action
    others = [flatten_actor(a) for a in args[2:]]  # become conditions

    action_atom = f"{functor}_{second}"

    return action_atom, None, [first] + others


############################################################
# Parse antecedent → list of conditions
############################################################

def parse_antecedent(ant_raw: str) -> List[str]:
    """
    Parse antecedent string into list of condition atoms.
    
    Args:
        ant_raw: Antecedent string (may contain "⊤", "AND", "∧", etc.)
        
    Returns:
        List of condition atoms
    """
    ant_raw = ant_raw.strip()

    if ant_raw == "⊤":
        return []

    ant_raw = ant_raw.replace("AND", "∧").replace("&&", "∧")

    parts = [p.strip() for p in ant_raw.split("∧") if p.strip()]
    return [flatten_actor(p) for p in parts]


############################################################
# Convert a single formal rule to ASP
############################################################

def parse_rule_tuple(idx: int, rule_tuple: Tuple[str, str, List[str]]) -> str:
    (rule_text, nl_text, priority_over) = rule_tuple
    rule_id = f"r{idx}"

    # Split antecedent and head
    ant_raw, rest = rule_text.split("->")
    ant_raw = ant_raw.strip()

    # Modality
    modality = re.search(r"\[(.*?)\]", rest).group(1).lower()

    # Head predicate
    head_pred = rest.split("]")[-1].strip()
    action_atom, actor_atom, extra_conds = parse_head_predicate(head_pred)

    # Antecedents
    antecedents = parse_antecedent(ant_raw)

    asp = []
    asp.append(f"rule({rule_id}, {modality}, {action_atom}).")

    # Add antecedent conditions
    for cond in antecedents:
        asp.append(f"condition({rule_id}, {cond}).")

    # Add extra conditions from multi-arg predicate
    for cond in extra_conds:
        asp.append(f"condition({rule_id}, {cond}).")

    # Add actor (only in 2-arg case)
    if actor_atom:
        asp.append(f"condition({rule_id}, {actor_atom}).")

    # Priorities
    for higher in priority_over:
        asp.append(f"overrides({rule_id}, {higher}).")

    return "\n".join(asp)


############################################################
# Convert all rules to full ASP program
############################################################

def convert_all_rules(rule_tuples: List[Tuple[str, str, List[str]]]) -> str:
    """
    Convert all rule tuples to a complete Clingo ASP program.
    
    Args:
        rule_tuples: List of (formal_text, nl_text, priority_list) tuples
        
    Returns:
        Complete Clingo ASP program string
    """
    parts = []

    for idx, tup in enumerate(rule_tuples, start=1):
        parts.append(parse_rule_tuple(idx, tup))

    # Optional: Add DDL meta-theory (commented out by default)
    # Uncomment if you want to include the DDL reasoning rules
    ddl_meta = r"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% META-THEORY (DDL)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

applicable(R) :-
    rule(R,_,_),
    not not_applicable(R).

not_applicable(R) :-
    condition(R, C),
    not holds(C).

holds(true).

defeated(R1) :-
    overrides(R2, R1),
    applicable(R2).

forbidden(A) :-
    applicable(R),
    rule(R, forbidden, A),
    not defeated(R).

permission(A) :-
    applicable(R),
    rule(R, permission, A),
    not defeated(R).

obligation(A) :-
    applicable(R),
    rule(R, obligation, A),
    not defeated(R).

#show forbidden/1.
#show permission/1.
#show obligation/1.
"""
    # Uncomment to include meta-theory:
    # parts.append(ddl_meta)
    
    return "\n".join(parts)


############################################################
# Load rules from JSON file
############################################################

def get_rules(path: str) -> List[Tuple[str, str, List[str]]]:
    """
    Load rules from pipeline output JSON file.
    
    Args:
        path: Path to JSON file with pipeline results
        
    Returns:
        List of (formal_rule, nl_text, priority) tuples
    """
    with open(path) as f:
        content = json.load(f)
        rules = []
        for i, r in enumerate(content["segmentation"]["rules"]):
            antecedent = content["formal_rules"][i]["antecedent"]
            modality = content["formal_rules"][i]["modality"]
            head = content["formal_rules"][i]["head"]
            full_rule = f"{antecedent} -> [{modality.upper()}] {head}"
            nl_text = r["natural_language_rule"]
            priority = content["priorities"][i][1]
            rules.append((full_rule, nl_text, priority))
    return rules


############################################################
# Solve with Clingo
############################################################

def solve(program: str, facts: str):
    """
    Solve a Clingo program with given facts.
    
    Args:
        program: Clingo program string
        facts: Additional facts to add
    """
    ctl = clingo.Control()
    ctl.add("base", [], program)
    ctl.add("base", [], facts)
    ctl.ground([("base", [])])
    print("=== Models ===")
    ctl.solve(on_model=lambda m: print(m))


def get_constraints(original: str, scenario_text: str, rules: str) -> str:
    """
    Use LLM to determine which conditions are true/false in a scenario.
    
    Args:
        original: Original legal text
        scenario_text: Test scenario description
        rules: Clingo rules string
        
    Returns:
        String with condition assignments (format: "condition = True/False")
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    prompt = f'''You are a reasoning assistant. Your task is to determine which conditions are TRUE or FALSE
    in a test scenario, based on:

    1. The natural-language scenario.
    2. The Clingo-style rule encoding provided.

    Rules describe permissions and prohibitions, but your task is NOT to decide validity.
    Your ONLY job is to infer which CONDITION predicates should be marked true or false
    based on the scenario description.

    ------------------------------------
    INPUT FORMAT
    ------------------------------------
    ORIGINAL SCENARIO:
    {original}

    RULES (in Clingo-like format):
    {rules}

    SCENARIO:
    {scenario_text}

    ------------------------------------
    YOUR TASK
    ------------------------------------
    1. Read the scenario.
    2. Look at every condition(...) appearing in the rules.
    3. For each condition, output a TRUE or FALSE value based ONLY on what the scenario states.
    - If the scenario clearly supports the condition → TRUE
    - If the scenario contradicts it → FALSE
    - If the scenario gives no information → FALSE (default to false; do NOT guess)

    4. Output ONLY key = boolean pairs for each unique condition argument.
    Example format:
        firefighter = True
        unresolved_safety_violation = False
        during_wildfire_alert = True

    5. Do NOT output explanations. Do NOT restate the rules. Do NOT infer anything not in the scenario.

    ------------------------------------
    OUTPUT FORMAT (strict)
    ------------------------------------
    <condition_name> = True/False
    (one per line)'''
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def parse_constraints(constraints: str) -> str:
    """
    Parse constraint string into Clingo facts.
    
    Args:
        constraints: String with "condition = True/False" lines
        
    Returns:
        Clingo facts string
    """
    constraints = constraints.split("\n")
    constraints = [constraint.split(" = ") for constraint in constraints if "=" in constraint]
    constraints = {constraint[0].strip(): constraint[1].strip() for constraint in constraints if len(constraint) == 2}
    constraints_str = ""

    for key, value in constraints.items():
        if value == "True":
            constraints_str += f"holds({key}).\n"
        else:
            constraints_str += f"not_holds({key}).\n"
    return constraints_str


def get_clingo_rules(path: str) -> str:
    """Load Clingo program from file, removing comments."""
    with open(path) as f:
        program = f.read()
    return program.split("%")[0]


def get_original_scenario(example_id: int, input_file: str = "testing_texts.json") -> Tuple[str, str]:
    """
    Get original text and scenario for an example.
    
    Args:
        example_id: Example ID number
        input_file: Path to JSON file with examples
        
    Returns:
        Tuple of (original_text, scenario_text)
    """
    with open(input_file) as f:
        data = json.load(f)
    example = data[example_id - 1]
    return example["text"], example.get("scenario", "")


def test_clingo(constraints: str, clingo_program_path: str) -> List[List[str]]:
    """
    Test Clingo program with constraints and return models.
    
    Args:
        constraints: Clingo facts string
        clingo_program_path: Path to Clingo program file
        
    Returns:
        List of models (each model is a list of atom strings)
    """
    with open(clingo_program_path) as f:
        clingo_program = f.read()

    ctl = clingo.Control()
    ctl.add("base", [], clingo_program)
    ctl.add("base", [], constraints)
    ctl.ground([("base", [])])
    print("=== Models ===")
    models = []

    def on_model(m):
        # Only store shown atoms (clean output)
        print(m)
        models.append([str(sym) for sym in m.symbols(shown=True)])

    ctl.solve(on_model=on_model)

    return models


def test_clingo_all(
    ids_list: List[int],
    input_file: str = "testing_texts.json",
    clingo_dir: str = "clingo_programs",
    output_file: str = "clingo_test_results.json"
):
    """
    Test all examples with Clingo solver.
    
    Args:
        ids_list: List of example IDs to test
        input_file: Path to JSON file with examples
        clingo_dir: Directory containing Clingo programs
        output_file: Path to save results
    """
    results = []
    for i in ids_list:
        clingo_path = Path(clingo_dir) / f"clingo_program_{i}.txt"
        if not clingo_path.exists():
            print(f"Warning: Clingo program not found: {clingo_path}")
            continue
            
        print(f"\nTesting example {i}")
        print(f"Clingo program: {clingo_path}")
        
        original, scenario_text = get_original_scenario(i, input_file)
        rules = get_clingo_rules(str(clingo_path))
        
        print("ORIGINAL:", original)
        print("SCENARIO:", scenario_text)
        print("RULES:", rules[:200] + "..." if len(rules) > 200 else rules)
        
        constraints_raw = get_constraints(original, scenario_text, rules)
        constraints = parse_constraints(constraints_raw)
        print("CONSTRAINTS:", constraints)

        result = test_clingo(constraints, str(clingo_path))
        results.append({
            "id": i,
            "original": original,
            "scenario_text": scenario_text,
            "result": result,
            "constraints": constraints
        })

    with open(output_file, "w") as outfile:
        json.dump(results, outfile, indent=2)
    
    print(f"\nResults saved to {output_file}")


def test_clingo_zero_shot(
    ids_list: List[int],
    input_file: str = "testing_texts.json",
    output_file: str = "clingo_zero_shot_test_results.json"
):
    """
    Test zero-shot LLM conversion to Clingo (no pipeline).
    
    Args:
        ids_list: List of example IDs to test
        input_file: Path to JSON file with examples
        output_file: Path to save results
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = OpenAI(api_key=api_key)
    results = []
    
    with open(input_file) as f:
        texts = {ex["id"]: ex for ex in json.load(f)}
    
    for i in ids_list:
        if i not in texts:
            print(f"Warning: Example {i} not found in {input_file}")
            continue
            
        example = texts[i]
        original = example["text"]
        scenario_text = example.get("scenario", "")
        
        prompt = f'''Convert the following legal text into Answer Set Programming (ASP) rules using 
        Clingo syntax. Represent obligations, permissions, and prohibitions as ASP 
        predicates. Extract all actors, actions, conditions, and exceptions directly 
        from the text. Use your best judgment to infer the rule structure and logical 
        form. 

        Requirements:
        - Rewrite the legal text into a set of ASP rules.
        - Invent predicate names as needed using lowercase letters.
        - Convert natural-language modality words ("shall", "may", "must not", 
        "is prohibited", etc.) into ASP predicates such as obligation(...), 
        permission(...), forbidden(...).
        - If the text implies conditions or exceptions, encode them using rule bodies.
        - If multiple rules are implied, split them into separate ASP rules.
        - All rules must be grounded directly from your interpretation of the sentence.
        - Do not ask questions. Output only Clingo rules.
        - clingo rules have the form rule(ID, modality, head_predicate), condition(ID, antecedent_atom).
        - only output the clingo rules according to the schema, no other text.

        Input legal text:
        """
        {original}
        """

        Output JSON:
            {{
            "rules": "...",
            "explanation": "..."
            }}
        '''
        curr_results = []
        for run in range(3):
            print(f"Zero-shot run {run+1} for example {i}...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            curr_results.append(result)
        
        results.append({
            "id": i,
            "original": original,
            "scenario_text": scenario_text,
            "result": curr_results,
        })
        print(f"Finished example {i}")
    
    with open(output_file, "w") as outfile:
        json.dump(results, outfile, indent=2)
    
    print(f"\nResults saved to {output_file}")

