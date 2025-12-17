import json
import re
import spacy
import argparse
from typing import List, Dict, Tuple

from llm_interface import LLMInterface
# from z3_interface import compile_to_z3
from symbol_table import SymbolTable
from clingo_interface import (
    convert_all_rules,
    get_rules,
    test_clingo_all
)

# =============================================================
# Core DDL Pipeline (Used by SimplePipeline + FullPipeline)
# =============================================================

class DDLChain:
    def __init__(self):
        self.llm = LLMInterface()
        self.symbol_table = SymbolTable()
        self.nlp = spacy.load("en_core_web_sm")
    
    def canonicalize(self, text: str) -> str:
        """CamelCase and clean noun/verb phrases."""
        if not text:
            return ""
        text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if t.lower() not in {"the", "a", "an"}]
        return "".join(word.capitalize() for word in tokens)
    

    def extract_atomic_phrases(self, text: str) -> List[Dict[str, str]]:
        """
        Extract subject-verb-object atomic phrases with full NPs.
        """
        doc = self.nlp(text)
        triples = []

        def expand_np(token):
            left_mods = [t for t in token.lefts if t.dep_ in {"compound", "amod"}]
            rights = [
                f"of {c.text}" for t in token.rights
                if t.dep_ == "prep" and t.text.lower() == "of"
                for c in t.children if c.dep_ == "pobj"
            ]
            phrase = " ".join([t.text for t in left_mods] + [token.text] + rights)
            return self.canonicalize(phrase)

        for tok in doc:
            # VERB or xcomp-embedded verb
            if tok.pos_ != "VERB":
                continue

            # Skip auxiliary verbs entirely
            if tok.dep_ == "aux":
                continue

            # If the verb has an xcomp like "to edit", switch to that instead
            xcomp_child = next((c for c in tok.children if c.dep_ == "xcomp" and c.pos_ == "VERB"), None)
            if xcomp_child:
                tok = xcomp_child

            subj, obj, extras = None, None, []
            for ch in tok.children:
                if ch.dep_ in {"nsubj", "nsubjpass"}:
                    subj = expand_np(ch)
                elif ch.dep_ in {"dobj", "attr", "pobj"}:
                    obj = expand_np(ch)
                elif ch.dep_ == "prep":
                    pobj = next((c for c in ch.children if c.dep_ == "pobj"), None)
                    if pobj:
                        extras.append(self.canonicalize(f"{ch.text} {expand_np(pobj)}"))

            if subj or obj:
                triples.append({
                    "subject": subj or "_",
                    "verb": self.canonicalize(tok.lemma_),
                    "object": obj or "",
                    "extras": extras
                })

        return triples
    
    def collect_atoms(self, triples: List[Dict[str, str]]) -> List[str]:
        """Collect all non-empty atoms from extracted triples."""
        atoms = set()
        for t in triples:
            for val in t.values():
                if val in ["", "_", []]:
                    continue
                if isinstance(val, list):
                    atoms.update(val)
                else:
                    atoms.add(val)
        return list(atoms)


    def run(self, rewritten_text: str, symbol_table: Dict[str, str], verbose=True) -> Dict:
        """
        Full pipeline:
            1. extract atoms
            2. match atoms → symbols
            3. segment rules
            4. formalize rules
            5. update symbol table
        """
        # ---------------------------------------------------------
        # 1. Extract atoms
        # ---------------------------------------------------------
        triples = self.extract_atomic_phrases(rewritten_text)
        atoms = self.collect_atoms(triples)
        print("Triples: ", triples, "\nAtoms: ", atoms)

        # ---------------------------------------------------------
        # 2. Symbol matching
        # ---------------------------------------------------------
        atoms_emb, symbols_emb = self.llm.get_symbol_embeddings(atoms, symbol_table)
        matches = self.llm.match_atoms_to_symbols(atoms_emb, symbols_emb, verbose=verbose)

        symbol_table = self.symbol_table.update_entries(matches, verbose=verbose)

        # ---------------------------------------------------------
        # 3. Rule segmentation
        # ---------------------------------------------------------
        segmentation = self.llm.extract_rule_segments(rewritten_text)
        priorities = self.llm.extract_priorities(segmentation)

        # ---------------------------------------------------------
        # 4. Formalization
        # ---------------------------------------------------------
        formal_rules = [
            self.llm.formalize_rule(rule, symbol_table)
            for rule in segmentation["rules"]
        ]

        # ---------------------------------------------------------
        # 5. Final symbol harvesting
        # ---------------------------------------------------------
        new_syms = self.symbol_table.extract_symbols_from_rules(formal_rules)
        for s in new_syms:
            symbol_table[s] = s

        return {
            "atoms": atoms,
            "atom_matches": matches,
            "segmentation": segmentation,
            "formal_rules": formal_rules,
            "final_symbol_table": symbol_table,
            "priorities": priorities
        }

# =============================================================
# Full Experiment Pipeline (Multiple Examples)
# =============================================================

class FullPipeline:
    """
    Runs your *full evaluation pipeline* with:
        - multi-run reproducibility
        - dataset file support
        - Clingo program outputs
    """

    def __init__(self, verbose=True):
        self.ddl = DDLChain()
        self.symbol_table = SymbolTable().entries.copy()
        self.verbose = verbose
        print("Initialized FullPipeline")

    def reset_symbols(self):
        self.symbol_table = SymbolTable().entries.copy()

    def run_one(self, text: str):
        rewritten = self.ddl.llm.rewrite_rule(text)
        rewritten_str = " ".join(rewritten["simplified_rules"])

        # DDL processing
        result = self.ddl.run(
            rewritten_text=rewritten_str,
            symbol_table=self.symbol_table,
            verbose=self.verbose
        )

        self.symbol_table = result["final_symbol_table"]

        # Build ASP
        rules = []
        for i, r in enumerate(result["segmentation"]["rules"]):
            ant = result["formal_rules"][i]["antecedent"]
            mod = result["formal_rules"][i]["modality"]
            head = result["formal_rules"][i]["head"]
            nl = r["natural_language_rule"]
            pr = result["priorities"][i][1]
            rules.append((f"{ant} -> [{mod}] {head}", nl, pr))
        

        # z3_code = compile_to_z3(result["formal_rules"])
        # print("Z3 code:")
        # print(z3_code)

        asp_program = convert_all_rules(rules)

        return {
            "rewritten": rewritten_str,
            "formal_rules": result["formal_rules"],
            "asp_program": asp_program,
            "symbol_table": self.symbol_table,
            "segmentation": result["segmentation"],
            "priorities": result["priorities"]
        }
    
    def run_n_times(self, text: str, num_times: int = 3):
        """
        Run the pipeline 3 times with a FRESH symbol table each run.
        Does NOT let earlier runs influence later ones.
        """
        outputs = []
        for _ in range(num_times):
            self.reset_symbols()

            out = self.run_one(text)
            outputs.append(out)
        return outputs

    def run_many(self, texts: List[str]):
        return [self.run_one(t) for t in texts]

    def run_from_file(self, filename: str):
        with open(filename) as f:
            data = json.load(f)

        outputs = []
        for ex in data:
            tid = ex["id"]
            text = ex["text"]

            print(f"\n=== EXAMPLE {tid} ===")
            out = self.run_one(text)

            # Save JSON output
            with open(f"testing_results/test_output_{tid}.json", "w") as fp:
                json.dump(out, fp, indent=2)

            # Save Clingo program
            with open(f"clingo_programs/clingo_program_{tid}.txt", "w") as fp:
                fp.write(out["asp_program"])

            outputs.append(out)

        return outputs

class ZeroShotPipeline:
    """
    Zero-shot rule generator:
        - No symbol tables
        - No DDL pipeline
        - No segmentation or formalization
        - Direct LLM → ASP rule generation
        - Supports single or batch processing
        - Matches original test_clingo_zero_shot behavior
    """

    def __init__(self, verbose=True):
        self.client = LLMInterface().client   # reuse OpenAI client
        self.verbose = verbose

    # ---------------------------------------------------------
    # Core zero-shot prompt
    # ---------------------------------------------------------
    def _build_prompt(self, text: str) -> str:
        return f'''
            Convert the following legal text into Answer Set Programming (ASP) rules using 
            Clingo syntax. Represent obligations, permissions, and prohibitions as ASP 
            predicates. Extract all actors, actions, conditions, and exceptions directly 
            from the text. Use your best judgment to infer the rule structure and logical 
            form. 

            Requirements:
            - Rewrite the legal text into a set of ASP rules.
            - Invent predicate names as needed using lowercase letters.
            - Convert natural-language modality words (“shall”, “may”, “must not”, 
            “is prohibited”, etc.) into ASP predicates such as obligation(...), 
            permission(...), forbidden(...).
            - If the text implies conditions or exceptions, encode them using rule bodies.
            - If multiple rules are implied, split them into separate ASP rules.
            - All rules must be grounded directly from your interpretation of the sentence.
            - Do not ask questions. Output only Clingo rules.
            - clingo rules have the form:
                -- rule(ID, modality, head_predicate).
                -- condition(ID, antecedent_atom).
                -- overrides(rule1, rule2).
            - only output the clingo rules according to the schema, no other text.

            Input legal text:
            \"\"\" 
            {text}
            \"\"\"

            Output JSON:
            {{
            "rules": "...",
            "explanation": "..."
            }}
        '''

    # ---------------------------------------------------------
    # Run zero-shot generation ONCE
    # ---------------------------------------------------------
    def run_one(self, text: str) -> Dict:
        prompt = self._build_prompt(text)

        if self.verbose:
            print("\n[ZeroShot] Generating rules...")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )

        result = json.loads(response.choices[0].message.content)

        return {
            "input": text,
            "rules": result.get("rules", ""),
            "explanation": result.get("explanation", "")
        }

    # ---------------------------------------------------------
    # Run THREE zero-shot passes (reproducibility testing)
    # ---------------------------------------------------------
    def run_three(self, text: str) -> List[Dict]:
        if self.verbose:
            print("[ZeroShot] Running 3 reproducibility passes...")
        return [self.run_one(text) for _ in range(3)]

    # ---------------------------------------------------------
    # Run zero-shot pipeline on a list of texts
    # ---------------------------------------------------------
    def run_many(self, text_list: List[str]):
        outputs = []
        for t in text_list:
            outputs.append(self.run_three(t))
        return outputs

    # ---------------------------------------------------------
    # Run zero-shot pipeline from JSON dataset file
    # ---------------------------------------------------------
    def run_from_file(self, filename: str, save_path="clingo_zero_shot_test_results.json"):
        with open(filename, "r") as f:
            data = json.load(f)

        results = []

        for ex in data:
            tid = ex["id"]
            text = ex["text"]

            if self.verbose:
                print(f"\n=== ZERO-SHOT EXAMPLE {tid} ===")

            three_runs = self.run_three(text)

            results.append({
                "id": tid,
                "original": text,
                "scenario_text": ex.get("scenario", None),
                "result": three_runs
            })

        # Save final test output
        with open(save_path, "w") as fp:
            json.dump(results, fp, indent=2)

        return results
