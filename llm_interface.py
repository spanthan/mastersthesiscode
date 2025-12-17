from openai import OpenAI
from typing import List, Dict, Set, Tuple
import json
import re
import numpy as np

class LLMInterface:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def rewrite_rule(self, rule_text: Dict) -> Dict:
        prompt = f"""
            You are a legal analyst trained in logical interpretation.

            Given this legal rule:
            \"\"\"{rule_text}\"\"\"

            Rewrite it into one or more **clear, logically distinct statements** that make the implicit conditions explicit.
            - If the rule includes phrases like "without", "unless", or "except", split it into two statements:
            1. A default prohibition or obligation.
            2. An exception or permission that applies when the condition is met.
            - Keep the language simple, declarative, and neutral.
            - Preserve the original meaning and legal intent.
            - Try to keep the words across the sentences as similar as possible.

            Return JSON only in this format:
            {{
            "simplified_rules": [
                "...", 
                "..."
            ],
            "explanation": "..."
            }}
        """
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    def extract_rule_segments(self, text: str) -> Dict:
        """Split statute into distinct normative statements."""
        prompt = f"""
            You are a legal analyst.

            Step 1: Split this statute into *distinct normative statements*,
            each expressing a single obligation, permission, or prohibition.

            For each statement, include:
            - id: r1, r2, ...
            - natural_language_rule: concise restatement in plain English. restate in a way that is easier to understand if needed, paying attention to logical flow and structure.
            - reason_role: MAIN / EXCEPTION / STRONG_EXCEPTION / CTD
            - priority_over: list of ids it overrides (if any)
            - modality_hint: shall / may / shall not / must not …

            Text:
            \"\"\"{text}\"\"\"

            Output JSON:
            {{
            "rules": [
                {{
                "id": "r1",
                "natural_language_rule": "...",
                "reason_role": "MAIN",
                "priority_over": [],
                "modality_hint": "shall"
                }}
            ]
            }}
        """
        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    def formalize_rule(self, rule: Dict, symbol_table: Dict[str, str]) -> Dict:
        """Convert rule into Defeasible Deontic Logic form."""
        prompt = f"""
            You are a legal logician and formal logic compiler.

            Your task is to convert a natural-language rule into a strict,
            machine-readable **intermediate representation** suitable for both:

            (1) Defeasible Deontic Logic (DDL), and  
            (2) Clingo-compatible Answer Set Programming (ASP).

            STRICT GLOBAL REQUIREMENTS:
            - Do NOT invent new names, entities, individuals, variables, or symbols.
            - Use ONLY terms explicitly present in the rule text OR in the symbol table.
            - Treat roles/classes (e.g., "Policyholder", "Insurer", "Physician") as atomic symbols.
            - Never create example people like “John”, “Alice”, “PersonA”, etc.
            - Never add implicit variables. Do not introduce person placeholders.
            - Output must contain ONLY ASCII characters.

            PREDICATE + SYMBOL FORMAT
            - Predicate names: UpperCamelCase  
            - Argument names: UpperCamelCase  
            - No spaces, no hyphens, no lowercase words unless they already appear that way in symbol table.
            - If a term is in the symbol table, reuse it EXACTLY.

            ARGUMENT RULES:
            - Arg1 = primary actor
            - Arg2 = primary action object
            - Arg3+ = additional context parameters
            - Nested actions allowed:  Require(Insurer, Provide(...))

            NO predicate may have more than 3 arguments unless forced by the natural-language rule.

            ANTECDENT RULES (PRECISE AND FORMAL)
            - Antecedent must use ONLY: AND, NOT, TRUE  
            - NO Unicode symbols (no ¬, ∧, ⊤).  
            - Each antecedent atom must be a **valid predicate expression**:
                Predicate(A)
                Predicate(A,B)
                Predicate(A,B,C)  
            - NO English text inside antecedent.

            If rule is unconditional, antecedent = "TRUE".

            HEAD RULES
            - Head must be EXACTLY ONE predicate expression.
            - No conjunctions, no multiple heads.
            - Must follow predicate syntax exactly.

            MODALITY RULES
            - Modality must be one of:
            - FORBIDDEN
            - PERMISSION
            - OBLIGATION

            MAPPING GUIDELINES:
            1. Identify actors, objects, and predicates ONLY from the rule text.  
            2. Reuse symbols from symbol table when possible.  
            3. Convert multi-word phrases into CamelCase symbols:
                "non emergency services" → NonEmergencyServices
            4. Do NOT infer participants.  
            5. Do NOT generate examples.  

            INPUT
            Rule (text): "{rule['natural_language_rule']}"
            Modality hint: {rule['modality_hint']}
            Known symbols: {json.dumps(list(symbol_table.keys()), indent=2)}

            Convert the rule now.
            Output JSON:
            {{
            "id": "{rule['id']}",
            "antecedent": "TRUE or Atom1 AND Atom2 AND ...",
            "modality": "FORBIDDEN | PERMISSION | OBLIGATION",
            "head": "Predicate(Arg1, Arg2, Arg3)",
            "explanation": "Short explanation of how the structure was derived."
            }}
        """

        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)

    def extract_priorities(self, formal_rules: List[Dict]) -> List[str]:
        priorities = []
        for rule in formal_rules['rules']:
            priorities.append((rule['id'], rule['priority_over']))
        return priorities

    def get_embedding(self, text: str) -> np.ndarray:
        """Get OpenAI embedding vector for a word or phrase."""
        if not text:
            return np.zeros(1536)
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=float)


    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def get_symbol_embeddings(self, atoms: List[str], symbol_table: Dict[str, str]) -> Tuple[Dict, Dict]:
        """Get embeddings for both atoms and existing symbols."""
        atoms_embeddings = {
            atom: self.get_embedding(atom) for atom in atoms
        }
        symbol_embeddings = {
            key: self.get_embedding(key) for key in symbol_table.keys()
        }
        return atoms_embeddings, symbol_embeddings


    def match_atoms_to_symbols(self, atoms_embeddings: Dict, symbol_embeddings: Dict, 
                            threshold: float = 0.50, verbose: bool = True) -> Dict[str, str]:
        """
        Return dict mapping each atom → closest symbol name or None.
        If None, it means we need to add this atom as a new symbol.
        """
        matches = {}
        for atom in atoms_embeddings.keys():
            atom_emb = atoms_embeddings[atom]
            best_symbol = None
            best_score = 0.0

            for sym, emb in symbol_embeddings.items():
                score = self.cosine_similarity(atom_emb, emb)
                if score > best_score:
                    best_score, best_symbol = score, sym

            print(f"{atom:20} → {best_symbol:20} (score: {best_score:.3f})") if verbose else None   

            if best_score >= threshold:
                matches[atom] = best_symbol
            else:
                matches[atom] = None  # New symbol needed

        return matches

    def generate_z3_from_rules(self, rules):
        

        # Serialize input rules into readable JSON for the prompt
        rules_json = json.dumps(rules, indent=2)

        # ======================== PROMPT ===============================
        prompt = f"""
            You are a formal verification engineer. Convert the following natural-language rules
            into fully executable Z3 Python code.

            ======================================================================
            INPUT RULES
            ======================================================================
            {rules_json}

            ======================================================================
            YOUR TASK
            ======================================================================
            Produce **complete executable Z3 Python code** that:

            1. Imports Z3:
                from z3 import *
                s = Solver()

            2. Defines:
            - Entity sort:
                    Entity = DeclareSort('Entity')
            - Constants for all named roles/objects (Insurer, Policyholder, etc.)
            - Predicates for all action verbs (Provide, Require, Pay, etc.)
            - Modal operators as Bool -> Bool functions:
                    Obligation = Function('Obligation', BoolSort(), BoolSort())
                    Forbidden  = Function('Forbidden',  BoolSort(), BoolSort())
                    Permission = Function('Permission', BoolSort(), BoolSort())

            3. Converts each rule into an implication:
                s.add(Implies( antecedent , Modality( HeadPredicate(...) ) ))

            4. Maps modalities:
                "shall"      → Obligation(...)
                "must"       → Obligation(...)
                "shall not"  → Forbidden(...)
                "must not"   → Forbidden(...)
                "may"        → Permission(...)

            5. Builds antecedents using ONLY:
                AND, NOT, TRUE
            No Unicode symbols (¬, ∧, ⊤).

            6. Creates action predicates in CamelCase:
                Provide(Insurer, EmergencyMedicalCoverage, Policyholder)
                Require(Insurer, PriorAuthorization, NonEmergencyServices)

            7. NO quantifiers. Everything must be quantifier-free.

            8. NO invented persons or entities.
            Use ONLY:
            - terms from the input rules
            - CamelCase conversion for multi-word tokens

            9. Do NOT encode rule priorities.

            10. Program MUST end with:
                print("SAT?", s.check())
                print("Model:", s.model())

            ======================================================================
            OUTPUT FORMAT
            ======================================================================
            Return ONLY the Z3 Python code in a **single fenced block**:

            ```python
            # Z3 encoding
            ...
            ======================================================================
            BEGIN NOW
        """

        # ======================== LLM CALL ============================
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
            )
            return resp.choices[0].message.content

        except Exception as e:
            return f"Error generating Z3 code: {str(e)}"