"""
Symbol Table for managing legal entities and predicates.

This module maintains a canonical mapping of legal terms to their
standardized representations for consistent rule formalization.
"""

from typing import Dict, List, Set
import re


class SymbolTable:
    """
    Manages a table of canonical symbols for legal entities and predicates.
    
    The symbol table ensures consistent representation of legal concepts
    across different rules and examples.
    """
    
    def __init__(self, entries: Dict[str, str] = None):
        """
        Initialize symbol table.
        
        Args:
            entries: Optional initial symbol mappings. If None, uses default entries.
        """
        if entries is None:
            self.entries = {
                "Court": "Court",
                "Injunction": "Injunction",
                "CivilPenalty": "CivilPenalty",
                "Imprisonment": "Imprisonment",
                "Violation": "Violation",
                "Enforcement": "Enforcement",
            }
        else:
            self.entries = entries

    def update_entries(
        self, 
        matches: Dict[str, str], 
        verbose: bool = True
    ) -> Dict[str, str]:
        """
        Add new symbols to the symbol table for atoms that didn't match existing symbols.
        
        Args:
            matches: Dictionary mapping atoms to matched symbols (None if no match)
            verbose: Whether to print updates
            
        Returns:
            Updated entries dictionary
        """
        new_symbols_added = []
        
        for atom, matched_symbol in matches.items():
            if matched_symbol is None:
                # Add new symbol if not already in table
                if atom not in self.entries:
                    self.entries[atom] = atom
                    new_symbols_added.append(atom)
                    if verbose:
                        print(f"  ✓ Added new symbol: {atom}")
        
        if new_symbols_added and verbose:
            print(f"\n{len(new_symbols_added)} new symbol(s) added to table")
        
        return self.entries
    
    def extract_symbols_from_rules(self, formal_rules: List[Dict]) -> Set[str]:
        """
        Extract all symbols used in the formalized rule heads.
        
        Args:
            formal_rules: List of formalized rule dictionaries with 'head' field
            
        Returns:
            Set of symbol strings found in rules
        """
        symbols = set()
        
        for rule in formal_rules:
            head = rule.get('head', '')
            # Extract predicate name (before parenthesis)
            if '(' in head:
                predicate = head.split('(')[0].strip()
                # Split by spaces to get individual words
                words = predicate.split()
                symbols.update(words)
                
                # Also extract arguments inside parentheses
                args_part = head.split('(')[1].split(')')[0]
                args = [arg.strip() for arg in args_part.split(',')]
                for arg in args:
                    # Remove any logical operators or spaces
                    clean_arg = re.sub(r'[^A-Za-z0-9]', '', arg)
                    if clean_arg:
                        symbols.add(clean_arg)
            else:
                # No parentheses, treat entire head as symbol
                clean_head = re.sub(r'[^A-Za-z0-9]', '', head)
                if clean_head:
                    symbols.add(clean_head)
        
        return symbols

