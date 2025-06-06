import re
from typing import List

from drug_named_entity_recognition import find_drugs

from pk_el.tokenizers.patterns import (SPECIAL_CHARACTER_MAP,
                                       CHEMICALS_SET, TOKEN_REMOVALS_SET, GENERAL_REPLACEMENTS, TOKEN_RE,
                                       TERM_STANDARDIZATION,
                                       RANGE_STANDARDIZATION, NUMERIC_RANGE_RE, ORDERED_PARAMETER_REPLACEMENTS)
from pk_el.tokenizers.basic_tokenizer import basic_preprocessing

def remove_drugnames(text: str) -> str:
    """
    Removes drug name spans from the input string and returns the cleaned string.
    """
    tokens = text.split()
    results = find_drugs(tokens)
    if results:
        # Extract indexes of drug name tokens to remove
        token_indexes_to_remove = {res[2] for res in results}  # Use a set for O(1) lookup
        tokens = [token for index, token in enumerate(tokens) if index not in token_indexes_to_remove]

    return " ".join(tokens)

def remove_chemicals(text: str) -> str:
    """
    Removes common chemical names from a string using whole-word matching.
    Returns the cleaned string.
    """
    # Join into a regex that matches whole chemical names (with optional plural 's')
    pattern = re.compile(
        r'\b(?:' + '|'.join(re.escape(chem) for chem in CHEMICALS_SET) + r')\b',
        flags=re.IGNORECASE
    )
    cleaned = pattern.sub('', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

def replace_special_chars(tokens):
    """Replace special characters in tokens using direct dict lookup."""
    return [SPECIAL_CHARACTER_MAP.get(token, token) for token in tokens]


def deduplicate_tokens(tokens):
    """Remove duplicates while maintaining order."""
    return list(dict.fromkeys(tokens))


def pk_tokenizer(text: str) -> List[str]:
    """
    Tokenizer for pharmacokinetic (PK) parameters that applies preprocessing,
    normalization, pattern extraction, and filtering.
    """
    # Basic preprocessing
    text = basic_preprocessing(text)
    # remove drug names & chemicals
    text = remove_drugnames(text)
    text = remove_chemicals(text)

    # Apply grouped regex replacements for better performance
    for replacement_group in GENERAL_REPLACEMENTS:
        for pattern, replacement in replacement_group:
            text = pattern.sub(replacement, text)

    # Apply parameter tokenizers in the correct order
    for pattern, replacement in ORDERED_PARAMETER_REPLACEMENTS:
        text = pattern.sub(replacement, text)
        """print(pattern)
        print(text)
        print("\n")"""

    # Additional specific tokenizers
    text = re.sub(r"\bc\s*(?:versus|vs|[-:/])\s*t\b|\bcxt\b", "", text, flags=re.IGNORECASE)  # Remove conc vs time
    text = re.sub(r'\b(?:phase)\b', "", text, flags=re.IGNORECASE)  # Remove phase
    text = re.sub(r"\b(?:apparent|app|z)\b", "bionorm", text, flags=re.IGNORECASE)  # Normalize 'apparent' terms

    # Extract tokens using compiled regex
    tokens = TOKEN_RE.findall(text)

    # Clean tokens
    tokens = [SPECIAL_CHARACTER_MAP.get(token, token) for token in tokens]
    updated_tokens = []
    for token in tokens:
        if token not in TOKEN_REMOVALS_SET:
            # Apply term standardizations
            for pattern, replacement in TERM_STANDARDIZATION.items():
                token = pattern.sub(replacement, token)

            # Apply range standardizations
            for pattern, replacement in RANGE_STANDARDIZATION.items():
                token = pattern.sub(replacement, token)

            # Skip numeric-numeric ranges
            if not NUMERIC_RANGE_RE.match(token):
                updated_tokens.append(token)

    # Remove duplicate tokens and sort
    return sorted(dict.fromkeys(updated_tokens))




