# -----------------------------------------------------------------------------
# Regex to categories patterns
# -----------------------------------------------------------------------------
import re

category_patterns = {
    "exposure_patterns" : (
    r"(?:"
    r"auc"
    r"|aumc"
    r"|area[\s\-_]*under"
    r")"
    ),
    "bioav_patterns": (
        r"(?:"
        r"(?<![\w/])f(?!\w)"
        r"|\bf[\s\-_]*r\b"
        r"|\bf[\s\-_]*g"
        r"|\bf[\s\-_]*h"
        r"|\bf[\s\-_]*rel"
        r"|\bf[\s\-_]*ab"
        r"|bioavailability"
        r"|absorption"
        r")"
    ),
    "cl_patterns": (
        r"(?:"
        r"\bcl"
        r"|clearance"
        r"|\bcl[\s\-_]*r"
        r"|\bcl[\s\-_]*b"
        r"|\bcl[\s\-_]*nr"
        r"|\bcl[\s\-_]*u\b"
        r"|\bcl[\s\-_]*h"
        r"|(?<![\w/])q(?!\w)"
        r"|\bq[\s\-_]*2"
        r"|\bae"
        r"|\bfe\b"
        r"|excretion"
        r"|excreted"
        r"|recovery"
        r")"
    ),
    "time_patterns": (
        r"(?:"
        "t[\s\-_]*max"
        "|t[\s\-_]*lag"
        "|t[\s\-_]*ss"
        "|ss[\s\-_]*t"
        "|mrt"
        r"|(?<!\w)mat\b"
        "|mtt"
        r")"
    ),
    "conc_patterns": (
        r"(?:"
        r"concentration"
        r"|\bc[\s\-_]*max"
        r"|\bc[\s\-_]*0"
        r"|\bc[\s\-_]*t"
        r"|\bc[\s\-_]*min"
        r"|\bc[\s\-_]*ss"
        r"|\bc[\s\-_]*avg"
        r"|\bc[\s\-_]*trough"
        "|peaked"
        "|peak"
        r"|(?<![\w/])c\b"
        r")"
    ),
    "half_life_patterns": (
        r"(?:"
        r"\bt[\s\-_]*1/2"
        "|half[\s\-_]*life"
        r")"
    ),
    "rate_constant_patterns": (
        r"(?:"
        "rate[\s\-_]constant"
        r"|\bk[\s\-_]*el"
        r"|\bk[\s\-_]*excr"
        r"|\bk[\s\-_]*abs"
        r"|\bk[\s\-_]*met"
        r"|\bk[\s\-_]*m"
        r"|\bk[\s\-_]*comp"
        r"|(?<![\w/])k\b"
        r"|alpha|beta\b"
        r")"
    ),
    "volume_patterns": (
        r"(?:"
        "volume"
        r"|\bv[\s\-_]*1"
        r"|\bv[\s\-_]*2"
        r"|\bv[\s\-_]*3"
        r"|(?<![\w/])v"
        r")"
    ),
    "metab_excr_patterns": (
        r"(?:"
        r"\bv[\s\-_]*max"
        r"|\bk[\s\-_]*m"
        r"|\bf[\s\-_]*m"
        r"|\bf[\s\-_]*u"
        r"|\beh"
        r"|(?<![\w/])e(?!\w)"
        "|blood[\s\-_]*flow"
        "|metabolic"
        "|metabolism"
        "|extraction"
        "|michaelis"
        "|enzyme"
        "|unbound"
        r")"
    ),

}


PATTERNS_TO_CATEGORIES = {
            "exposure_patterns": ['G1'],
            "bioav_patterns": ['G2'],
            "cl_patterns": ['G3'],
            r"conc_patterns": ['G4'],
            r"half_life_patterns": ['G5'],
            "rate_constant_patterns": ['G6'],
            "time_patterns": ['G7'],
            "volume_patterns": ['G8'],
            "metab_excr_patterns": ['G9'],
        }

CATEGORY_PRIORITIES = [
    {'G1', 'G2', 'G3', 'G5'},
    {'G4', 'G5',  'G9', 'G7'},
    {'G8', 'G6'}
]

# Create mapping from compiled pattern to categories
COMPILED_PATTERN_TO_CATEGORIES = {
    re.compile(category_patterns[key], re.IGNORECASE): categories
    for key, categories in PATTERNS_TO_CATEGORIES.items()
    if key in category_patterns
}


patterns_to_parameters = {
    r"ratio": ['Q21', 'Q31', 'Q33', 'Q35', 'Q58', 'Q93'],
    r"ss\b": ['Q18', 'Q34', 'Q72', 'Q65'],
    r"bionorm": ["Q27", "Q69", "Q80", "Q76", "Q290", "Q82", "Q78"],
    r"dosenorm": ["Q189", "Q174"],
    r'\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?': ["Q19"],
    r'\d+(?:\.\d+)?\s*-\s*[a-zA-Z]+': ["Q19", "Q18", "Q74", "Q17", "Q20", "Q189", "Q21"],
    r"max\b": ["Q66", "Q56", "Q93", "Q32", "Q174", "Q33"],
}

COMPILED_PATTERNS_TO_PARAMETERS = {
    re.compile(pattern, re.IGNORECASE): parameters
    for pattern, parameters in patterns_to_parameters.items()
}