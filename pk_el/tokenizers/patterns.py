"""
PK Patterns Module

This module contains all regex tokenizers and replacements for the PK tokenizer.
Patterns are precompiled at import time for efficiency.
"""

import re

# -----------------------------------------------------------------------------
# Basic character tokenizers
# -----------------------------------------------------------------------------

DASH_VARIANTS = [
    "\u2010",  # Hyphen (‐)
    "\u2011",  # Non-breaking hyphen (‑)
    "\u2012",  # Figure dash (‒)
    "\u2013",  # En dash (–)
    "\u2014",  # Em dash (—)
    "\u2015",  # Horizontal bar (―)
    "\u2212",  # Minus sign (−)
    "\uFE58",  # Small em dash (﹘)
    "\uFE63",  # Small hyphen-minus (﹣)
    "\uFF0D"  # Full-width hyphen-minus (－)
]
DASH_PATTERN = "[" + "".join(DASH_VARIANTS) + "]"

# -----------------------------------------------------------------------------
# Common word lists and sets
# -----------------------------------------------------------------------------

COMMON_STOPWORD_PATTERNS = (
    r"\bof\b"
    r"|\bat\b"
    r"|\bby\b"
    r"|\bon\b"
    r"|\bthe\b"
    r"|\bfrom\b"
    r"|\band\b"
    r"|\bis\b"
    r"|\bin\b"
    r"|\ban\b"
    r"|\bwith\b"
    r"|\bby\b"
    r"|\bfor\b"
    r"|\bit\b"
    r"|\bas\b"
    r"|\bbetween\b"
    r"|\bbased\b"
)

COMMON_CHEMICALS_STUDIED = [
    "glucose", "inulin", "lactose", "sucrose", "fructose", "galactose", "mannitol",  # Sugars & Carbohydrates
    "creatinine", "bilirubin",  # Metabolites
    "cortisol", "testosterone", "estradiol", "progesterone", "insulin",  # Endogenous Steroids & Hormones
    "cholesterol", "triglycerides", "ffa",  # Lipids & Fatty Acids
    "vitamin", "folate", "biotin",  # Vitamins & Cofactors
    "caffeine", "theobromine", "nicotine", "ethanol",  # Xenobiotics & Dietary Components
    "hemoglobin", "albumin", "total protein"  # Plasma & Blood Components
]
CHEMICALS_SET = set(COMMON_CHEMICALS_STUDIED)

COMMON_PK_SAMPLES = (
    r"\b(fluid|plasma|serum|"
    r"cerebrospinal|csf|"
    r"urine|feces|"
    r"saliva|bile|gastric|sputum|"
    r"breast\s*milk|lung|"
    r"subcutaneous|sweat|tears)\b"
)

TOKEN_REMOVALS = [
    "h", "hr", "hours", "hour", "min", "mins",
    "minutes", "seconds", "s", "secs",
    "day", "days", "week", "weeks", "month",
    "months", "time", "obs", "ob", "observed",
    "systemic", "to", "period", "compartment", "level"
]
TOKEN_REMOVALS_SET = set(TOKEN_REMOVALS)

SPECIAL_CHARACTER_MAP = {
    "λ": "lambda",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "%": "percentage",
    "z": "zeta",
    "θ": "theta"
}

RATE_CONSTANT_UNIT_REGEX = re.compile(r"(?:1\s*/\s*h|h\s*[-⁻−]\s*1)", re.IGNORECASE)
ENZYME_CONTEXT_REGEX = re.compile(
    r"\b(?:enzyme|michaelis[\s\-_]*menten|michaelis|menten|substrate|vmax|"
    r"kinetic(?:s)?|affinity|saturation"
    r"|nmol\s*(?:/|·|per)?\s*[lL](?:[-−⁻]1)?)\b",
    re.IGNORECASE
)
T_HALF_BETA_CONTEXT_REGEX = re.compile(
    r"\b(?:"
    r"two[\s\-_]*compartment(?:al)?|2[\s\-_]*compartment(?:al)?|multi[\s\-_]*compartment(?:al)?|"
    r"biphasic|slow[\s\-_]*phase|second[\s\-_]*phase|latter[\s\-_]*phase|beta[\s\-_]*phase"
    r")\b",
    re.IGNORECASE
)

T_HALF_GAMMA_CONTEXT_REGEX = re.compile(
    r"\b(?:"
    r"gamma[\s\-_]*phase|triphasic|third[\s\-_]*phase|"
    r"three[\s\-_]*compartment(?:al)?|3[\s\-_]*compartment(?:al)?"
    r")\b",
    re.IGNORECASE
)

T_HALF_Z_CONTEXT_REGEX = re.compile(
    r"\b(?:"
    r"non[\s\-_]*compartment(?:al)?|nca|"
    r"1[\s\-_]*compartment(?:al)?|one[\s\-_]*compartment(?:al)?"
    r")\b",
    re.IGNORECASE
)

# -----------------------------------------------------------------------------
# Precompiled basic tokenizers
# -----------------------------------------------------------------------------

# Text preprocessing tokenizers
STOP_WORDS_RE = re.compile(COMMON_STOPWORD_PATTERNS, re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
PLURAL_RE = re.compile(r"(?<!s)(?<!mea)(?<!michaeli)s\b")
BIO_PLURAL_RE = re.compile(r"bioavailabilities|bioavailabilitie")
HL_PLURAL_RE = re.compile(r"half[\s\-_]*(lives|live|times)")
DASH_RE = re.compile(DASH_PATTERN)
MULTIPLE_DASH_RE = re.compile(r"--+")
FRACTION_SLASH_RE = re.compile(r"⁄", re.IGNORECASE)

# Tokenization tokenizers
TOKEN_RE = re.compile(
    r't[\s\_]*1/2|'  # t1/2
    r'\d+/\d+|'  # fractions
    r'[a-zA-Z]+(?=\d+-)|'  # words before numbers with dash
    r'\d+(?:\.\d+)?\s*-\s*[a-zA-Z]+|'  # ranges like 0-anyword
    r'\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?|'  # numeric ranges
    r'\b[a-zA-Z]\d{1}\b|'  # single letter followed by single number
    r'[a-zA-Z]+|'  # words
    r'\d+(?:\.\d+)?|'  # numbers
    r'[%λαβγ]'  # special symbols
)

# Term standardization tokenizers
TERM_STANDARDIZATION = {
    re.compile(r"\b(?:ave|av|avg|mean)\b"): "average",
    re.compile(r"\b(ext|extrap)\b"): "extrapolated"
}

# Range standardization tokenizers
RANGE_STANDARDIZATION = {
    re.compile(r"0-\d+(?:\.\d+)?"): "t",
    re.compile(r"0-t"): "t",
    re.compile(r"0-inf"): "inf",
    re.compile(r"0-last"): "last"
}

NUMERIC_RANGE_RE = re.compile(r"^\d+(?:\.\d+)?-\d+(?:\.\d+)?$")

# -----------------------------------------------------------------------------
# Parameter normalization tokenizers
# -----------------------------------------------------------------------------

PARAMETER_NORMALISATION_PATTERNS = {
    # dose-normalised
    "dose_adjusted": (
        r"/dose(\d|[a-zA-Z])?\b"  # Match '/Dose' first (whole word with single number or letter at end (footnotes))
        r"|/d\b"  # Match '/D' only if not part of '/Dose'
        r"|dose[\s\-\_]+normalised"  # Match variations like 'dose normalised' or 'dose-normalised'
        r"|\(\s*dn\s*\)"  # Match '(dn)' exactly (with word boundaries)
        r"|per[\s\-\_]+dose"  # Match 'per dose', allowing underscores or hyphens
        r"|normalised"  # usually refers to does normalised when occurs alone.
    ),
    # bioavailability-normalised
    "bioav_adjusted": (
        r"_f\b"
        r"|/\s*f"
        r"|/\s*bioavailability\b"
        r"|\bbioavailability[\s\-\_]+normalised\b"
        r"|\bbioavailability[\s\-\_]+adjusted\b"
    ),
    "bw_adjusted": (
        r"\b\d+\s*kg\b"
    )
}



# -----------------------------------------------------------------------------
# General replacement tokenizers
# -----------------------------------------------------------------------------

GENERAL_REPLACEMENTS = [
    # PUNCTUATION NORMALISATIONS
    [
        #(re.compile(r'_', re.IGNORECASE), ' '),
        (re.compile(DASH_PATTERN, re.IGNORECASE), "-"),
        (re.compile(r"--+", re.IGNORECASE), "-")
    ],

    # NUMERIC NORMALISATIONS
    [
        (re.compile(r"\b(1st|primary)\b", re.IGNORECASE), "first"),
        (re.compile(r"\b(2nd|secondary)\b", re.IGNORECASE), "second"),
        (re.compile(r"\b(3rd|tertiary)\b", re.IGNORECASE), "third"),
        (re.compile(r"\bzero\b", re.IGNORECASE), "0"),
        (re.compile(r'∞|infinity', re.IGNORECASE), 'inf'),
        (re.compile(r"(?<=\d),(?=\d{3}\b)", re.IGNORECASE), ""),
        (re.compile(r"\bfraction\b", re.IGNORECASE), "percentage")
    ],

    # TIME NORMALISATIONS
    [
        (re.compile(r"\b(\d+)\s*time\b", re.IGNORECASE), r"\1"),
        (re.compile(r"(^|[\s\(\[\{.,;:-])(d|day)\s*(?=\d)", re.IGNORECASE), "")
    ],

    # RANGES NORMALISATIONS
    [
        (re.compile(r"(\d+(\.\d+)?)\s*\bto\b\s*(\d+(\.\d+)?)|\b(\d+(\.\d+)?)\s*\bto\b\s*(inf|t|last)\b", re.IGNORECASE),
         lambda match: f"{match.group(1) or match.group(5)}-{match.group(3) or match.group(7)}"),
        (re.compile(r'\(\s*(\d+)\s*,\s*([^)]+)\)', re.IGNORECASE), r'\1-\2'),
        (re.compile(r'(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)|(\d+(\.\d+)?)\s*-\s*(inf|t)', re.IGNORECASE),
         lambda m: f"{m.group(1) or m.group(5)}-{m.group(3) or m.group(7)}")
    ],

    # FORMATTING NORMALISATIONS
    [
        (re.compile(r'^\b([a-zA-Z]) ([a-zA-Z]+)\b', re.IGNORECASE), r'\1\2'),
        (re.compile(r"(\b[a-zA-Z]+)\(\s*([a-zA-Z]+)\s*\)", re.IGNORECASE), r"\1\2")
    ],

    # TERM NORMALISATIONS
    [
        (re.compile(r"\bdistributional\b", re.IGNORECASE), "distribution"),
        (re.compile(r"\bperiperhal\b", re.IGNORECASE), "peripheral"),
        (re.compile(r'\b(?:steady[\s\-]state)\b', re.IGNORECASE), 'ss'),
        (re.compile(r"(τ|tau)"), "t"),
        (re.compile(r'\b(?:normalized|normalised|normalize|normalise|norm)\b', re.IGNORECASE), "normalised"),
        (re.compile(r'(maximal|maximum)', re.IGNORECASE), 'max'),
        (re.compile(r'(minimum|minimal)', re.IGNORECASE), 'min'),
        (re.compile(r"bloavailability", re.IGNORECASE), "bioavailability"),
        (re.compile(r"t\s*\(*?\s*l\s*/\s*2\s*\)?", re.IGNORECASE), "t1/2"),
        (re.compile(r"^(θ|theta|tv)", re.IGNORECASE), ""),
    ],

    # PARAMETER SCALING NORMALISATIONS
    [
        (re.compile(PARAMETER_NORMALISATION_PATTERNS["bioav_adjusted"], re.IGNORECASE), " bionorm "),
        (re.compile(PARAMETER_NORMALISATION_PATTERNS["bw_adjusted"], re.IGNORECASE), " "),
        (re.compile(PARAMETER_NORMALISATION_PATTERNS["dose_adjusted"], re.IGNORECASE), " dosenorm ")
    ],

    # TERM REMOVALS
    [
        (re.compile(r'_', re.IGNORECASE), ' '),
        (re.compile(COMMON_PK_SAMPLES, re.IGNORECASE), ""),
        (re.compile(r"\b(?:(tv|θ)(?=\s*\w))", re.IGNORECASE), ""),
        (re.compile(r'\b(?:dose|drug|body|total|value|observed|compartment|observed)\b', re.IGNORECASE), "")
    ],

    # COMMON PARAM FORMS NORMALISATIONS
    [
        (re.compile(r"\b(?<!-)(auc|aumc|mrt|c|vd|v|t)(tend|all|tlast|last|inf|ss|z|t)\b", re.IGNORECASE), r"\1 \2")
    ]
]

# -----------------------------------------------------------------------------
# NIL patterns - common mistakes that are not PK
# -----------------------------------------------------------------------------

NIL_PATTERNS = [
    r"\bmic\b",
    r"\bminimum[\s\-_]*inhibitory[\s\-_]*concentration\b",
    r"\bceoinf\b",
    r"\bcl[\s\-_/:]*(cr|creatinine)\b",
    r"\bauc[\s\-_/:]*mic",
    r"\bauc\d+(\.\d+)?[\s\-_/:]*mic",
    r"\bauc\d+(\.\d+)?\s*[-–]\s*\d+(\.\d+)?[\s\-_/:]*mic",
    r"\bauc\d+(\.\d+)?(\s*[-–]\s*\d+(\.\d+)?)?[a-z]{1,2}\b[\s\-_/:]*mic",
    r"\beffective[\s\-_]*concentration\b",
    r"\beffective[\s\-_]*half[\s\-_]*life\b",
    r"\binhibitory[\s\-_]*concentration\b",
    r"\bic[\s\-_]*50\b", r"\bec[\s\-_]*50\b",
    r"\be[\s\-_]*max\b", r"\bi[\s\-_]*max\b",
    r"\bk[\s\-_]*off\b", r"\bk[\s\-_]*on\b",
    r"\bgfr\b", r"glomerular[\s\-_]*filtration[\s\-_]*rate",
    r"hill[\s\-_]*(slope|coefficient|constant)",
    r"\bp[\s\-_]*ka\b", r"\blog[\s\-_]*[pd]\b", r"\blog[\s\-_]*p\b",
    r"\bmolecular[\s\-_]*weight\b", r"\bmolecular[\s\-_]*mass\b",
    r"\bweight\b", r"\bmass\b",
    r"\bsolubility\b", r"\bpermeability\b", r"\bbinding[\s\-_]*affinity\b",
    r"therapeutic[\s\-_]*index",
    r"\btgf[-\s]*beta\b", r"\btnf[-\s]*alpha\b",
    r"\bcrp\b", r"\binterleukin\b",
    r"gene[\s\-_]*expression", r"\bmrna\b",
    r"\b(k[\s\-_]*d|k[\s\-_]*(tran|transit)|k[\s\-_]*feces|k[\s\-_]*i)\b",
    r"\bhvd\b", r"\barc[\s\-_]*trough\b",
    r"\bps[\s\-_]*dif\b", r"\bmtd\b",
]
NIL_PATTERNS_COMPILED = [re.compile(r, re.IGNORECASE) for r in NIL_PATTERNS]

# -----------------------------------------------------------------------------
# Parameter tokenizers and their replacements
# -----------------------------------------------------------------------------

PARAMETER_PATTERNS = {
    # AUCS
    # auc%ext, aucss, auc ratio, auc dose, aucinf covered by general tokenizers
    "auc_patterns": (
        r"\b(?:"
        r"area[\s\-_]*under\s+.*?\s*curve"
        r"|(?:systemic[\s\-_]*)exposure"
        r"|total[\s\-_]*exposure"
        r"|exposure"
        r"|auc\s*to"
        r")\b"
    ),
    "auclast_patterns": (
        r"(?:"
        r"last[\s\-\_]*(measurable|measured)[\s\-\_]*c"
        r"|last[\s\-\_]*(?:obs|meas)"
        r"|0-(?:tend|tlast|last)"
        r"|\ball"
        r"|\btend"
        r"|\btlast"
        r"|\blast"
        r")\b"
    ),
    "auct_patterns": (
        r"\b(?:"
        r"auc[\s\-_]*(\d+(?:\.\d+)?)(?![.\d-])\s*(?:hours|hour|hr|h|days|day|d|minutes|min?)?"  # Matches e.g. 'AUC24', 'AUC24h', 'AUC24 h', etc.
        r"|auc[\s\-\_]*(?:d|day)(\d+)" # matches aucd90
        r"|interval\s+auc"
        r")\b"
    ),
    "aumc_patterns": (
        r"\b(?:"
        r"(?:auc[\s\-_]*)?first[\s\-_]*moment[\s\-_]*(curve)?"
        r")\b"
    ),
    "auc_ratio_patterns": (
        r"\b(?:"
        r"auc[\w–\-\,\s]*\/\s*auc[\w–\-\,\s]*"
        r"|auc[\s\-_]?r"
        r"|auc[\s\-_]?dr"
        r"|relative[\s\-_]?auc"
        r"|auc[\s\-_]?interaction[\s\-_]?ratio"
        r"|parent[\s\-:_]*metabolite[\s\-_]*(?:auc[\s\-_]*)?ratio"
        r")\b"
    ),

    # BIOAVAILABILITIES
    "frel_patterns": (
        r"\b(?:"
        r"relative\s+(?:bioavailability|systemic\s+availability)"
        r")\b"
    ),
    "fg_patterns": (
        r"\b(?:"
        "gut[\s\-_]*wall[\s\-_]*bioavailability"
        r")\b"
    ),
    "fh_patterns": (
        r"\b(?:"
        r"hepatic[\s\-_]*bioavailability"
        r")\b"
    ),
    "fr_patterns": (
         r"\b(?:"
        "renal[\s\-_]*bioavailability"
        r")\b"
    ),
    "fab_patterns": (
      r"\b(?:"
      r"(?:absolute\s+)?(?:oral\s+)?bioavailability"
      r"|percentage\s+absorbed"
      r"|systemic\s+availability"
      r"|f[\s\-_]*(abs|ab|a)"
      r"|f(?!\S)"
      r")\b"
    ),

    # CLEARANCES
    # cl ratio, cl/f covered by general tokenizers
    "cl_patterns": (
            r"\b(?:"
            r"(clearance|cl)[\s\-]*rate"
            r"|(?:total\s+)?clearance"
            r"|elimination[\s\-]*cl"
            r"|cl(total|tot|t)"
            r"|clp" 
            r"|cl(?!\d+-\d+)\d+" # number not range 
            r"|\(\s*cl\s*\)"
            r")\b"
        ),
    "clr_patterns": (
        r"\b(?:"
        r"(renal|urinary)[\s\-]*cl"
        "|cl[,\s\-]*r"
        r")\b"
    ),
    "clh_patterns": (
        r"\b(?:"
        r"hepatic[\s\-]*cl"
        "|cl[,\s\-]*h"
        r")\b"
    ),
    "clb_patterns": (
        r"\b(?:"
        r"blood[\s\-]*cl"
        r"|cl[,\s\-]*b"
        r")\b"
    ),
    "clnr_patterns": (
        r"\b(?:"
        r"nonrenal[\s\-]*cl"
        r"|cl[,\s\-]*nr"
        r")\b"
    ),
    "clu_patterns": (
        r"\b(?:"
        r"unbound[\s\-]*cl"
        r"|cl[,\s\-]*u"
        r")\b"
    ),
    "clint_patterns": (
        r"\b(?:"
        r"intrinsic[\s\-]*metabolic[\s\-]*cl"
        r"|intrinsic[\s\-]*cl"
        r"|enzyme[\s\-]*mediated[\s\-]*cl"
        r"|intrinsic[\s\-]*clint"
        r"|cl[,\s\-]*int"
        r")\b"
    ),
    "cl_f_patterns":(
        r"\b(?:"
        r"cl[\s\-_]*(?:po|(zeta|z))"
        r"|(?:oral|apparent)(?:[\s\-_]*(oral|apparent))?[\s\-_]*cl"
        r")\b"
    ),
    "ae_patterns":(
            r"\b(?:"
            r"cumulative[\s\-_]*excretion[\s\-_]*amount"
            r"|amount[\s\-_]*excreted"
            r")\b"
        ),
    "fe_patterns": (
        r"\b(?:"
        r"percentage[\s\-_]*excreted[\s\-_]*unchanged"
        r"|(cumulative[\s\-_]*)?(urinary|fecal)[\s\-_]*(excretion|excr|recovery(\s*rate)?)"
        r"|renal[\s\-_]*(excretion|excr)"
        r"|(excretion|excr)"
        r")\b"
    ),


    # INTERCOMPARTMENTAL CLEARANCES
    # Q/F and Q2/F covered by general rules & synonyms
    "q_patterns": (
        r"\b(?:"
        r"inter[\s\-]*compartment(?:al)?[\s\-\_]*cl(\s*central\s*peripheral)?"
        r"|q\s*(?:p)?\s*1"
        r"|first[\s\-]*inter[\s\-]*compartment(?:al)?[\s\-\_]*cl"
        r"|distribution(?:al)?[\s\-\_]*cl"
        r")\b"
    ),
    "q2_patterns": (
        r"\b(?:"
        r"q\s*central[\s\-]*second[\s\-]*peripheral"
        r"|q\s*(?:p)?\s*2"
        r"|second[\s\-]*q"
        r")\b"
    ),

    # CONCENTRATIONS
    # cmax_ratio, cmax/dose, css, css_ratio covered by general
    "c_patterns": (
        r"\b(?:"
        r"concentration"
        r"|concn"
        r")\b"
    ),
    "ct_patterns" : (
        r"\b(?:"
        r"c\s?\(?(?![^)]*-)(?!0)[0-9]+\s?(?:hr|h|hours|days|day|minutes|mins|min|')?\)?"  # C24, C30hr, C20mins etc. but not C0 
        r"|day\s*\d+\s*c|c\s*day\s*\d+" #day 3 concentration, concentration day 2 etc.
        r"|c[\s\-]*(d|day)[\s\-]*\d+" # cd90
        r")\b"
    ),
    "cmax_patterns": (
        r"\b(?:"
        r"c[\s\-_]*peak(\d|[a-zA-Z])?"
        r"|c[\s\-_]*max(\d|[a-zA-Z])?"
        r"|peak(\s*c)?"
        r"|max(?:imum|imal)?\s*(?:c|level|peak)" #NB need to be applied before tmax
        r"|cp[\s\-_]*max"
        r")\b"
    ),
    "cmin_patterns": (
        r"\b(?:"
        r"min(?:imum|imal)?\s*c"
        r"|c[\s\-_]*min(\d|[a-zA-Z])?"
        r")\b"
    ),
    "ctrough_patterns": (
        r"\b(?:"
        r"c[\s\-_]*trough"
        r"|c[\s\-_]*pre"
        r"|pre[\s\-_]*level"
        r"|trough(?:[\s\-_]*(c|level))?"
        r")\b"
    ),
    "c0_patterns": (
        r"\b(?:"
        r"c[\s\-_]*0"
        r"|extrapolated[\s\-_]*c"
        r")\b"
    ),
    "cavg_patterns": (
        r"\b(?:"
        r"c[\s\-_]*average"
        r"|c[\s\-_]*(avg|ave|av)(?:[\s\-_,]*\d+)?"
        r"|average[\s\-_]*c"
        r")\b"
    ),

    # HALF-LIVES: complete
    # t1/2_ratio covered by general
    "t1/2_patterns": (
    r"(?:"
    r"\b(?:half[\s\-\_]*(life|lives|live|time|t)|hl)\b"
    r"|t\s*\(*?\s*1\s*/\s*2\s*\)?"
    r"|t[\s\-\_]*0.5"
    r")"
    ),
    "t1/2_alpha_patterns": (
        r"\b(?:"
        r"(?:initial|fast|first|early|distribution|alpha)(?:[\s\-_]*phase)?[\s\-_]*t1/2"
        r"|t1/2[\s\-_]*(?:initial|fast|first|early|distribution|alpha)(?:[\s\-_]*phase)?"
        r"|(distribution)[\s\-_]*t1/2"
        r"|t1/2[\s\-_]*distribution"
        r")\b"
    ),
    "t1/2_beta_patterns": (
        r"\b(?:"
        r"(?:secondary|second|slow|beta|late|latter)(?:[\s\-_]*phase)?[\s\-_]*t1/2"
        r"|t1/2[\s\-_]*(?:secondary|second|slow|beta|late|latter)(?:[\s\-_]*phase)?"
        r"|t1/2[\s\-_]*calculated[\s\-_]*beta[\s\-_]*slope"
        r")\b"
    ),
    "t1/2_gamma_patterns": (
        r"\b(?:"
        r"(?:tertiary|third|gamma)(?:[\s\-_]*phase)?[\s\-_]*t1/2"
        r"|t1/2[\s\-_]*(?:tertiary|third|gamma)(?:[\s\-_]*phase)?"
        r"|t1/2[\s\-_]*calculated[\s\-_]*gamma[\s\-_]*slope"
        r")\b"
    ),
    "t1/2_ka_patterns": (
        r"\b(?:"
        r"absorption(?:[\s\-_]*phase)?[\s\-_]*t1/2"
        r"|t1/2[\s\-_]*absorption(?:[\s\-_]*phase)?"
        r"|t1/2[\s,/-]*(?:abs|ab|a)"
        r"|k[\s\-_]*01[\s\-_]*t1/2"
        r")\b"
    ),
    "t1/2_el_patterns": (
        r"\b(?:"
        r"(apparent|terminal)[\s\-_]*(?:(elimination|disposition|terminal)[\s\-_]*)?t1/2(?:[\s\-_]*(elimination|disposition))?"
        r"|(elimination|disposition|terminal)[\s\-_]*t1/2"
        r"|terminal(?:[\s\-_]*phase)?[\s\-_]*t1/2"
        r"|t1/2[\s\-_]*terminal(?:[\s\-_]*phase)?"
        r"|t1/2(?:[\s,/-]*(?:elimination|el|apparent|app|terminal|λ(zeta|z)|(zeta|z)|ke))"
        r"|k[\s\-_]*10[\s\-_]*t1/2"
        r"|effective[\s\-_]*t1/2"
        r")\b"
    ),
    "z_patterns": (
            r"\b(?:"
            "terminal([\s\-_]*phase)?"
            r")\b"
        ),

    # RATE CONSTANTS
    # λ1, λ2 covered by general
    "k_patterns": (
            r"\b(?:"
            r"rate[\s\-_]*constant"
            r")\b"
        ),
    "kcomp_patterns": (
        r"\b(?:"
        r"(transfer|distribution)\s*(k|rate)\s*(central\s*(?:to\s*)?peripheral|(central\s*(?:to\s*)?second\s*peripheral|peripheral\s*(?:to\s*)?central|second\s*peripheral\s*(?:to\s*)?central)?)"
        r"|(k|rate)\s*(transfer|distribution)\s*(central\s*(?:to\s*)?peripheral|(central\s*(?:to\s*)?second\s*peripheral|peripheral\s*(?:to\s*)?central|second\s*peripheral\s*(?:to\s*)?central)?)"
        r"|k[\s\-_]*(12|21|13|31|24|42|23|32|34|43)\b"
        r")\b"
    ),
    "kabs_patterns": (
        r"\b(?:"
        r"k[\s\-_]*(?:absorption|abs|ab|a|01)"
        r"|(?:first[\s\-_]*order\s*)?absorption[\s\-_]*(k|rate)"
        r"|rate[\s\-_]*absorption"
        r")\b"
    ),
    "kexcr_patterns": (
        r"\b(?:"
        r"(?:urinary[\s\-_]*)?excretion[\s\-_]*(k|rate)"
        r"|(k|rate)[\s\-_]*excr(?:etion)?"
        r")\b"
    ),
    "kmet_patterns": (
        r"\b(?:"
        r"metabolite[\s\-_]*elimination[\s\-_]*(k|rate)"
        r"|k[\s\-_]*met(?:abolite)?"
        r")\b"
    ),
    "kel_patterns": (
        r"\b(?:"
        r"k[\s\-_]*(?:elimination|el|e|10)"
        r"|rate[\s\-_]*elimination"
        r"|elimination[\s\-_]*constant"
        r"|(?:first[\s\-_]*order\s*)?elimination[\s\-_]*(k|rate)"
        r"|(lambda|λ)[\s\-_]*(zeta|z)"
        r"|(lambda|λ)"
        r")\b"
    ),


    # TIMES
    "t_patterns": (
            r"\b(?:"
            r"time(?:[\s\-_]*(to[\s\-_]*reach|to))?"
            r")\b"
        ),
    "tmax_patterns": (
        r"\b(?:"
        r"(?:(max|peak)[\s\-_]*)?t(?:[\s\-_](cmax|c))"
        r"|t[\s\-_]*(?:max|peak)(?:[\s\-_]*c)?"
        r"|t[\s\-_]*cmax"
        r"|t[\s\-_]*(max)"
        r"|(?:max|peak)[\s\-_]*t"
        r")\b"
    ),
    "tlag_patterns": (
        r"\b(?:"
        r"alag"
        r"|absorption\s*lag\s*t"
        r"|lag\s*t"
        r")\b"
    ),
    "mrt_patterns": (
        r"\b(?:"
        r"mean\s*residence\s*t"
        r"|average\s*residence\s*(time|t)"
        r"|mrt0-[a-zA-Z]+"
        r"|mrt[a-zA-Z]+"
        r")\b"
    ),
    "mat_patterns": (
        r"\b(?:"
        r"mean\s*absorption\s*(time|t)"
        r"|average\s*absorption\s*(time|t)"
        r"|mat"
        r")\b"
    ),
    "mtt_patterns": (
        r"\b(?:"
        r"m\s*tt"
        r"|mean\s*transit\s*t"
        r")"
    ),

    # VOLUMES
    # Vc, Vp, Vp2, (/F) and Vss covered by general
    "v_patterns": (
        r"\b(?:"
        r"v[\s\-_]*z"
        r"|v[\s\-_]*d"
        r"|v[\s\-_]*beta"
        r"|v[\s\-_]*gamma"
        r"|distribution\s*volume"
        r"|volume\s*distribution"
        r"|volume"
        r"|v"
        r")\b"
    ),
    "v1_patterns": (
        r"\b(?:"
        r"v[\s\-_]*1"
        r"|v[\s\-_]*c"
        r"|central[\s\-_]*v"
        r"|v[\s\-_]*central"
        r")\b"
    ),
    "v3_patterns": (
        r"\b(?:"
        r"v[\s\-_]*3"
        r"|v[\s\-_]*p[\s\-_]*2"
        r"|second[\s\-_]?peripheral[\s\-_]?v"
        r")\b"
    ),
    "v2_patterns": (
        r"\b(?:"
        r"v[\s\-_]*p(?:1)?"
        r"|v[\s\-_]*2"
        r"|peripheral[\s\-_]*v"
        r")\b"
    ),

    # METABOLISM/EXCRETION
    "km_patterns": (
        r"\b(?:"
        r"michaelis[\s‐\-]*menten[\s‐\-]*(constant)?"
        r")\b"
    ),
    "vmax_patterns": (
        r"\b(?:"
        r"max(?:imum|imal)?[\s\-_]*enzyme[\s\-_]*activity"
        r")\b"
    ),
    "eh_patterns": (
        r"\b(?:"
        r"hepatic[\s\-_]*extraction[\s\-_]*ratio"
        r")\b"
    ),
    "e_patterns":(
        r"\b(?:"
        r"extraction[\s\-_]*ratio"
        r")\b"
    ),
    "fm_patterns": (
        r"\b(?:"
        r"metabolic[\s\-_]*conversion[\s\-_]*percentage"
        r")\b"
    ),
    "fu_patterns": (
        r"\b(?:"
        r"percentage[\s\-_]*unbound"
        r"|percentage[\s\-_]*protein[\s\-_]*unbound"
        r"|free[\s\-_]*percentage"
        r"|protein[\s\-_]*binding[\s\-_]*ratio"
        r"|protein[\s\-_]*binding"
        r")\b"
    ),
    "blood_flow_patterns": (
        r"\b(?:"
        r"blood[\s\-_]*flow"
        r")\b"
    ),
}


PARAMETER_REPLACEMENTS = {
    # concentration
    "c_patterns": "c",
    # aucs
    "auc_patterns": "auc",
    "auc_ratio_patterns": 'auc ratio',
    "auclast_patterns": ' last',
    "auct_patterns": 'auc t',
    "aumc_patterns": 'aumc',

    # bioavailabilities
    "fr_patterns": "fr",
    "fg_patterns": "fg",
    "fh_patterns": "fh",
    "frel_patterns": "frel",
    "fab_patterns": " f ",
    # clearance
    "cl_patterns": "cl",
    "clr_patterns": "clr",
    "clb_patterns": "clb",
    "clnr_patterns": "clnr",
    "clu_patterns": "clu",
    "clint_patterns": "clint",
    "clh_patterns": "clh",
    "cl_f_patterns": "cl bionorm",
    # intercompartmental clearances
    "q_patterns": "q",
    "q2_patterns": "q2",
    # times
    "t_patterns": "t",
    "tmax_patterns": "tmax",
    "tlag_patterns": "tlag",
    "mrt_patterns": "mrt",
    "mtt_patterns": "mtt",
    "mat_patterns": "mat",
    # concentrations
    "ct_patterns": "c t",
    "cmax_patterns": 'cmax',
    "cmin_patterns": 'cmin',
    "ctrough_patterns": 'ctrough',
    "c0_patterns": 'c0',
    "cavg_patterns": 'cavg',
    # volumes
    "v_patterns": "v",
    "v1_patterns": "v1",
    "v3_patterns": "v3",
    "v2_patterns": "v2",
    # half lives
    "t1/2_patterns": 't1/2',
    "t1/2_alpha_patterns": 't1/2 alpha',
    "t1/2_beta_patterns": 't1/2 beta',
    "t1/2_gamma_patterns": 't1/2 gamma',
    "t1/2_ka_patterns": "t1/2,ka",
    "t1/2_el_patterns": "t1/2",
    "z_patterns": "z",
    # Ks
    "k_patterns": " k ",
    "kabs_patterns": " kabs ",
    "kexcr_patterns": " kexcr ",
    "kmet_patterns": " kmet ",
    "kel_patterns": " kel ",
    "kcomp_patterns": " kcomp ",
    # urinary excr
    "ae_patterns": " ae ",
    "fe_patterns": " fe ",
    # metab/excr
    "km_patterns": " km ",
    "vmax_patterns": " vmax ",
    "eh_patterns": " eh ",
    "e_patterns": " e ",
    "fm_patterns": " fm ",
    "fu_patterns": " fu ",
    "blood_flow_patterns": " blood flow ",
}

# -----------------------------------------------------------------------------
# Compile parameter tokenizers
# -----------------------------------------------------------------------------

COMPILED_PARAMETER_PATTERNS = {}
for key, pattern in PARAMETER_PATTERNS.items():
    COMPILED_PARAMETER_PATTERNS[key] = re.compile(pattern, re.IGNORECASE)

# Use the order from PARAMETER_REPLACEMENTS
ORDERED_PARAMETER_REPLACEMENTS = []
for key in PARAMETER_REPLACEMENTS:
    if key in COMPILED_PARAMETER_PATTERNS:
        ORDERED_PARAMETER_REPLACEMENTS.append(
            (COMPILED_PARAMETER_PATTERNS[key], PARAMETER_REPLACEMENTS[key])
        )



