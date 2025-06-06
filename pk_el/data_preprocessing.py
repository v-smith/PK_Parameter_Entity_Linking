import re
from dataclasses import dataclass
from typing import Dict, List

from bs4 import BeautifulSoup
from tqdm import tqdm


@dataclass
class PreprocessingConfig:
    special_tokens: bool = True
    window_size: int = 0  # How many words before/after mention
    include_combined_context: bool = True  # Include footer/caption
    mention_tokens: tuple = ("[MENTION]", "[/MENTION]")  # Custom special tokens
    truncate: bool = True
    max_items: int = 10


TEXT_DEFAULT_CONFIG = PreprocessingConfig(
    special_tokens=True,
    window_size=5,
    include_combined_context=False
)

TABLE_DEFAULT_CONFIG = PreprocessingConfig(
    special_tokens=True,
    window_size=0,
    include_combined_context=True,
    truncate=True,
    max_items=10,
)

# --- Table Parsing and Context Extraction --- #

def is_index_sequence(values: List[str], allow_leading_blank: bool = False) -> bool:
    """Check if a list is a 0-based increasing integer sequence, optionally allowing an initial empty string."""
    if allow_leading_blank and values and values[0].strip() == "":
        values = values[1:]  # Strip blank header corner
    if not values:
        return False
    return all(cell.isdigit() for cell in values) and list(map(int, values)) == list(range(len(values)))


def parse_html_table(html_table: str) -> Dict[str, any]:
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    if not table or not rows:
        return {"markdown": "", "table_data": []}

    # --- Parse all rows ---
    parsed_rows = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        parsed_rows.append([cell.get_text(strip=True) for cell in cells])

    # --- Detect and remove index row ---
    if is_index_sequence(parsed_rows[0], allow_leading_blank=True):
        parsed_rows = parsed_rows[1:]

    # --- Detect and remove index column ---
    first_col = [row[0] for row in parsed_rows if len(row) > 0]
    if is_index_sequence(first_col):
        parsed_rows = [row[1:] for row in parsed_rows]

    # --- Markdown ---
    markdown_lines = []
    if len(parsed_rows) > 0:
        markdown_lines.append("| " + " | ".join(parsed_rows[0]) + " |")
        #markdown_lines.append("|" + " --- |" * len(parsed_rows[0]))
        for row in parsed_rows[1:]:
            markdown_lines.append("| " + " | ".join(row) + " |")
    else:
        markdown_lines.append("")

    return {
        "markdown": "\n".join(markdown_lines),
        "table_data": parsed_rows,
    }


def extract_context_from_table(
    parsed_table: Dict,
    example: Dict,
    window: int = 3
) -> Dict[str, str]:
    """
    Extracts row and column context around a target cell, including up to `window` cells
    on either side of the target index. Highlights the target cell using [MENTION] tags.
    """
    full_table  = parsed_table["table_data"]
    total_rows = len(full_table)
    #mention = example["mention"]
    text_with_tagged_mention = example["text_with_tagged_mention"]
    #text = example["text"]
    row_idx = example.get("row_idx", -1)
    col_idx = example.get("col_idx", -1)
    target_cell = full_table[row_idx][col_idx]
    #print(f"Target cell: {target_cell}, text: {text}")

    # --- Row context ---
    row = full_table[row_idx] if 0 <= row_idx < total_rows else []
    row_start = max(0, col_idx - window)
    row_end = col_idx + window + 1
    row_context = []
    for i in range(row_start, row_end):
        if i < len(row):
            cell = row[i]
            if i == col_idx:
                cell = text_with_tagged_mention
            row_context.append(cell)

    # --- Column context ---
    col_context = []
    row_range_start = max(0, row_idx - window)
    row_range_end = row_idx + window + 1
    for i in range(row_range_start, row_range_end):
        if i < total_rows and col_idx < len(full_table[i]):
            cell = full_table[i][col_idx]
            if i == row_idx:
                cell = text_with_tagged_mention
            col_context.append(cell)

    return {
        "row_context": "| " + " | ".join(row_context) + " |" if row_context else "",
        "column_context": "| " + " | ".join(col_context) + " |" if col_context else "",
    }


# --- Text Preprocessing --- #

def insert_special_tokens(text, start, end, open_token='[MENTION]', close_token='[/MENTION]'):
    return text[:start] + f"{open_token}{text[start:end]}{close_token}" + text[end:]

def remove_special_tokens(text, open_token='[MENTION]', close_token='[/MENTION]'):
    """Remove special open/close tokens but keep the inner text intact."""
    text = text.replace(open_token, '')
    text = text.replace(close_token, '')
    return text

def get_text_mention_feature(item, special_tokens=True):
    text = item["text"]
    span_start = item["spans"][0]["start"]
    span_end = item["spans"][0]["end"]
    if special_tokens:
        return insert_special_tokens(text, span_start, span_end)
    else:
        return text[span_start:span_end]

def get_text_mention_feature_window(item, special_tokens=True, window_size=0, start_token="[MENTION]", end_token="[/MENTION]"):
    text = item["text"]
    span_start = item["spans"][0]["start"]
    span_end = item["spans"][0]["end"]
    mention = text[span_start:span_end]

    if window_size == 0:
        return f"{start_token} {mention} {end_token}" if special_tokens else mention

    tokens = re.findall(r'\S+|\n', text)
    token_spans = []
    offset = 0
    for token in tokens:
        start = text.find(token, offset)
        token_spans.append((start, start + len(token)))
        offset = start + len(token)

    mention_start_token = mention_end_token = None
    for i, (start, end) in enumerate(token_spans):
        if start <= span_start < end:
            mention_start_token = i
        if start < span_end <= end:
            mention_end_token = i
            break

    if mention_start_token is None or mention_end_token is None:
        raise ValueError("Could not align span with tokens.")

    start_context = max(0, mention_start_token - window_size)
    end_context = min(len(tokens), mention_end_token + 1 + window_size)

    before = tokens[start_context:mention_start_token]
    after = tokens[mention_end_token + 1:end_context]
    mention_tokens = tokens[mention_start_token:mention_end_token + 1]

    if special_tokens:
        mention_tokens = [start_token] + mention_tokens + [end_token]

    return " ".join(before + mention_tokens + after)

# --- Table Cell Context Features ---

def get_cell_headers_feature(sample, parsed_table=None):
    """
    Add [ROW-HEADER] and [COL-HEADER] based on full table including headers.
    """
    if parsed_table is None:
        parsed_table = parse_html_table(sample["table_html"])

    full_table = parsed_table["table_data"]
    row_idx = sample["row"]
    col_idx = sample["col"]
    cell_text = sample["text_with_tagged_mention"]

    components = [f"[TABLE CELL] {cell_text}"]

    if row_idx < len(full_table) and col_idx < len(full_table[row_idx]):
        row_header = full_table[row_idx][0] if full_table[row_idx] else ""
        col_header = full_table[0][col_idx] if full_table and col_idx < len(full_table[0]) else ""

        if col_idx != 0 and row_header:
            components.append(f"[ROW-HEADER] {row_header}")
        if row_idx != 0 and col_header:
            components.append(f"[COL-HEADER] {col_header}")

    return " ".join(components)


def get_cell_row_col_feature(sample, parsed_table=None):
    """
    Combine full [ROW] and [COL] context, treating headers as row 0, col 0.
    """
    if parsed_table is None:
        parsed_table = parse_html_table(sample["table_html"])

    full_table = parsed_table["table_data"]
    row_idx = sample.get("row_idx", -1)
    col_idx = sample.get("col_idx", -1)
    cell_text = sample["text_with_tagged_mention"]

    row_content = full_table[row_idx] if 0 <= row_idx < len(full_table) else []
    col_content = [r[col_idx] for r in full_table if col_idx < len(r)]

    components = [f"[CELL] {cell_text}"]
    if row_content:
        components.append(f"[ROW] {' | '.join(row_content)}")
    if col_content:
        components.append(f"[COL] {' | '.join(col_content)}")

    return " ".join(components)

def get_cell_footer_caption_feature(sample):
    cell_text = sample["text_with_tagged_mention"]
    caption = sample.get("caption")
    footer = sample.get("footer")

    parts = [f"[CELL] {cell_text}"]
    if caption:
        parts.append(f"[CAPTION] {caption.strip()}")
    if footer:
        parts.append(f"[FOOTER] {footer.strip()}")
    return " ".join(parts)

def get_cell_full_context_feature(sample, parsed_table=None):
    row_col_context = get_cell_row_col_feature(sample, parsed_table=parsed_table)
    caption = sample.get("caption")
    footer = sample.get("footer")

    parts = [row_col_context]
    if caption:
        parts.append(f"[CAPTION] {caption.strip()}")
    if footer:
        parts.append(f"[FOOTER] {footer.strip()}")
    return " ".join(parts)

# --- Preprocessing Pipelines --- #

def prep_text_features(data, config: PreprocessingConfig):
    for sample in tqdm(data):
        sample["mention"] = get_text_mention_feature(
            sample,
            special_tokens=False
        )
        sample["mention_with_window"] = get_text_mention_feature_window(
            sample,
            special_tokens=config.special_tokens,
            window_size=config.window_size,
            start_token=config.mention_tokens[0],
            end_token=config.mention_tokens[1]
        )

        sample["text_with_tagged_mention"] = get_text_mention_feature(
            sample,
            special_tokens=True
        )
        sample["text_context_llm"] = format_text_context_for_llm(sample)
    return data

def prep_table_features(data, config: PreprocessingConfig):
    for sample in tqdm(data):
        parsed_table = parse_html_table(sample["table_html"])

        sample["mention"] = get_text_mention_feature(
            sample,
            special_tokens=False
        )

        sample["text_with_tagged_mention"] = get_text_mention_feature(
            sample,
            special_tokens=True
        )

        context_parts = extract_context_from_table(parsed_table, sample)
        sample["row_context"] = context_parts["row_context"]
        sample["col_context"] = context_parts["column_context"]
        sample["table_context_retrieval"] = format_table_context_for_retrieval(sample)
        sample["table_context_llm"] = format_table_context_for_llm(sample)
    return data


def format_table_context_for_llm(sample: dict) -> str:
    """
    Formats table context for LLM input from a preprocessed sample.

    Args:
        sample (dict): A dictionary containing preprocessed table features.

    Returns:
        str: Formatted string ready for LLM input.
    """
    parts = ["The following context is provided to help you. \n It shows the table row and column from which the mention is derived (with mention tagged) and the table footer, if available."]

    # Include the row context
    row_context = sample.get("row_context", "")
    if row_context:
        parts.append(f"[ROW] {row_context.strip()}")

    # Always include the column context
    col_context = sample.get("col_context", "")
    if col_context:
        parts.append(f"[COLUMN] {col_context.strip()}")

    # Include footer if available
    footer = sample.get("footer", "")
    if footer:
        parts.append(f"[FOOTER] {footer.strip()}")

    """caption = sample.get("caption", "")
    if caption:
        caption = truncate_by_words(caption.strip(), max_words=20)
        parts.append(f"[CAPTION] {caption}")"""

    return "\n".join(parts)


def format_table_context_for_retrieval(sample: dict) -> str:
    """
    Formats table context for LLM input from a preprocessed sample.

    Args:
        sample (dict): A dictionary containing preprocessed table features.

    Returns:
        str: Formatted string ready for LLM input.
    """
    parts = []
    # Include mention
    parts.append(sample["text_with_tagged_mention"])

    # Include the row context
    row_context = sample.get("row_context", "")
    if row_context:
        parts.append(f"[ROW] {row_context.strip()}")

    # Always include the column context
    col_context = sample.get("col_context", "")
    if col_context:
        parts.append(f"[COLUMN] {col_context.strip()}")

    return "\n".join(parts)


def format_text_context_for_llm(sample: dict) -> str:
    """
    Formats table context for LLM input from a preprocessed sample.

    Args:
        sample (dict): A dictionary containing preprocessed table features.

    Returns:
        str: Formatted string ready for LLM input.
    """
    parts = []
    parts.append(f"The following context is provided to help you. It shows the sentence from which the mention is derived with the mention tagged.")

    # Include Context
    parts.append(f"{sample['text_with_tagged_mention']}")

    return "\n".join(parts)