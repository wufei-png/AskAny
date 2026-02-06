"""Query parser for extracting metadata filters from queries."""

import re
from typing import Dict, Optional, Tuple


def parse_query_filters(query: Optional[str]) -> Tuple[str, Dict[str, str]]:
    """Parse query to extract metadata filters and clean query text.

    Supports filter syntax: @tag key=value or @tag key="value with spaces"

    Examples:
        - "如何安装 @tag hardware=nv_t4" -> ("如何安装", {"hardware": "nv_t4"})
        - '@tag hardware="NVIDIA T4"' -> ("", {"hardware": "NVIDIA T4"})
        - "问题 @tag lang=zh-CN @tag hardware=nv_t4" -> ("问题", {"lang": "zh-CN", "hardware": "nv_t4"})

    Args:
        query: User query string that may contain @tag filters

    Returns:
        Tuple of (cleaned_query, filters_dict)
        - cleaned_query: Query text with @tag filters removed
        - filters_dict: Dictionary of metadata filters {key: value}
    """
    # Handle None or empty query
    if not query:
        return "", {}

    # Pattern to match @tag key=value or @tag key="value with spaces"
    # Matches: @tag key=value or @tag key="quoted value"
    pattern = r'@tag\s+(\w+)=(?:"([^"]+)"|([^\s]+))'

    filters = {}
    matches = re.finditer(pattern, query)

    for match in matches:
        key = match.group(1)
        # Group 2 is quoted value, group 3 is unquoted value
        value = match.group(2) if match.group(2) is not None else match.group(3)
        filters[key] = value

    # Remove all @tag filters from query
    cleaned_query = re.sub(pattern, "", query)
    # Clean up extra whitespace
    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

    return cleaned_query, filters
