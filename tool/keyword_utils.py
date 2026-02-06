"""Utility functions for keyword extraction and loading."""

from pathlib import Path
from typing import TYPE_CHECKING, Set, Dict

from askany.config import settings

if TYPE_CHECKING:
    from askany.ingest.vector_store import VectorStoreManager


def export_keywords_to_word_freq(
    vector_store_manager: "VectorStoreManager",
) -> None:
    """Export keywords and frequencies from FAQ and Docs keyword indices to word_freq.txt files.

    Loads keywords from both faq_keyword_storage_index and docs_keyword_storage_index,
    extracts keywords and their frequencies (based on number of node mappings),
    and writes them to word_freq.txt files in the respective directories.

    Format: Each line contains "keyword frequency"

    Args:
        vector_store_manager: VectorStoreManager instance with loaded keyword indices
    """
    # Process FAQ keyword index
    faq_keyword_index = vector_store_manager.get_faq_keyword_index()
    if faq_keyword_index is not None:
        faq_persist_dir = (
            Path(settings.storage_dir) / settings.faq_keyword_storage_index
        )
        faq_word_freq_file = faq_persist_dir / "word_freq.txt"

        try:
            index_struct = faq_keyword_index.index_struct
            if hasattr(index_struct, "table") and index_struct.table:
                keyword_table = index_struct.table

                # Extract keywords and frequencies
                keyword_freq_pairs = []
                for keyword, node_ids in keyword_table.items():
                    # Convert to list if needed
                    if isinstance(node_ids, set):
                        node_ids = list(node_ids)
                    elif not isinstance(node_ids, (list, tuple)):
                        node_ids = list(node_ids) if node_ids else []

                    # Frequency is the number of nodes this keyword maps to
                    frequency = len(node_ids)
                    keyword_freq_pairs.append((keyword, frequency))

                # Sort by frequency (descending), then by keyword (ascending)
                keyword_freq_pairs.sort(key=lambda x: (-x[1], x[0]))

                # Write to file
                faq_persist_dir.mkdir(parents=True, exist_ok=True)
                with open(faq_word_freq_file, "w", encoding="utf-8") as f:
                    for keyword, frequency in keyword_freq_pairs:
                        f.write(f"{keyword} {frequency}\n")

                print(
                    f"✅ Exported {len(keyword_freq_pairs)} keywords from FAQ keyword index to {faq_word_freq_file}"
                )
            else:
                print("⚠️  FAQ keyword table structure not found or empty")
        except Exception as e:
            print(f"❌ Error exporting FAQ keywords: {e}")
    else:
        print("⚠️  FAQ keyword index not found")

    # Process Docs keyword index
    docs_keyword_index = vector_store_manager.get_docs_keyword_index()
    if docs_keyword_index is not None:
        docs_persist_dir = (
            Path(settings.storage_dir) / settings.docs_keyword_storage_index
        )
        docs_word_freq_file = docs_persist_dir / "word_freq.txt"

        try:
            index_struct = docs_keyword_index.index_struct
            if hasattr(index_struct, "table") and index_struct.table:
                keyword_table = index_struct.table

                # Extract keywords and frequencies
                keyword_freq_pairs = []
                for keyword, node_ids in keyword_table.items():
                    # Convert to list if needed
                    if isinstance(node_ids, set):
                        node_ids = list(node_ids)
                    elif not isinstance(node_ids, (list, tuple)):
                        node_ids = list(node_ids) if node_ids else []

                    # Frequency is the number of nodes this keyword maps to
                    frequency = len(node_ids)
                    keyword_freq_pairs.append((keyword, frequency))

                # Sort by frequency (descending), then by keyword (ascending)
                keyword_freq_pairs.sort(key=lambda x: (-x[1], x[0]))

                # Write to file
                docs_persist_dir.mkdir(parents=True, exist_ok=True)
                with open(docs_word_freq_file, "w", encoding="utf-8") as f:
                    for keyword, frequency in keyword_freq_pairs:
                        f.write(f"{keyword} {frequency}\n")

                print(
                    f"✅ Exported {len(keyword_freq_pairs)} keywords from Docs keyword index to {docs_word_freq_file}"
                )
            else:
                print("⚠️  Docs keyword table structure not found or empty")
        except Exception as e:
            print(f"❌ Error exporting Docs keywords: {e}")
    else:
        print("⚠️  Docs keyword index not found")


def load_keywords_from_txt(txt_file: str) -> Set[str]:
    """Load keywords from a txt file, ignoring frequencies.

    Reads a txt file where each line contains "keyword frequency" or just "keyword",
    extracts only the keywords (ignoring frequencies), and returns them as a set.
    Handles multi-word keywords (e.g., "block scanner 2" -> keyword is "block scanner").

    Args:
        txt_file: Path to the txt file containing keywords

    Returns:
        Set of keywords (without frequencies)
    """
    keywords = set()
    txt_path = Path(txt_file)

    if not txt_path.exists():
        print(f"⚠️  File not found: {txt_file}")
        return keywords

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split by whitespace
                parts = line.split()
                if not parts:
                    continue

                # If last part is numeric, it's the frequency; otherwise it's part of the keyword
                # Take everything except the last part if the last part is numeric
                if len(parts) > 1 and parts[-1].isdigit():
                    keyword = " ".join(parts[:-1])
                else:
                    keyword = " ".join(parts)
                
                if keyword:
                    keywords.add(keyword)

        print(f"✅ Loaded {len(keywords)} keywords from {txt_file}")
        return keywords

    except Exception as e:
        print(f"❌ Error loading keywords from {txt_file}: {e}")
        return keywords

def load_keywords_and_frequency_from_txt(txt_file: str) -> Dict[str, int]:
    """Load keywords and frequencies from a txt file.
    
    Handles multi-word keywords (e.g., "block scanner 2" -> keyword is "block scanner", frequency is 2).
    The frequency is always the last token on the line if it's numeric.
    
    Args:
        txt_file: Path to the txt file containing keywords and frequencies.
        
    Returns:
        Dictionary of keywords and frequencies.
    """
    keywords_and_frequency = {}
    txt_path = Path(txt_file)
    
    if not txt_path.exists():
        print(f"⚠️  File not found: {txt_file}")
        return keywords_and_frequency
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                
                # If last part is numeric, it's the frequency; otherwise keyword has no frequency
                if len(parts) > 1 and parts[-1].isdigit():
                    keyword = " ".join(parts[:-1])
                    frequency = int(parts[-1])
                else:
                    # No frequency provided, default to 0
                    keyword = " ".join(parts)
                    frequency = 0
                
                if keyword:
                    if keyword not in keywords_and_frequency:
                        keywords_and_frequency[keyword] = frequency
                    else:
                        keywords_and_frequency[keyword] += frequency
        
        print(f"✅ Loaded {len(keywords_and_frequency)} keywords and frequencies from {txt_file}")
        return keywords_and_frequency
    
    except Exception as e:
        print(f"❌ Error loading keywords from {txt_file}: {e}")
        return keywords_and_frequency
