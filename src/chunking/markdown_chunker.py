import re
from src.chunking.python_chunker import split_raw


def find_headers(text: str) -> list[int]:
    """
    Find the title pattern and return a list of found positions
    """
    result: list[int] = []
    pattern = r'^#{1,6}\s.*$'
    # each match is one markdown title line, record where it starts
    for find in re.finditer(pattern, text, re.MULTILINE):
        result.append(find.start())

    return result


def paragraph_starts(
        text: str,
        start: int,
        end: int
) -> list[int]:
    """
    Find the start positions of the paragraphs
    """
    pos = [start]
    # a new paragraph starts right after each blank-line separator
    for find in re.finditer(r'\n\s*\n', text[start:end]):
        pos.append(start + find.end())
    return pos


def chunk_section(
        text: str,
        start: int,
        end: int,
        max_chunk_size: int
) -> list[tuple[int, int]]:
    """
    Check the chunk size of the paragraph and returns the appropriate value
    """
    # if good size we return start and end
    if end - start <= max_chunk_size:
        return [(start, end)]

    # section too big: fall back to splitting it into paragraphs
    pos = paragraph_starts(text, start, end)
    limits = pos + [end]
    # drop any zero-length pair (e.g. a blank line right at the boundary)
    sections = [(limits[i], limits[i + 1]) for i in range(len(limits) - 1)
                if limits[i] < limits[i + 1]]

    result: list[tuple[int, int]] = []

    # if a paragraph is still too big on its own, raw split it
    for pstart, pend in sections:
        if pend - pstart > max_chunk_size:
            result.extend(split_raw(pstart, pend, max_chunk_size))
        else:
            result.append((pstart, pend))

    return result


def chunk_markdown(
        text: str,
        max_chunk_size: int = 2000
) -> list[tuple[int, int]]:
    """
    Create the different sections in a list of (start, end)
    """
    header_positions = find_headers(text)
    # keep a leading section for any text before the first title
    if header_positions and header_positions[0] == 0:
        starts = header_positions
    else:
        starts = [0] + header_positions

    limits = starts + [len(text)]
    sections = [(limits[i], limits[i + 1]) for i in range(len(limits) - 1)]

    result: list[tuple[int, int]] = []
    # chunk each section on its own, splitting it further if too big
    for section_start, section_end in sections:
        result.extend(
            chunk_section(text, section_start, section_end, max_chunk_size))

    return result


# if __name__ == "__main__":
#     print(find_headers(
#         "# Introduction\nSome intro text.\n\n## Setup\nSteps to install.\n"))
#     print(chunk_markdown(
#         "# Introduction\nSome intro text.\n\n## Setup\nSteps to install.\n"))
