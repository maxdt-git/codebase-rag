import re
from src.chunking.python_chunker import split_raw


def find_headers(text: str) -> list[tuple[int, str]]:
    """
    Find the title pattern and return a list of found positions
    """
    result: list[tuple[int, str]] = []
    pattern = r'^#{1,6}\s.*$'
    # each match is one markdown title line, record where it starts
    for find in re.finditer(pattern, text, re.MULTILINE):
        result.append((find.start(), find.group().strip()))

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
        max_chunk_size: int,
        header_title: str
) -> list[tuple[int, int, str]]:
    """
    Check the chunk size of the paragraph and returns the appropriate value
    """
    # if good size we return start and end
    if end - start <= max_chunk_size:
        return [(start, end, "")]

    # section too big: fall back to splitting it into paragraphs
    pos = paragraph_starts(text, start, end)
    limits = pos + [end]
    # drop any zero-length pair (e.g. a blank line right at the boundary)
    sections = [(limits[i], limits[i + 1]) for i in range(len(limits) - 1)
                if limits[i] < limits[i + 1]]

    result: list[tuple[int, int, str]] = []

    # if a paragraph is still too big on its own, raw split it
    for index, (pstart, pend) in enumerate(sections):
        context = header_title + "\n" if header_title and index > 0 else ""
        if pend - pstart > max_chunk_size:
            result.extend((s, e, context) for s, e in split_raw(
                pstart, pend, max_chunk_size))
        else:
            result.append((pstart, pend, context))

    return result


def chunk_markdown(
        text: str,
        max_chunk_size: int = 2000
) -> list[tuple[int, int, str]]:
    """
    Create the different sections in a list of (start, end, context)
    """
    headers = find_headers(text)
    # keep a leading section for any text before the first title
    if headers and headers[0][0] == 0:
        starts = headers
    else:
        starts = [(0, "")] + headers

    positions = [pos for pos, _ in starts] + [len(text)]
    titles = [title for _, title in starts]

    result: list[tuple[int, int, str]] = []
    # chunk each section on its own, splitting it further if too big
    for i in range(len(positions) - 1):
        result.extend(
            chunk_section(
                text, positions[i], positions[i + 1],
                max_chunk_size, titles[i]))

    return result


# if __name__ == "__main__":
#     print(find_headers(
#         "# Introduction\nSome intro text.\n\n## Setup\nSteps to install.\n"))
#     print(chunk_markdown(
#         "# Introduction\nSome intro text.\n\n## Setup\nSteps to install.\n"))
