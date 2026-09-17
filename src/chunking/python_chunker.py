import ast


def line_offsets(source: str) -> list[int]:
    """
    Get the number of each line of source
    """
    lines: list[str] = source.splitlines(keepends=True)
    offsets: list[int] = []
    count = 0
    for line in lines:
        offsets.append(count)
        count += len(line)
    return offsets


def split_raw(
        start: int,
        end: int,
        max_chunk_size: int
) -> list[tuple[int, int]]:
    """
    Handle the case when the chunk bigger than max_chunk_size
    """
    chunks: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        chunk_end = min(pos + max_chunk_size, end)
        chunks.append((pos, chunk_end))
        pos = chunk_end

    return chunks


def chunk_node(
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        offsets: list[int],
        max_chunk_size: int
) -> list[tuple[int, int]]:
    """
    Check the chunk size and returns the appropriate value
    """
    if node.end_lineno is None or node.end_col_offset is None:
        return []

    start = offsets[node.lineno - 1] + node.col_offset
    end = offsets[node.end_lineno - 1] + node.end_col_offset

    # if good size we return start and end
    if end - start <= max_chunk_size:
        return [(start, end)]

    # if its a class, dont break it and take the func one by one
    if isinstance(node, ast.ClassDef):
        result: list[tuple[int, int]] = []
        for res in node.body:
            if isinstance(res, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.extend(chunk_node(res, offsets, max_chunk_size))

        return result

    # if method too big, manual chunk
    return split_raw(start, end, max_chunk_size)


def chunk_python(
        source: str,
        max_chunk_size: int = 2000
) -> list[tuple[int, int]]:
    """
    Create a list of chunks
    """
    tree = ast.parse(source)
    offsets = line_offsets(source)
    chunks: list[tuple[int, int]] = []

    for node in tree.body:
        if isinstance(
                node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            chunks.extend(chunk_node(node, offsets, max_chunk_size))

    return chunks


# if __name__ == "__main__":
#     example = (
#         "import os\n"
#         "\n"
#         "def add(a, b):\n"
#         "    \"\"\"Ajoute deux nombres.\"\"\"\n"
#         "    return a + b\n"
#         "\n"
#         "class Greeter:\n"
#         "    def hello(self):\n"
#         "        return \"hi\"\n"
#     )
#     chunks = chunk_python(example)
#     for start, end in chunks:
#         print(repr(example[start:end]))
