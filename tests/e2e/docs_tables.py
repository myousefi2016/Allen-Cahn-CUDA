#!/usr/bin/env python3
"""Every table of the documentation keeps its columns, as GitHub's Markdown renders it.

GitHub's Markdown (GFM) starts a new table cell at every '|' that no backslash precedes, also
inside a code span. It drops the cells of a row beyond its header's, pads a shorter row with
empty cells, and makes no table at all of a header whose delimiter row has another number of
cells. A `max|Δφ|` in a two-column row of README.md cut that row's description short, and a
code span `|y·Lx − x·Ly|` in a header of docs/DESIGN.md left its table unrendered.

cmark-gfm, GitHub's implementation of GFM, finds the tables (its XML with source positions,
table extension), and this test fails for
  * every table row whose line has another number of cells than its header: the line's cells
    are counted as cmark-gfm's row_from_string counts them (extensions/table.c, the scanners
    of extensions/ext_scanners.re), and the count is checked against the cells cmark-gfm
    kept, which are the line's up to the header's number;
  * every line of a paragraph's text, or of a setext heading's (a paragraph's text with a
    line of '=' or '-' under it), that is a delimiter row once its leading spaces, tabs and
    '>' are removed: no table was made of it. A table whose delimiter row has another number
    of cells than its header, directly followed by such a line, is a setext heading.
A row's line keeps its cells after its leading spaces, tabs and block-quote markers, and its
first cell never starts with '>' (that line would open a block quote). It reads README.md,
docs/*.md, ci/**/*.md and .github/**/*.md; README.md and the three documents of docs/ must be
present.

No GPU; Python 3 standard library and cmark-gfm on PATH (Debian and Ubuntu package cmark-gfm).

Usage: docs_tables.py REPO_ROOT   (exit 0 = every table keeps its columns)
"""

import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REQUIRED = ["README.md", "docs/THEORY.md", "docs/DESIGN.md", "docs/ARCHITECTURE.md"]
PATTERNS = ["README.md", "docs/*.md", "ci/**/*.md", ".github/**/*.md"]
NS = "{http://commonmark.org/xml/1.0}"
# ext_scanners.re: spacechar, and the delimiter ("marker") row of _scan_table_start without
# its newline.
SPACECHAR = " \t\v\f"
MARKER = r"[ \t\v\f]*:?-+:?[ \t\v\f]*"
DELIMITER_ROW = re.compile(rf"\|?{MARKER}(?:\|{MARKER})*\|?[ \t\v\f]*")
# cmark's line endings (blocks.c S_parser_feed); str.splitlines would also split at \v, \f, ...
LINE_END = re.compile(rb"\r\n|\r|\n")
XML_INVALID = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# The opening of an ATX heading (CommonMark 4.2) at the heading's first column.
ATX = re.compile(rb"#{1,6}(?:[ \t]|$)")


def cell_end(s, i):
    """_scan_table_cell_end: '|' spacechar*."""
    if i < len(s) and s[i] == "|":
        j = i + 1
        while j < len(s) and s[j] in SPACECHAR:
            j += 1
        return j - i
    return 0


def cell(s, i):
    """_scan_table_cell, the longest match of (escaped_char | [^|\\r\\n])+ with escaped_char
    a backslash and an ASCII punctuation character: any character but '|' and the line
    endings, and a '|' right after a backslash (that backslash is a character of its own or
    escapes the '|')."""
    j = i
    while j < len(s) and s[j] not in "\r\n" and (s[j] != "|" or (j > i and s[j - 1] == "\\")):
        j += 1
    return j - i


def row_end(s, i):
    """_scan_table_row_end: spacechar* newline."""
    j = i
    while j < len(s) and s[j] in SPACECHAR:
        j += 1
    if s.startswith("\r\n", j):
        return j + 2 - i
    if j < len(s) and s[j] == "\n":
        return j + 1 - i
    return 0


def row_cells(line):
    """The number of cells of one line (at its first non-space character, with its newline)
    as table.c's row_from_string counts them; None when the line is not a row."""
    n, offset, more = 0, cell_end(line, 0), True
    while offset < len(line) and more:
        c = cell(line, offset)
        p = cell_end(line, offset + c)
        if c or p:
            n += 1
        offset += c + p
        if p:
            more = True
        else:
            r = row_end(line, offset)
            offset += r
            if r and offset != len(line):  # more lines: the row starts again after them
                n, offset, more = 0, offset + cell_end(line, offset), True
            else:
                more = False
    return n if offset == len(line) and n else None


def cells(n, kind=""):
    return f"{n} {kind}cell" + ("" if n == 1 else "s")


def sourcepos(node):
    m = re.fullmatch(r"(\d+):(\d+)-(\d+):(\d+)", node.get("sourcepos", ""))
    if not m:
        raise SystemExit(f"FAIL cmark-gfm gave {node.tag} no source position")
    return tuple(int(g) for g in m.groups())


def parse(cmark, data):
    """cmark-gfm's XML of the Markdown bytes `data`: (root element, None) or (None, its error)."""
    out = subprocess.run([cmark, "--to", "xml", "--sourcepos", "-e", "table"], input=data,
                         capture_output=True, timeout=60)
    if out.returncode != 0:
        return None, f"exited {out.returncode}: {out.stderr.decode().strip()}"
    # cmark-gfm's XML carries the text's control characters (U+000B, U+000C, ...), which XML
    # 1.0 does not allow; only the elements and their source positions are read here.
    return ET.fromstring(XML_INVALID.sub("\ufffd".encode(), out.stdout)), None


def setext_underline(cmark, lines, first, column):
    """The line number of the underline of the setext heading that starts at (first, column),
    or None. cmark-gfm's end position of a setext heading can lie past its underline (on the
    next line or block), so the underline is the line from which a parse of the file's first
    lines has this heading: block structure depends on no later line."""
    for m in range(first + 1, len(lines) + 1):
        root, _ = parse(cmark, b"\n".join(lines[:m]) + b"\n")
        if root is not None and any(sourcepos(h)[:2] == (first, column)
                                    for h in root.iter(NS + "heading")):
            return m
    return None


def check(cmark, path, rel, errors):
    """Returns the numbers of tables and rows of the file."""
    # cmark-gfm gets the file as it is. It skips one UTF-8 byte order mark at the start by
    # moving past it, so the columns of line 1 count its 3 bytes, and it reads a NUL as
    # U+FFFD, whose 3 bytes the columns count (blocks.c; columns count bytes). The lines here
    # are read alike, NUL as U+FFFD and the byte order mark kept, so a column points at the
    # same byte; a table row, read whole, is never line 1 (its header and delimiter row come
    # first).
    data = path.read_bytes()
    lines = LINE_END.split(data.replace(b"\0", "\ufffd".encode()))
    root, failure = parse(cmark, data)
    if root is None:
        errors.append(f"cmark-gfm {failure} on {rel}")
        return 0, 0

    tables = rows = 0
    # Lines of text to check: (line number, column of its text or None after the first line,
    # "paragraph" or "setext heading").
    text_lines = []
    for table in root.iter(NS + "table"):
        tables += 1
        start, column, end, _ = sourcepos(table)
        header = table.find(NS + "table_header")
        columns = len(header.findall(NS + "table_cell"))
        table_rows = table.findall(NS + "table_row")
        for row in table_rows:
            rows += 1
            number = sourcepos(row)[0]
            text = lines[number - 1].decode("utf-8").lstrip(" \t>")
            n = row_cells(text + "\n")
            kept = sum(1 for c in row.findall(NS + "table_cell") if sourcepos(c)[1] != 0)
            if n is None or kept != min(n, columns):
                errors.append(f"{rel}:{number}: cmark-gfm kept {kept} cells of this row, this "
                              f"test counts {n} against a header of {columns}: the test's "
                              f"count is not cmark-gfm's")
            elif n > columns:
                errors.append(f"{rel}:{number}: row has {cells(n)}, its header {columns}: GFM drops "
                              f"its last {cells(n - columns)} (an unescaped '|'?)")
            elif n < columns:
                errors.append(f"{rel}:{number}: row has {cells(n)}, its header {columns}: GFM adds "
                              f"{cells(columns - n, 'empty ')} (text right below the table?)")
        # The header is the line above the delimiter row, the line above the first row (or the
        # table's last line). Lines of a paragraph above the header stay a paragraph, which
        # cmark-gfm (0.29.0.gfm.6 and .13) inserts before the table without a source position;
        # the table keeps that paragraph's first line and column.
        header_line = (sourcepos(table_rows[0])[0] - 1 if table_rows else end) - 1
        if start < header_line:
            text_lines += [(n, column if n == start else None, "paragraph")
                           for n in range(start, header_line)]
            table.set("split", "1")

    # A paragraph without a source position is the text above a table header (above).
    for parent in root.iter():
        children = list(parent)
        for i, node in enumerate(children):
            if node.tag == NS + "paragraph" and node.get("sourcepos") is None and not (
                    i + 1 < len(children) and children[i + 1].get("split") == "1"):
                errors.append(f"{rel}: cmark-gfm gave a paragraph no source position that is not "
                              f"the text above a table header (unknown to this test)")
    for paragraph in root.iter(NS + "paragraph"):
        if paragraph.get("sourcepos") is not None:
            first, column, last, _ = sourcepos(paragraph)
            text_lines += [(n, column if n == first else None, "paragraph")
                           for n in range(first, last + 1)]
    # A setext heading's text is the lines above its underline; an ATX heading is one line that
    # opens with 1-6 '#' at its column.
    for heading in root.iter(NS + "heading"):
        first, column, _, _ = sourcepos(heading)
        if ATX.match(lines[first - 1], column - 1):
            continue
        underline = setext_underline(cmark, lines, first, column)
        if underline is None:
            errors.append(f"{rel}:{first}: cmark-gfm gave a heading here that no parse of the "
                          f"file's first lines has (unknown to this test)")
            continue
        text_lines += [(n, column if n == first else None, "setext heading")
                       for n in range(first, underline)]

    # A text's first line starts at its column, the others after their leading spaces, tabs
    # and block-quote markers.
    text = {}
    for n, c, kind in text_lines:
        line = lines[n - 1][c - 1:] if c else lines[n - 1]
        text[n] = (line.decode("utf-8").lstrip(" \t>"), kind)
    for n in sorted(text):
        line, kind = text[n]
        if not DELIMITER_ROW.fullmatch(line):
            continue
        detail = "; no line above it"
        if n - 1 in text:
            above, delimiter = row_cells(text[n - 1][0] + "\n"), row_cells(line + "\n")
            detail = (f"; the line above has {cells(above) if above else 'no cells'}, this row "
                      f"{delimiter}")
        errors.append(f"{rel}:{n}: a delimiter row in a {kind}: GFM made no table of it{detail}")
    return tables, rows


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: docs_tables.py REPO_ROOT")
    root = Path(sys.argv[1]).resolve()
    cmark = shutil.which("cmark-gfm")
    if cmark is None:
        print("FAIL cmark-gfm is not on PATH (Debian and Ubuntu package cmark-gfm)")
        return 1
    version = subprocess.run([cmark, "--version"], capture_output=True, text=True,
                             timeout=60).stdout.split(" - ")[0].strip()
    errors = [f"{r} is missing" for r in REQUIRED if not (root / r).is_file()]
    files = sorted({p for pat in PATTERNS for p in root.glob(pat) if p.is_file()})
    tables = rows = 0
    for p in files:
        t, r = check(cmark, p, p.relative_to(root), errors)
        tables, rows = tables + t, rows + r
    for e in errors:
        print(f"FAIL {e}")
    if not errors:
        print(f"ok   {tables} tables ({rows} rows) in {len(files)} Markdown files keep their "
              f"columns ({version})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
