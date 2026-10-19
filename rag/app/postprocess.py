"""Pure text post-processing of model answers: citation syntax, leaked
reasoning, citation repair and renumbering. No LLM or database calls."""
import re

from . import config

# --- Citation syntax ------------------------------------------------------------
# gpt-oss emits citations in the format it was trained on rather than the [6]
# this pipeline expects. Two variants seen in production:
#   【6】            -- CJK/full-width brackets
#   【2†L31-L38】    -- OpenAI's file-citation form, with a dagger and a
#                      line-range annotation invented from whole cloth
# Every downstream consumer -- remap_citations, verify_citations, the
# no-citation retry, and the frontend's link injection -- matches on ASCII
# \[\d+\] only. An un-normalised answer therefore loses its entire citation
# chain: verification is skipped, no link renders, and the grounding judge
# reports "cites a source that is not provided" -- which triggers the
# expensive expansion path on essentially every query.
#
# The \d{1,3} bound is deliberate: it keeps bracketed years such as [1963],
# which are common in this archive, from being rewritten as citations.
_CITE_BRACKETS = re.compile(r'[【\[［〔]\s*(\d{1,3})\s*(?:†[^】\]］〕]*)?\s*[】\]］〕]')
HAS_CITATION = re.compile(r'\[\d+\]')
_REFUSAL = re.escape(config.REFUSAL_LINE.rstrip('.')) + r'\.?'


def normalize_citations(text):
    return _CITE_BRACKETS.sub(r'[\1]', text)


def strip_artifacts(text):
    """Normalise citations and remove leaked reasoning from an answer."""
    text = normalize_citations(text)
    text = re.sub(r'\$\\boxed\{([^}]*)\}\$', r'\1', text)
    text = re.sub(r'(?m)^.*The final answer is:?.*$', '', text)
    # Heading optional: plain "Step 1:" lines leaked through when this required a '#'.
    text = re.sub(r'(?m)^(?:#+\s*)?Step \d+:.*$', '', text)
    # Simple mode's "start with one bold sentence" rule bolds the refusal too,
    # which then fails every exact-match check on it.
    text = re.sub(rf'^\s*\*\*\s*({_REFUSAL})\s*\*\*', r'\1', text)
    # Refusal line should never carry citations — it claims no facts.
    text = re.sub(rf'({_REFUSAL})(\s*\[\d+\])+', r'\1', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def strip_doc_echo(text):
    """Document answers sometimes open by restating the question."""
    text = re.sub(
        r'^\s*(?:#+\s*)?(?:User\s*Inquiry|USER\s*INQUIRY|The user (?:has requested|is asking)|Based on your question|The provided document pages?)[^\n]*\n+',
        '',
        text,
        flags=re.IGNORECASE,
    )
    return strip_artifacts(text)


def is_refusal(text):
    return text.strip().startswith(config.REFUSAL_LINE[:40])


_WS_RE = re.compile(r'\s+')


def normalize_ws(s):
    # OCR splits phrases across newlines (e.g. "Robert\nFulton"), so any
    # exact-substring search on raw content under-counts hits. Collapse
    # whitespace before searching.
    return _WS_RE.sub(' ', s)


# --- Citation units ---------------------------------------------------------------
# The verifier judges each citation *instance* — "does source [N] support the
# sentence it is attached to" — so its verdicts have to be applied at the same
# granularity. Stripping every `[N]` in the answer because one attachment of N
# was weak deleted whole documents from an answer (and, via remap_citations,
# from the source panel) on the strength of a single marginal judgement. To
# scope a verdict we need a stable address for each occurrence, so the answer
# is split into numbered units and the verifier reports (unit, source) pairs.
#
# The split must be lossless: re.split with a capturing group keeps the
# separators, so units[i] + seps[i] reassembles the original text byte for
# byte. Over-splitting (abbreviations, "No." in a file reference) is harmless
# — it only makes units smaller, never misaligns them.
_CITE_UNIT_SPLIT = re.compile(r'(\n+|(?<=[.!?])\s+)')


def split_cite_units(text):
    """Split an answer into sentence/line units. Returns (units, separators)
    such that join_cite_units(units, separators) == text."""
    parts = _CITE_UNIT_SPLIT.split(text)
    return parts[0::2], parts[1::2]


def join_cite_units(units, seps):
    out = []
    for i, unit in enumerate(units):
        out.append(unit)
        if i < len(seps):
            out.append(seps[i])
    return "".join(out)


# --- Citation repair ------------------------------------------------------------
def _tidy_after_removal(text):
    # [^\S\n] = any horizontal whitespace, including the narrow no-break space
    # (U+202F) the model often puts before a citation; [ \t] missed it and
    # left "Kennedy ." behind.
    text = re.sub(r'[^\S\n]{2,}', ' ', text)
    return re.sub(r'[^\S\n]+([.,;:!?])', r'\1', text)


def strip_unsupported_citations(answer, unsupported):
    """Fix only the flagged [N] occurrences, each in the one unit where the
    verifier judged it unsupported. Other citations of the same source, in
    sentences it does support, are left alone.

    Each verdict is (unit, source, correct_source). A dropped citation means
    the claim and its [N] did not line up -- usually the fact is real but was
    attached to the neighbouring source -- so when the verifier names the
    source that does contain it, re-point the citation instead of deleting it.
    Only a claim no source supports loses its citation."""
    if not unsupported:
        return answer
    units, seps = split_cite_units(answer)
    by_unit = {}
    for verdict in unsupported:
        unit_idx, src = verdict[0], verdict[1]
        fix = verdict[2] if len(verdict) > 2 else None
        by_unit.setdefault(unit_idx, {})[src] = fix
    for unit_idx, fixes in by_unit.items():
        if not 0 <= unit_idx < len(units):
            continue
        text = units[unit_idx]
        # Remove the marker only, leaving surrounding whitespace intact —
        # eating the leading space turns "April [2][5]" into "April[5]" when
        # the neighbouring citation survives. Tidy up afterwards instead.
        # Placeholders keep a re-pointed [2]->[5] from being caught by a
        # later [5] removal in the same unit.
        for n, fix in sorted(fixes.items(), reverse=True):
            text = re.sub(rf'\[{n}\]', f'[__FIX_{fix}__]' if fix else '', text)
        text = re.sub(r'\[__FIX_(\d+)__\]', r'[\1]', text)
        # A re-point can duplicate a neighbour: "[5][5]" -> "[5]".
        text = re.sub(r'(\[\d+\])(?:\s*\1)+', r'\1', text)
        units[unit_idx] = _tidy_after_removal(text)
    return join_cite_units(units, seps)


_BULK_CITES = re.compile(r'[^\S\n]*(?:\[\d+\][^\S\n]*){3,}')


def drop_bulk_citations(text):
    """Remove runs of 3+ adjacent citations ("...the other files [1][2]...[20]").
    The prompts forbid them and they point at nothing in particular; left in,
    the verifier strips them one by one anyway, at the cost of a bigger call."""
    return _tidy_after_removal(_BULK_CITES.sub(' ', text))


_TABLE_SEP = re.compile(r'^\s*\|?\s*:?-{2,}')
_SOURCE_HEADER = re.compile(r'^\s*(source|sources|src|#|no\.?|ref)\s*$', re.I)


def cite_table_source_columns(text, n_sources):
    """Rewrite bare numbers in a table's Source column ("| 3 | memo ... |") as
    citations ("[3]"). Bare numbers are neither clickable nor checked by the
    citation verifier, and a model-built source table numbered them wrongly
    (row 3 described source 2). As [N] each row is a verifier unit, so a
    mismatched row gets re-pointed like any other citation."""
    lines = text.split('\n')
    for i in range(1, len(lines)):
        if not _TABLE_SEP.match(lines[i]) or '|' not in lines[i - 1]:
            continue
        header = [c.strip() for c in lines[i - 1].strip().strip('|').split('|')]
        cols = {k for k, h in enumerate(header) if _SOURCE_HEADER.match(h)}
        if not cols:
            continue
        j = i + 1
        while j < len(lines) and '|' in lines[j] and lines[j].strip():
            lead = '|' if lines[j].lstrip().startswith('|') else ''
            cells = lines[j].strip().strip('|').split('|')
            for k in cols:
                if k < len(cells):
                    cells[k] = re.sub(
                        r'(?<![\[\d])\b(\d{1,3})\b(?![\]\d])',
                        lambda m: f'[{m.group(1)}]' if 1 <= int(m.group(1)) <= n_sources else m.group(0),
                        cells[k],
                    )
            lines[j] = lead + '|'.join(cells) + ('|' if lead else '')
            j += 1
    return '\n'.join(lines)


def remap_citations(answer_text, all_sources):
    """Keep only actually-cited sources and renumber them 1..N in citation
    order. Returns (text, cited_sources); the caller must order the source
    panel the same way or every link resolves to the wrong document."""
    cited_nums = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', answer_text)))
    if not cited_nums:
        return answer_text, all_sources
    new_sources, remap = [], {}
    for new_idx, old_num in enumerate(cited_nums, 1):
        old_idx = old_num - 1
        if 0 <= old_idx < len(all_sources):
            new_sources.append(all_sources[old_idx])
            remap[old_num] = new_idx
    for old_num in sorted(remap.keys(), reverse=True):
        answer_text = answer_text.replace(f'[{old_num}]', f'[__CITE_{remap[old_num]}__]')
    for new_num in remap.values():
        answer_text = answer_text.replace(f'[__CITE_{new_num}__]', f'[{new_num}]')
    return answer_text, new_sources
