"""Report generation: per-document JSON + Markdown, plus aggregate outputs.

JSON follows the assignment's output schema exactly (document_id,
total_integrity_risk, scores, peer_matches, multimodal_matches, explanations)
with a few clearly-named additive fields (external_matches, boilerplate,
warnings) that make reports easier for humans to verify.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .ai_detector import AISignal
from .external import ExternalResult, external_match_to_dict
from .models import ExtractedDoc
from .similarity import Alignment, PairMatch, alignment_to_dict
from .utils import get_logger, truncate

logger = get_logger(__name__)

RISK_EMOJI = {"HIGH": "\U0001F534", "MEDIUM": "\U0001F7E0", "LOW": "\U0001F7E2"}


@dataclass
class DocAnalysis:
    """Everything computed about one submission; input to the reporters."""

    doc: ExtractedDoc
    peer_matches: Dict[str, PairMatch] = field(default_factory=dict)
    external: Optional[ExternalResult] = None
    ai: Optional[AISignal] = None
    scores: Dict[str, float] = field(default_factory=dict)
    risk: str = "LOW"
    explanation: str = ""
    boilerplate_summary: Dict[str, int] = field(default_factory=dict)
    embedding_backend: str = ""
    ai_backend: str = ""


def compute_scores(
    peer_matches: Dict[str, PairMatch],
    external: ExternalResult,
    ai: AISignal,
    risk_cfg: dict,
) -> Dict[str, float]:
    peer = max((pm.coverage for pm in peer_matches.values()), default=0.0)
    return {
        "peer_similarity_percentage": round(peer, 1),
        "external_similarity_percentage": float(external.coverage_percentage),
        "ai_generated_probability": float(ai.score_0_100),
    }


def compute_risk(scores: Dict[str, float], top_pair: Optional[PairMatch], risk_cfg: dict) -> str:
    weights = risk_cfg["weights"]
    composite = (
        weights["peer"] * scores["peer_similarity_percentage"]
        + weights["external"] * scores["external_similarity_percentage"]
        + weights["ai"] * scores["ai_generated_probability"]
    )
    if top_pair is not None and top_pair.coverage >= risk_cfg.get("pair_high_floor", 70):
        return "HIGH"
    if composite >= risk_cfg.get("high", 60):
        return "HIGH"
    if composite >= risk_cfg.get("medium", 35):
        return "MEDIUM"
    # A very likely machine-generated document always deserves human review.
    if scores.get("ai_generated_probability", 0) >= risk_cfg.get("ai_medium_floor", 60):
        return "MEDIUM"
    # So does a document half-covered by an external published source.
    if scores.get("external_similarity_percentage", 0) >= risk_cfg.get("external_medium_floor", 50):
        return "MEDIUM"
    return "LOW"


def compose_explanation(
    analysis: DocAnalysis,
    boilerplate_files: int,
    max_passages: int = 3,
) -> str:
    """Concise, human-verifiable narrative (target: <30 seconds to verify)."""
    parts: List[str] = []
    top_pairs = sorted(
        analysis.peer_matches.values(), key=lambda p: p.coverage, reverse=True
    )
    if top_pairs:
        p = top_pairs[0]
        strongest = p.top_alignments(1)[0]
        loc = strongest.matched_chunk.location()
        parts.append(
            f"Peer overlap: {p.coverage:.0f}% of this document aligns with "
            f"{p.matched_file} (strongest passage similarity {strongest.similarity:.2f}, "
            f"source {loc})."
        )
        for p2 in top_pairs[1:3]:
            parts.append(f"Secondary overlap with {p2.matched_file} ({p2.coverage:.0f}%).")
    else:
        parts.append("No peer overlap above threshold was found in the corpus.")
    if boilerplate_files > 0:
        bp_words = analysis.boilerplate_summary.get("words", 0)
        parts.append(
            f"A case-prompt/header template shared across {boilerplate_files} submissions "
            f"was detected and discounted ({bp_words} boilerplate words excluded from scoring)."
        )
    if analysis.external and analysis.external.matches:
        m = analysis.external.matches[0]
        parts.append(
            f"External reference overlap: {analysis.external.coverage_percentage:.0f}% "
            f"- closest source {m.source_ref} (similarity {m.similarity:.2f})."
        )
    elif analysis.external:
        parts.append("No external reference overlap above threshold.")
    if analysis.ai:
        f = analysis.ai.features
        drivers = []
        cv = f.get("sentence_length_cv")
        if isinstance(cv, (int, float)):
            drivers.append(f"sentence-length variation {cv:.2f} (low = uniform/AI-like)")
        drivers.append(f"discourse-marker density {f.get('transitions_per_100_words', 0):.1f}/100 words")
        parts.append(
            f"AI-content estimate {analysis.ai.score_0_100:.0f}% "
            f"({analysis.ai.backend}: {', '.join(drivers)})."
        )
    for w in analysis.doc.warnings:
        parts.append(f"Note: {w}")
    if analysis.doc.error:
        parts.append(f"Processing failure: {analysis.doc.error} - no integrity scores could be computed.")
    parts.append(
        "Verify the cited passages below against the source files before applying any penalty."
    )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# JSON report (assignment schema)
# --------------------------------------------------------------------------- #
def build_json_report(analysis: DocAnalysis, cfg: dict, boilerplate_files: int) -> dict:
    doc = analysis.doc
    rep_cfg = cfg["report"]
    peer_entries: List[dict] = []
    multimodal_entries: List[dict] = []
    for pm in sorted(analysis.peer_matches.values(), key=lambda p: p.coverage, reverse=True):
        best = pm.top_alignments(rep_cfg["max_passages_per_pair"])
        for a in best:
            entry = alignment_to_dict(a)
            if a.query_chunk.modality != a.matched_chunk.modality:
                if len(multimodal_entries) < rep_cfg["max_multimodal_matches"]:
                    multimodal_entries.append(entry)
            elif len(peer_entries) < rep_cfg["max_peer_matches"]:
                peer_entries.append(entry)
    peer_entries.sort(key=lambda e: e["similarity_score"], reverse=True)
    multimodal_entries.sort(key=lambda e: e["similarity_score"], reverse=True)

    report = {
        "document_id": doc.file_name,
        "total_integrity_risk": analysis.risk,
        "scores": analysis.scores,
        "peer_matches": peer_entries,
        "multimodal_matches": multimodal_entries,
        "explanations": analysis.explanation,
        # ---- additive, human-verification fields (not part of required schema) ----
        "external_matches": [
            external_match_to_dict(m)
            for m in (analysis.external.matches if analysis.external else [])[: rep_cfg["max_peer_matches"]]
        ],
        "boilerplate_discount": analysis.boilerplate_summary,
        "processing": {
            "modality": doc.modality,
            "chunks": len(doc.chunks),
            "language": doc.language,
            "duration_seconds": doc.duration_s,
            "pages_estimated": doc.n_pages,
            "warnings": doc.warnings,
            "error": doc.error,
            "embedding_backend": analysis.embedding_backend,
            "ai_backend": analysis.ai_backend,
            "ai_features": analysis.ai.features if analysis.ai else {},
        },
    }
    return report


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _passage_block(a: Alignment) -> str:
    q, m = a.query_chunk, a.matched_chunk
    lines = [
        f"**Similarity {a.similarity:.3f}** (lexical Jaccard {a.lexical_jaccard:.2f})",
        f"> **Query [{q.location()}]:** \u201c{truncate(q.text, 300)}\u201d",
        f"> **Matched [{m.location()}, {m.source_file}]:** \u201c{truncate(m.text, 300)}\u201d",
    ]
    return "\n".join(lines)


def build_markdown_report(analysis: DocAnalysis, boilerplate_files: int) -> str:
    doc = analysis.doc
    emoji = RISK_EMOJI.get(analysis.risk, "\u26AA")
    lines = [
        f"# Integrity Report \u2014 `{doc.file_name}`",
        "",
        f"**Verdict: {emoji} {analysis.risk} RISK** (modality: {doc.modality})",
        "",
        "| Signal | Score |",
        "|---|---|",
        f"| Internal peer similarity | {analysis.scores.get('peer_similarity_percentage', 0):.1f}% |",
        f"| External / web similarity | {analysis.scores.get('external_similarity_percentage', 0):.1f}% |",
        f"| AI-content likelihood | {analysis.scores.get('ai_generated_probability', 0):.1f}% |",
        "",
        f"**Summary:** {analysis.explanation}",
        "",
    ]
    top_pairs = sorted(
        analysis.peer_matches.values(), key=lambda p: p.coverage, reverse=True
    )
    if top_pairs:
        lines += ["## Matched passages (verify these first)", ""]
        for pm in top_pairs[:4]:
            modal = "cross-modal (video \u2194 document)" if pm.cross_modal else "same modality"
            lines.append(
                f"### vs `{pm.matched_file}` \u2014 {pm.coverage:.1f}% coverage, {modal}"
            )
            lines.append("")
            for a in pm.top_alignments(3):
                lines += [_passage_block(a), ""]
    else:
        lines += ["## Matched passages", "", "No peer matches above threshold.", ""]
    if analysis.external and analysis.external.matches:
        lines += ["## External reference matches", ""]
        for m in analysis.external.matches[:3]:
            lines.append(f"- `{m.source_ref}` \u2014 similarity {m.similarity:.3f}: "
                         f"\u201c{truncate(m.matched_chunk.text, 160)}\u201d")
        lines.append("")
    if analysis.ai:
        f = analysis.ai.features
        lines += ["## AI-content analysis", ""]
        lines.append(
            f"Backend `{analysis.ai.backend}` estimate: **{analysis.ai.score_0_100:.1f}%** "
            "(an estimate of machine-generated style, not proof)."
        )
        lines.append("")
        if isinstance(f.get("sentence_length_cv"), (int, float)):
            lines.append(f"- sentence-length coefficient of variation: {f['sentence_length_cv']} "
                         "(low = uniform, AI-like)")
            lines.append(f"- discourse markers per 100 words: {f.get('transitions_per_100_words', 0)}")
            lines.append(f"- contractions: {f.get('contractions', 0)}")
            lines.append(f"- polished punctuation per 100 words: {f.get('polished_punct_per_100w', 0)}")
            lines.append(f"- first-person references: {f.get('first_person_hits', 0)}")
        else:
            lines.append(f"- features: {f}")
        lines.append("")
    if analysis.boilerplate_summary:
        lines += [
            "## Boilerplate discount",
            "",
            f"{analysis.boilerplate_summary.get('chunks', 0)} chunk(s) / "
            f"{analysis.boilerplate_summary.get('words', 0)} words matched the shared "
            f"case-prompt template present in {boilerplate_files} files and were excluded "
            "from the peer score (Scenario 4: benign common formatting).",
            "",
        ]
    if doc.warnings or doc.error:
        lines += ["## Processing notes", ""]
        for w in doc.warnings:
            lines.append(f"- \u26A0\uFE0F {w}")
        if doc.error:
            lines.append(f"- \u274C {doc.error}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Aggregate reports
# --------------------------------------------------------------------------- #
def build_summary_markdown(analyses: List[DocAnalysis], matrix: Dict[str, Dict[str, float]], cfg: dict) -> str:
    lines = [
        "# Submission Integrity \u2014 Batch Summary",
        "",
        f"**{len(analyses)} submissions processed.** Sorted by composite risk.",
        "",
        "| Document | Risk | Peer % | External % | AI % |",
        "|---|---|---|---|---|",
    ]
    ordered = sorted(
        analyses,
        key=lambda a: (
            -(
                cfg["risk"]["weights"]["peer"] * a.scores.get("peer_similarity_percentage", 0)
                + cfg["risk"]["weights"]["external"] * a.scores.get("external_similarity_percentage", 0)
                + cfg["risk"]["weights"]["ai"] * a.scores.get("ai_generated_probability", 0)
            )
        ),
    )
    for a in ordered:
        if a.doc.error and not a.doc.chunks:
            lines.append(f"| `{a.doc.file_name}` | \u274C FAILED | - | - | - |")
            continue
        lines.append(
            f"| `{a.doc.file_name}` | {a.risk} "
            f"| {a.scores.get('peer_similarity_percentage', 0):.1f} "
            f"| {a.scores.get('external_similarity_percentage', 0):.1f} "
            f"| {a.scores.get('ai_generated_probability', 0):.1f} |"
        )
    files = [a.doc.file_name for a in analyses if a.doc.chunks]
    lines += ["", "## Pairwise peer-coverage matrix (%)", ""]
    header = "| | " + " | ".join(files) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(files) + 1))
    for row_file in files:
        cells = []
        for col_file in files:
            v = matrix.get(row_file, {}).get(col_file)
            cells.append("-" if v is None or row_file == col_file else f"{v:.0f}")
        lines.append(f"| `{row_file}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "*Cell value = % of the **row** document's words aligned to the **column** "
        "document at the similarity threshold. Self-comparisons excluded by design.*",
        "",
        "## Flagged submissions",
        "",
    ]
    flagged = [a for a in ordered if a.risk in {"HIGH", "MEDIUM"} and a.doc.chunks]
    if flagged:
        for a in flagged:
            top = sorted(a.peer_matches.values(), key=lambda p: p.coverage, reverse=True)
            src = top[0].matched_file if top else "-"
            lines.append(
                f"- **{a.doc.file_name} ({a.risk})** \u2014 top peer `{src}`, "
                f"external {a.scores.get('external_similarity_percentage', 0):.0f}%, "
                f"AI {a.scores.get('ai_generated_probability', 0):.0f}%"
            )
    else:
        lines.append("- None above the flagging threshold.")
    return "\n".join(lines)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
