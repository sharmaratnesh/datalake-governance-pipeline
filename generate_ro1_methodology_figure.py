"""
generate_ro1_methodology_figure.py — Regenerate research_paper/figures/ro1_methodology.png

Renders the RO1 (Experiment 1) methodology overview as a six-stage flowchart,
followed by an evaluation summary panel, using matplotlib only (no external
diagram tool, so the figure can be regenerated whenever the methodology changes).

Usage:
  python generate_ro1_methodology_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT_PATH = Path(__file__).parent / "research_paper" / "figures" / "ro1_methodology.png"

FIG_W, FIG_H = 8.6, 13.0
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 14.2)
ax.axis("off")

INK       = "#1C2530"
MUTED     = "#6B7280"
BOX_FILL  = "#FFFFFF"
BOX_LINE  = "#D8DCE2"
ITEM_FILL = "#F6F7F9"
HEAD_FONT = "DejaVu Serif"
BODY_FONT = "DejaVu Sans"

LEFT, RIGHT = 0.55, 9.45
WIDTH = RIGHT - LEFT


def badge(cx, cy, number, color):
    ax.add_patch(Circle((cx, cy), 0.20, facecolor=color, edgecolor="none", zorder=4))
    ax.text(cx, cy, str(number), fontsize=10.5, fontweight="bold", color="white",
            ha="center", va="center", zorder=5, fontfamily=BODY_FONT)


def stage_box(y_top, height, number, title, subtitle, accent, items):
    """Outer card (white, thin border, coloured left accent bar) + numbered badge
    + title/subtitle + a row of inner sub-boxes."""
    y_bottom = y_top - height

    outer = FancyBboxPatch(
        (LEFT, y_bottom), WIDTH, height,
        boxstyle="round,pad=0,rounding_size=0.07",
        linewidth=1.0, edgecolor=BOX_LINE, facecolor=BOX_FILL, zorder=1,
    )
    ax.add_patch(outer)
    accent_bar = FancyBboxPatch(
        (LEFT, y_bottom), 0.09, height,
        boxstyle="round,pad=0,rounding_size=0.045",
        linewidth=0, facecolor=accent, zorder=2,
    )
    ax.add_patch(accent_bar)

    badge(LEFT + 0.42, y_top - 0.34, number, accent)
    ax.text(LEFT + 0.74, y_top - 0.27, title, fontsize=11.8, fontweight="bold",
            color=INK, va="top", fontfamily=HEAD_FONT)
    if subtitle:
        ax.text(LEFT + 0.74, y_top - 0.58, subtitle, fontsize=8.2,
                 color=MUTED, va="top", style="italic", fontfamily=BODY_FONT)

    n = len(items)
    gap = 0.16
    inner_top = y_top - 0.80
    inner_bottom = y_bottom + 0.16
    box_w = (WIDTH - 0.34 - gap * (n - 1)) / n
    x0 = LEFT + 0.17

    for i, (head, lines) in enumerate(items):
        bx = x0 + i * (box_w + gap)
        inner = FancyBboxPatch(
            (bx, inner_bottom), box_w, inner_top - inner_bottom,
            boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=0, facecolor=ITEM_FILL, zorder=2,
        )
        ax.add_patch(inner)
        ty = inner_top - 0.20
        ax.text(bx + box_w / 2, ty, head, fontsize=8.1, fontweight="bold",
                color=INK, ha="center", va="top", fontfamily=BODY_FONT)
        for j, line in enumerate(lines):
            ax.text(bx + box_w / 2, ty - 0.26 - j * 0.225, line, fontsize=7.1,
                     color=MUTED, ha="center", va="top", fontfamily=BODY_FONT)

    return y_bottom


def arrow(y_from, y_to):
    a = FancyArrowPatch(
        (5.0, y_from), (5.0, y_to),
        arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color="#9AA1AC", zorder=3,
    )
    ax.add_patch(a)


# ── Title block ────────────────────────────────────────────────────────────────
ax.text(5.0, 13.95, "Semantic Data Lake Transformation Pipeline",
        fontsize=15.5, fontweight="bold", color=INK, ha="center", fontfamily=HEAD_FONT)
ax.text(5.0, 13.62,
        "From an ungoverned data swamp to a semantically enriched, metadata-driven repository",
        fontsize=9, color=MUTED, ha="center", style="italic", fontfamily=BODY_FONT)

y = 13.18
ACCENTS = ["#B23A48", "#2E6E9E", "#6A4C93", "#C08A1E", "#2E8B57", "#1C2530"]

# ── Stage 1 ──────────────────────────────────────────────────────────────────
y_bot = stage_box(
    y, 1.45, 1,
    "Unstructured Data Swamp",
    "55 files, mixed manual and programmatic generation, no metadata or naming convention",
    ACCENTS[0],
    [
        ("Text / CSV", []),
        ("JPEG (scans)", []),
        ("WAV (audio)", []),
        ("JSON / Tabular", []),
    ],
)
arrow(y_bot - 0.03, y_bot - 0.30)
y = y_bot - 0.33

# ── Stage 2 ──────────────────────────────────────────────────────────────────
y_bot = stage_box(
    y, 1.30, 2,
    "Format-Specific Content Extraction",
    "Uniform text representation across modalities",
    ACCENTS[1],
    [
        ("Parse text / CSV / JSON", ["Python stdlib"]),
        ("OCR (Tesseract)", ["image documents"]),
        ("Speech-to-text", ["local Whisper", "(faster-whisper)"]),
    ],
)
arrow(y_bot - 0.03, y_bot - 0.30)
y = y_bot - 0.33

# ── Stage 3 ──────────────────────────────────────────────────────────────────
y_bot = stage_box(
    y, 1.65, 3,
    "Semantic Analysis: Three Parallel Pipelines",
    "Data class category and security classification, independently computed by each pipeline",
    ACCENTS[2],
    [
        ("Rule-Based", ["keyword rules only", "priority-ordered"]),
        ("NLP", ["TF-IDF vectorisation +", "cosine similarity"]),
        ("LLM (Ollama)", ["Phi-3 Mini, LLaMA 3.2 1B", "+ auditable rationale"]),
    ],
)
arrow(y_bot - 0.03, y_bot - 0.30)
y = y_bot - 0.33

# ── Stage 4 ──────────────────────────────────────────────────────────────────
y_bot = stage_box(
    y, 1.45, 4,
    "Entity Extraction & Relationship Mapping",
    "Driving-licence image used as the authoritative source of truth for identity",
    ACCENTS[3],
    [
        ("Extract customer entity", ["name, ID, DOB", "address (regex)"]),
        ("Link complaints", ["match text/audio to", "known customer"]),
        ("Persist to SQLite", ["customers, complaints", "data_files"]),
    ],
)
arrow(y_bot - 0.03, y_bot - 0.30)
y = y_bot - 0.33

# ── Stage 5 ──────────────────────────────────────────────────────────────────
y_bot = stage_box(
    y, 1.30, 5,
    "Graph Modelling & Ontology Generation",
    "Knowledge graph (NetworkX) and dynamic OWL ontology (RDF/XML)",
    ACCENTS[4],
    [
        ("Knowledge graph", ["Customer / Complaint / DataFile", "HAS_COMPLAINT, RELATED_TO"]),
        ("OWL ontology", ["Protege-compatible", "SPARQL-ready schema"]),
    ],
)
arrow(y_bot - 0.03, y_bot - 0.30)
y = y_bot - 0.33

# ── Stage 6 ──────────────────────────────────────────────────────────────────
y_bot = stage_box(
    y, 0.95, 6,
    "Governed, Semantically Enriched Data Lake",
    "",
    ACCENTS[5],
    [
        ("Metadata summary (CSV)", []),
        ("Entity knowledge graph", []),
        ("Record-level governance", []),
    ],
)

y = y_bot - 0.28

# ── Evaluation summary panel ───────────────────────────────────────────────────
eval_h = 2.35
eval_box = FancyBboxPatch(
    (LEFT, y - eval_h), WIDTH, eval_h,
    boxstyle="round,pad=0,rounding_size=0.07",
    linewidth=1.0, edgecolor=BOX_LINE, facecolor="#FAFBFC", zorder=1,
)
ax.add_patch(eval_box)
ax.text(5.0, y - 0.27, "Evaluation: 55-File Data Swamp and External Corpus vs Expert Golden Labels",
        fontsize=10.8, fontweight="bold", color=INK, ha="center", fontfamily=HEAD_FONT)

ax.text(LEFT + 0.45, y - 0.62, "Metrics", fontsize=8.6, fontweight="bold",
        color=INK, fontfamily=BODY_FONT)
metrics = ["Information-type accuracy, per class", "Sensitivity accuracy and Cohen's kappa",
           "Customer linkage accuracy",
           "Entity and relation precision / recall", "Throughput, latency and break-even"]
for i, m in enumerate(metrics):
    ax.text(LEFT + 0.45, y - 0.88 - i * 0.205, m, fontsize=7.5, color=MUTED, fontfamily=BODY_FONT)

ax.text(LEFT + 5.05, y - 0.62, "Compared against", fontsize=8.6, fontweight="bold",
        color=INK, fontfamily=BODY_FONT)
comps = ["3 blinded human participants (isolated)", "4-expert Delphi consensus baseline",
         "Supervised baselines: logistic regression,", "   linear SVM, majority-class dummy",
         "Text Anonymization Benchmark (1,268 docs)"]
for i, c in enumerate(comps):
    ax.text(LEFT + 5.05, y - 0.88 - i * 0.205, c, fontsize=7.5, color=MUTED, fontfamily=BODY_FONT)

ax.text(5.0, y - 1.98, "All inference self-hosted \u2014 no external or third-party API at any stage",
        fontsize=8.3, color=ACCENTS[4], fontweight="bold", ha="center", style="italic",
        fontfamily=BODY_FONT)
ax.text(5.0, y - 2.18,
        "Robustness: five decoding conditions  |  1B\u20138B parameter scale probe  |  "
        "inference-runtime and CPU/GPU comparison",
        fontsize=6.9, color="#9AA1AC", ha="center", fontfamily=BODY_FONT)

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=600, bbox_inches="tight", facecolor="white")
plt.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")   # vector copy
plt.close()
print(f"Saved -> {OUT_PATH} and {OUT_PATH.with_suffix('.pdf')}")
