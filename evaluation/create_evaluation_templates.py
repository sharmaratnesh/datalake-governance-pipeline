"""
Creates two XLSX evaluation templates for the 55-file data swamp study:

  evaluation_template_participant.xlsx  â€” distribute to each of the 3 participants
  evaluation_template_golden.xlsx       â€” researcher fills in golden labels
                                          (includes content_preview + rationale columns)

Run:
  python evaluation/create_evaluation_templates.py
"""

import csv
import json
from pathlib import Path

import openpyxl
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import (Alignment, Border, Font, PatternFill, Side)
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_SWAMP = Path(__file__).parent.parent / "data_swamp"
OUT_DIR    = Path(__file__).parent

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
HDR_FILL   = PatternFill("solid", fgColor="1F4E79")   # dark navy â€“ header row
LOCK_FILL  = PatternFill("solid", fgColor="D6DCE4")   # grey  â€“ read-only cells
INPUT_FILL = PatternFill("solid", fgColor="FFFBE6")   # cream â€“ participant input
NOTE_FILL  = PatternFill("solid", fgColor="F0F4F8")   # pale blue-grey â€“ content preview
ALT_FILL   = PatternFill("solid", fgColor="EBF3FB")   # light blue â€“ alternating row
WHITE      = PatternFill("solid", fgColor="FFFFFF")
GOLD_HDR   = PatternFill("solid", fgColor="7B3F00")   # brown â€“ golden labels header

HDR_FONT   = Font(name="Calibri", bold=True,  color="FFFFFF", size=11)
LOCK_FONT  = Font(name="Calibri", color="444444", size=10)
INPUT_FONT = Font(name="Calibri", color="000000", size=10)
PREV_FONT  = Font(name="Calibri", color="555555", size=9, italic=True)
TITLE_FONT = Font(name="Calibri", bold=True,  size=14, color="1F4E79")
GOLD_TITLE = Font(name="Calibri", bold=True,  size=14, color="7B3F00")
INSTR_HDR  = Font(name="Calibri", bold=True,  size=10, color="1F4E79")

_thin = Side(style="thin", color="BFBFBF")
CELL_BORDER = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)

# Conditional-format fills
CF_RED   = PatternFill("solid", fgColor="FCE4D6")   # confidential â†’ salmon
CF_GREEN = PatternFill("solid", fgColor="E2EFDA")   # general      â†’ light green

# ---------------------------------------------------------------------------
# Dropdown validation strings
# ---------------------------------------------------------------------------
CAT_LIST  = '"financial_transactions,identities,service_requests,unknown"'
SEC_LIST  = '"general,confidential"'
# Phase-1 known customers; participants type names for Phase-2 files
CUST_LIST = '"unknown,Jane Phillip,John Murray"'

# ---------------------------------------------------------------------------
# Extension â†’ file_type label
# ---------------------------------------------------------------------------
EXT_TYPE = {
    ".txt":  "text",
    ".csv":  "csv",
    ".json": "json",
    ".jpg":  "image",
    ".jpeg": "image",
    ".wav":  "audio",
}


def get_file_type(p: Path) -> str:
    return EXT_TYPE.get(p.suffix.lower(), "other")


# ---------------------------------------------------------------------------
# Content preview (golden template only)
# ---------------------------------------------------------------------------
def content_preview(p: Path, max_chars: int = 400) -> str:
    ext = p.suffix.lower()
    try:
        if ext == ".txt":
            return p.read_text(encoding="utf-8", errors="replace").strip()[:max_chars]

        if ext == ".csv":
            rows = []
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                for i, row in enumerate(csv.reader(f)):
                    if i >= 6:
                        break
                    rows.append(" | ".join(row))
            return "\n".join(rows)[:max_chars]

        if ext == ".json":
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            parts = []
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    parts.append(f"{k}: {v}")
                elif isinstance(v, dict):
                    for k2, v2 in list(v.items())[:4]:
                        if isinstance(v2, (str, int, float)):
                            parts.append(f"{k2}: {v2}")
            return " | ".join(parts)[:max_chars]

        if ext in (".jpg", ".jpeg"):
            return "[Image file â€” use OCR / visual inspection to classify]"

        if ext == ".wav":
            return "[Audio file â€” use speech-to-text transcription to classify]"

    except Exception as exc:
        return f"[Preview error: {exc}]"
    return ""


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------
def _cell(ws, row, col, value=None, *, fill=None, font=None,
          h_align="left", wrap=False, border=True):
    c = ws.cell(row=row, column=col, value=value)
    if fill:   c.fill   = fill
    if font:   c.font   = font
    c.alignment = Alignment(horizontal=h_align, vertical="center", wrap_text=wrap)
    if border: c.border = CELL_BORDER
    return c


def header(ws, row, col, text):
    return _cell(ws, row, col, text, fill=HDR_FILL, font=HDR_FONT, h_align="center")


def gold_header(ws, row, col, text):
    c = _cell(ws, row, col, text, fill=GOLD_HDR, font=HDR_FONT, h_align="center")
    return c


def locked(ws, row, col, value, alt=False):
    return _cell(ws, row, col, value, fill=ALT_FILL if alt else LOCK_FILL,
                 font=LOCK_FONT)


def inp(ws, row, col, value=None, alt=False):
    return _cell(ws, row, col, value, fill=ALT_FILL if alt else INPUT_FILL,
                 font=INPUT_FONT)


def preview_cell(ws, row, col, value, alt=False):
    c = _cell(ws, row, col, value, fill=ALT_FILL if alt else NOTE_FILL,
              font=PREV_FONT, wrap=True)
    return c


# ---------------------------------------------------------------------------
# Data-validation helper
# ---------------------------------------------------------------------------
def add_dv(ws, formula, first_row, last_row, col_letter, allow_other=False):
    """Add a list dropdown.  allow_other=True â†’ warning style (user can still type freely)."""
    dv = DataValidation(
        type="list",
        formula1=formula,
        allowBlank=True,
        showDropDown=False,                   # False = dropdown arrow IS shown
        showErrorMessage=not allow_other,
        error="Select a value from the dropdown list.",
        errorTitle="Invalid input",
        errorStyle="warning",                 # 'warning' lets users override
    )
    ws.add_data_validation(dv)
    dv.sqref = f"{col_letter}{first_row}:{col_letter}{last_row}"


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
# (header_label, width, kind)   kind: "lock" | "input" | "preview" | "rationale"
PARTICIPANT_COLS = [
    ("file_name",                18, "lock"),
    ("file_type",                10, "lock"),
    ("data_class_category",      24, "input"),
    ("security_classification",  24, "input"),
    ("linked_to_customer",       26, "input"),
    ("Total_time_taken_seconds", 22, "input"),
    ("notes",                    36, "input"),
]

GOLDEN_COLS = [
    ("file_name",                18, "lock"),
    ("file_type",                10, "lock"),
    ("data_class_category",      24, "input"),
    ("security_classification",  24, "input"),
    ("linked_to_customer",       26, "input"),
    ("Total_time_taken_seconds", 22, "input"),
    ("notes",                    36, "input"),
    ("content_preview",          55, "preview"),
    ("classification_rationale", 45, "rationale"),
]


# ---------------------------------------------------------------------------
# Instructions sheet
# ---------------------------------------------------------------------------
INSTRUCTIONS = [
    ("Column", "Description / Instructions"),
    ("file_name",
     "Pre-filled and locked. The exact filename as it appears in the data swamp folder. "
     "Do NOT edit this column."),
    ("file_type",
     "Pre-filled and locked. The format of the file:\n"
     "  image  â€” JPEG photograph or scanned document\n"
     "  audio  â€” WAV voice recording or call recording\n"
     "  text   â€” Plain text (.txt) document\n"
     "  csv    â€” Comma-separated values spreadsheet\n"
     "  json   â€” Structured JSON data file"),
    ("data_class_category",
     "REQUIRED â€” Select ONE value from the dropdown:\n"
     "  financial_transactions â€” bank statements, transaction records, payment/spending data\n"
     "  identities             â€” identity documents (driver licence, passport, KYC reports,\n"
     "                           customer profiles with personal identity information)\n"
     "  service_requests       â€” complaints, call recordings, emails, internal case notes\n"
     "  unknown                â€” file content cannot be determined or does not fit any category"),
    ("security_classification",
     "REQUIRED â€” Select ONE value from the dropdown:\n"
     "  confidential â€” file contains PII, account numbers, financial data, or identity docs;\n"
     "                 should not be accessible to general staff\n"
     "  general      â€” file contains no sensitive personal or financial information;\n"
     "                 can be shared broadly within the organisation"),
    ("linked_to_customer",
     "REQUIRED â€” If you can identify a specific customer in the file, type their full name.\n"
     "The dropdown suggests 'unknown', 'Jane Phillip', and 'John Murray' (from Phase 1 files).\n"
     "For all other files, TYPE the customer name you identify from the file content.\n"
     "If no customer can be identified, select or type  unknown."),
    ("Total_time_taken_seconds",
     "TIMING â€” Fill this field in the FIRST DATA ROW ONLY (row 3).\n"
     "Record the TOTAL number of seconds you spent completing the entire evaluation sheet.\n"
     "Start a stopwatch before opening the evaluation tab and stop it when you submit.\n"
     "Leave this column BLANK for all other rows."),
    ("notes",
     "OPTIONAL â€” Record any uncertainty, observations, or comments about individual files.\n"
     "Example: 'Could not play audio â€” classified based on filename only'"),
    ("Submission",
     "When complete, save this file as:  participant_<your_initials>.xlsx\n"
     "and return it to the corresponding author (see paper)"),
]


def add_instructions(wb, golden=False):
    ws = wb.create_sheet("Instructions", 0)
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 88

    # Title
    title_text = ("Researcher â€” Golden Labels Template" if golden
                  else "Participant Evaluation â€” Data Lake Management Study")
    c = ws.cell(row=1, column=1, value=title_text)
    c.font = GOLD_TITLE if golden else TITLE_FONT
    ws.merge_cells("A1:B1")
    ws.row_dimensions[1].height = 32

    sub = ws.cell(row=2, column=1,
                  value="University of Technology Sydney  |  Data Swamp: 55 files")
    sub.font = Font(name="Calibri", italic=True, size=10, color="666666")
    ws.merge_cells("A2:B2")
    ws.row_dimensions[2].height = 18

    ws.row_dimensions[3].height = 8   # spacer

    for i, (label, desc) in enumerate(INSTRUCTIONS, start=4):
        a = ws.cell(row=i, column=1, value=label)
        a.font = INSTR_HDR
        a.fill = PatternFill("solid", fgColor="D6E4F0")
        a.alignment = Alignment(vertical="top", wrap_text=True)
        a.border = CELL_BORDER

        b = ws.cell(row=i, column=2, value=desc)
        b.font = Font(name="Calibri", size=10)
        b.alignment = Alignment(vertical="top", wrap_text=True)
        b.border = CELL_BORDER

        n_lines = desc.count("\n") + 1
        ws.row_dimensions[i].height = max(20, n_lines * 16 + 6)

    if golden:
        # Extra instructions row for golden-specific columns
        extra_rows = [
            ("content_preview",
             "Auto-extracted text preview of the file content. "
             "Use this to inform your golden label decisions. "
             "For image and audio files, manually inspect the file before labelling."),
            ("classification_rationale",
             "REQUIRED for golden labels â€” Write a brief justification (1-2 sentences) "
             "explaining why you assigned each label. This rationale will be used to "
             "resolve participant disagreements and to document the baseline decisions."),
        ]
        for i, (label, desc) in enumerate(extra_rows,
                                           start=4 + len(INSTRUCTIONS)):
            a = ws.cell(row=i, column=1, value=label)
            a.font = Font(name="Calibri", bold=True, size=10, color="7B3F00")
            a.fill = PatternFill("solid", fgColor="F9E4CF")
            a.alignment = Alignment(vertical="top", wrap_text=True)
            a.border = CELL_BORDER
            b = ws.cell(row=i, column=2, value=desc)
            b.font = Font(name="Calibri", size=10)
            b.alignment = Alignment(vertical="top", wrap_text=True)
            b.border = CELL_BORDER
            n_lines = desc.count("\n") + 1
            ws.row_dimensions[i].height = max(20, n_lines * 16 + 6)


# ---------------------------------------------------------------------------
# Dropdown reference sheet
# ---------------------------------------------------------------------------
def add_reference_sheet(wb):
    ws = wb.create_sheet("Dropdown Reference")
    ws.sheet_view.showGridLines = False

    headers = ["data_class_category", "security_classification",
               "linked_to_customer (suggestions)"]
    values  = [
        ["financial_transactions", "general",      "unknown"],
        ["identities",             "confidential", "Jane Phillip"],
        ["service_requests",       "",             "John Murray"],
        ["unknown",                "",             "(or type any name you identify)"],
    ]

    for ci, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=ci, value=h)
        c.fill = HDR_FILL
        c.font = HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = CELL_BORDER
        ws.column_dimensions[get_column_letter(ci)].width = 32

    ws.row_dimensions[1].height = 24

    for ri, row_vals in enumerate(values, 2):
        for ci, val in enumerate(row_vals, 1):
            c = ws.cell(row=ri, column=ci, value=val)
            c.font = Font(name="Calibri", size=10)
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = CELL_BORDER
            if ri % 2 == 0:
                c.fill = ALT_FILL
        ws.row_dimensions[ri].height = 18


# ---------------------------------------------------------------------------
# Main evaluation sheet builder
# ---------------------------------------------------------------------------
def build_eval_sheet(wb, files, cols, golden=False):
    ws = wb.create_sheet("evaluation")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "C3"   # keep file_name + file_type visible while scrolling

    n_cols = len(cols)
    last_col_letter = get_column_letter(n_cols)

    # â”€â”€ Row 1: banner â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    banner = ("GOLDEN LABELS â€” Data Lake Evaluation | 55 files | Research Use Only"
              if golden else
              "Participant Evaluation â€” Data Lake Management Study | 55 files")
    c = ws.cell(row=1, column=1, value=banner)
    c.font = GOLD_TITLE if golden else TITLE_FONT
    c.fill = PatternFill("solid", fgColor="FDE9D9") if golden else PatternFill("solid", fgColor="D6E4F0")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.merge_cells(f"A1:{last_col_letter}1")
    ws.row_dimensions[1].height = 28

    # â”€â”€ Row 2: column headers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    col_map = {}   # name â†’ (1-based index, letter)
    for ci, (name, width, kind) in enumerate(cols, start=1):
        ltr = get_column_letter(ci)
        col_map[name] = (ci, ltr)
        ws.column_dimensions[ltr].width = width
        if golden and kind in ("preview", "rationale"):
            gold_header(ws, 2, ci, name.replace("_", " ").title())
        else:
            header(ws, 2, ci, name.replace("_", " ").title())
    ws.row_dimensions[2].height = 34

    # â”€â”€ Data rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for ri, p in enumerate(files, start=3):
        alt = (ri % 2 == 0)
        ftype = get_file_type(p)

        for ci, (name, _, kind) in enumerate(cols, start=1):
            if name == "file_name":
                locked(ws, ri, ci, p.name, alt)
            elif name == "file_type":
                locked(ws, ri, ci, ftype, alt)
            elif name == "content_preview":
                preview_cell(ws, ri, ci, content_preview(p), alt)
                ws.row_dimensions[ri].height = 60   # taller for preview text
            elif name == "Total_time_taken_seconds":
                # Leave blank; only first row gets a visual hint via note
                inp(ws, ri, ci, None, alt)
            else:
                inp(ws, ri, ci, None, alt)

        if not any(c[0] == "content_preview" for c in cols):
            ws.row_dimensions[ri].height = 18

    last_data_row = 2 + len(files)

    # â”€â”€ Add a visual hint in Total_time row 3 only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    time_ci, time_ltr = col_map["Total_time_taken_seconds"]
    hint_cell = ws.cell(row=3, column=time_ci)
    hint_cell.value = None   # participant fills this
    from openpyxl.comments import Comment
    cmt = Comment("Record TOTAL seconds for the entire sheet here (row 3 only).",
                  "Researcher")
    cmt.width, cmt.height = 220, 60
    hint_cell.comment = cmt

    # â”€â”€ Dropdowns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cat_ltr  = col_map["data_class_category"][1]
    sec_ltr  = col_map["security_classification"][1]
    cust_ltr = col_map["linked_to_customer"][1]

    add_dv(ws, CAT_LIST,  3, last_data_row, cat_ltr,  allow_other=False)
    add_dv(ws, SEC_LIST,  3, last_data_row, sec_ltr,  allow_other=False)
    add_dv(ws, CUST_LIST, 3, last_data_row, cust_ltr, allow_other=True)

    # â”€â”€ Conditional formatting: security column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sec_range = f"{sec_ltr}3:{sec_ltr}{last_data_row}"
    ws.conditional_formatting.add(
        sec_range,
        CellIsRule(operator="equal", formula=['"confidential"'], fill=CF_RED)
    )
    ws.conditional_formatting.add(
        sec_range,
        CellIsRule(operator="equal", formula=['"general"'], fill=CF_GREEN)
    )

    # â”€â”€ Conditional formatting: data_class_category â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    cat_colours = {
        "financial_transactions": "D9EAD3",  # green
        "identities":             "CFE2F3",  # blue
        "service_requests":       "FFF2CC",  # yellow
        "unknown":                "EFEFEF",  # grey
    }
    for value, hex_colour in cat_colours.items():
        ws.conditional_formatting.add(
            f"{cat_ltr}3:{cat_ltr}{last_data_row}",
            CellIsRule(
                operator="equal",
                formula=[f'"{value}"'],
                fill=PatternFill("solid", fgColor=hex_colour)
            )
        )

    return ws


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    files = sorted(DATA_SWAMP.glob("*.*"), key=lambda p: p.name.lower())
    print(f"Found {len(files)} files in data_swamp/\n")

    # â”€â”€ Participant template â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Building participant template â€¦")
    wb_p = openpyxl.Workbook()
    wb_p.remove(wb_p.active)
    add_instructions(wb_p, golden=False)
    add_reference_sheet(wb_p)
    build_eval_sheet(wb_p, files, PARTICIPANT_COLS, golden=False)
    # Set active sheet to evaluation for participants
    wb_p.active = wb_p["evaluation"]
    out_p = OUT_DIR / "evaluation_template_participant.xlsx"
    wb_p.save(str(out_p))
    print(f"  Saved: {out_p.name}")

    # â”€â”€ Golden labels template â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("Building golden labels template â€¦")
    wb_g = openpyxl.Workbook()
    wb_g.remove(wb_g.active)
    add_instructions(wb_g, golden=True)
    add_reference_sheet(wb_g)
    build_eval_sheet(wb_g, files, GOLDEN_COLS, golden=True)
    wb_g.active = wb_g["evaluation"]
    out_g = OUT_DIR / "evaluation_template_golden.xlsx"
    wb_g.save(str(out_g))
    print(f"  Saved: {out_g.name}")

    print(f"\nDone.  Both files written to:  {OUT_DIR}")


if __name__ == "__main__":
    main()

