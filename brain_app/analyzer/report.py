"""
PDF report generator — produces a structured medical-imaging analysis report.
Uses ReportLab for PDF creation.
"""

import io
import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    HRFlowable, PageBreak,
)

from .inference_onnx import AnalysisResult


# ---------------------------------------------------------------------------
# NLP-style narrative builder
# ---------------------------------------------------------------------------

def _generate_narrative(r: AnalysisResult) -> str:
    """Build a natural-language findings paragraph from the analysis result."""
    if not r.tumor_detected:
        return (
            "The automated segmentation model did not identify any region of abnormal signal "
            "intensity consistent with a tumor in the submitted MRI slice. "
            "The probability map remained below the detection threshold "
            f"({r.threshold:.2f}) across the entire image (peak probability: {r.max_prob:.2f}). "
            "No further characterization is provided for this slice. "
            "Clinical correlation and review of the full volumetric acquisition are recommended."
        )

    pct = r.tumor_pixel_fraction * 100
    confidence = "high" if r.mean_prob_in_mask > 0.80 else ("moderate" if r.mean_prob_in_mask > 0.55 else "low")

    narrative = (
        f"The model detected a region of abnormal signal consistent with a tumor, "
        f"occupying approximately {pct:.1f}% of the image area "
        f"(estimated area {r.tumor_area_mm2:.0f} mm²). "
        f"The lesion is centered in the {r.quadrant} quadrant of the axial slice. "
        f"Mean prediction confidence within the segmented region is {r.mean_prob_in_mask:.2f} "
        f"(classified as {confidence} confidence), with a peak probability of {r.max_prob:.2f}. "
    )

    if r.tumor_area_mm2 < 100:
        narrative += (
            "The lesion appears relatively small; this may represent an early-stage finding "
            "or a partial-volume effect at the tumor boundary. "
        )
    elif r.tumor_area_mm2 > 1500:
        narrative += (
            "The lesion is large, potentially indicating advanced disease or mass effect. "
            "Urgent clinical review is strongly recommended. "
        )

    narrative += (
        "This result is produced by an automated algorithm and is NOT a clinical diagnosis. "
        "Findings should be reviewed by a qualified radiologist in the context of the full "
        "imaging study and patient history."
    )
    return narrative


def _generate_recommendation(r: AnalysisResult) -> str:
    """Build a short recommendation section."""
    if not r.tumor_detected:
        return (
            "• No actionable finding on this slice.\n"
            "• If clinical suspicion persists, review additional slices or consider "
            "contrast-enhanced MRI."
        )
    lines = [
        "• Review the full volumetric acquisition for 3-D extent of the lesion.",
        "• Correlate with patient symptoms, prior imaging, and clinical history.",
    ]
    if r.tumor_area_mm2 > 500:
        lines.append("• Consider neurosurgery and/or oncology referral.")
    if r.mean_prob_in_mask < 0.6:
        lines.append(
            "• Model confidence is moderate-to-low — manual review is especially important."
        )
    lines.append("• This tool is for research/educational purposes only; it is NOT FDA-cleared.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def generate_pdf_report(result: AnalysisResult, overlay_png_bytes: bytes) -> bytes:
    """Return PDF bytes for the analysis report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        spaceAfter=6,
        textColor=colors.HexColor("#1a5276"),
    ))
    styles.add(ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.grey,
        spaceAfter=4,
    ))

    elements = []

    # Title
    elements.append(Paragraph("Brain MRI Tumor Segmentation Report", styles["Title"]))
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a5276")))
    elements.append(Spacer(1, 10))

    # Metadata table
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        ["Report generated", now],
        ["Model", result.model_name],
        ["Input resolution", f"{result.img_size} × {result.img_size} px"],
        ["Original image size", f"{result.original_size[0]} × {result.original_size[1]} px"],
        ["Detection threshold", f"{result.threshold:.2f}"],
    ]
    meta_table = Table(meta_data, colWidths=[2.2 * inch, 4 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # Detection summary
    elements.append(Paragraph("Detection Summary", styles["SectionTitle"]))
    status = (
        '<font color="red"><b>TUMOR DETECTED</b></font>'
        if result.tumor_detected
        else '<font color="green"><b>NO TUMOR DETECTED</b></font>'
    )
    elements.append(Paragraph(f"Status: {status}", styles["Body"]))

    if result.tumor_detected:
        findings_data = [
            ["Metric", "Value"],
            ["Tumor pixel fraction", f"{result.tumor_pixel_fraction * 100:.2f} %"],
            ["Estimated area", f"{result.tumor_area_mm2:.0f} mm²"],
            ["Location (quadrant)", result.quadrant],
            ["Centroid (x, y frac)", f"({result.centroid_x_frac:.2f}, {result.centroid_y_frac:.2f})"],
            ["Peak probability", f"{result.max_prob:.3f}"],
            ["Mean prob in mask", f"{result.mean_prob_in_mask:.3f}"],
        ]
        f_table = Table(findings_data, colWidths=[2.5 * inch, 3 * inch])
        f_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(Spacer(1, 6))
        elements.append(f_table)

    elements.append(Spacer(1, 14))

    # Overlay image
    elements.append(Paragraph("Segmentation Overlay", styles["SectionTitle"]))
    img_stream = io.BytesIO(overlay_png_bytes)
    rl_img = RLImage(img_stream, width=3.8 * inch, height=3.8 * inch)
    elements.append(rl_img)
    elements.append(Spacer(1, 14))

    # Narrative findings (NLP summary)
    elements.append(Paragraph("Findings", styles["SectionTitle"]))
    elements.append(Paragraph(_generate_narrative(result), styles["Body"]))
    elements.append(Spacer(1, 10))

    # Recommendations
    elements.append(Paragraph("Recommendations", styles["SectionTitle"]))
    for line in _generate_recommendation(result).split("\n"):
        elements.append(Paragraph(line, styles["Body"]))
    elements.append(Spacer(1, 16))

    # Disclaimer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an automated research prototype and "
        "does not constitute medical advice or clinical diagnosis. It has not been validated "
        "for clinical use and is not approved by any regulatory body (FDA, CE, etc.). "
        "All findings must be reviewed and confirmed by a qualified healthcare professional.",
        styles["Disclaimer"],
    ))

    doc.build(elements)
    buf.seek(0)
    return buf.read()
