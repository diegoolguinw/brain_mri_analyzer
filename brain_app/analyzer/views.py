import base64
import os
import uuid

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from .forms import MRIUploadForm
from .inference_onnx import analyze_image
from .report import generate_pdf_report


def upload_view(request):
    """Main page: upload an MRI image for analysis."""
    if request.method == "POST":
        form = MRIUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = request.FILES["image"]

            # Save uploaded file
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            ext = os.path.splitext(uploaded.name)[1] or ".png"
            filename = f"{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(upload_dir, filename)
            with open(filepath, "wb") as f:
                for chunk in uploaded.chunks():
                    f.write(chunk)

            # Run inference
            result = analyze_image(filepath)

            # Encode overlay as base64 for inline display
            overlay_b64 = base64.b64encode(result.overlay_png_bytes).decode("ascii")

            # Save PDF to disk so we can serve it later
            pdf_bytes = generate_pdf_report(result, result.overlay_png_bytes)
            results_dir = os.path.join(settings.MEDIA_ROOT, "results")
            os.makedirs(results_dir, exist_ok=True)
            pdf_filename = f"report_{uuid.uuid4().hex[:12]}.pdf"
            pdf_path = os.path.join(results_dir, pdf_filename)
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            context = {
                "result": result,
                "overlay_b64": overlay_b64,
                "pdf_filename": pdf_filename,
                "uploaded_name": uploaded.name,
            }
            return render(request, "analyzer/result.html", context)
    else:
        form = MRIUploadForm()

    return render(request, "analyzer/upload.html", {"form": form})


def download_report(request, filename):
    """Serve a previously generated PDF report."""
    pdf_path = os.path.join(settings.MEDIA_ROOT, "results", filename)
    if not os.path.exists(pdf_path):
        return HttpResponse("Report not found.", status=404)

    with open(pdf_path, "rb") as f:
        response = HttpResponse(f.read(), content_type="application/pdf")
        response["Content-Disposition"] = f'attachment; filename="{filename}"'
        return response
