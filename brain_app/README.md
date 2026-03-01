# Brain MRI Tumor Analyzer — Django App

A web application that lets you upload a brain MRI slice, runs automated tumor
segmentation using the trained U-Net model, and generates a downloadable PDF
report with findings.

## Quick Start

```bash
# 1. Install dependencies (from the brain_app/ directory)
cd brain_app
pip install -r requirements.txt

# 2. Run database migrations
python manage.py migrate

# 3. Start the development server
python manage.py runserver
```

Then open **http://127.0.0.1:8000** in your browser.

## How It Works

1. **Upload** a grayscale brain MRI image (PNG, JPEG, TIFF, BMP).
2. The app loads the trained U-Net checkpoint (`../checkpoints/unet_resse_best.pt`)
   and runs inference with **test-time augmentation** (4-fold flip).
3. The segmentation result is displayed as an overlay on the MRI slice.
4. A structured **PDF report** is generated with:
   - Detection status (tumor detected / not detected)
   - Quantitative metrics (area, location, confidence)
   - Natural-language findings summary
   - Recommendations
   - Disclaimer

## Project Structure

```
brain_app/
├── manage.py
├── requirements.txt
├── brain_app/          # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── analyzer/           # Main application
    ├── nn_models.py    # PyTorch model definitions (SmallUNet, AttentionUNet)
    ├── inference.py    # Model loading + TTA inference
    ├── report.py       # PDF report generator (ReportLab)
    ├── views.py        # Upload + result views
    ├── forms.py        # Image upload form
    ├── urls.py
    ├── templates/analyzer/
    │   ├── upload.html
    │   └── result.html
    └── static/analyzer/
        └── style.css
```

## Configuration

- **Checkpoint path**: defaults to `../checkpoints/unet_resse_best.pt`.
  Override with the `CHECKPOINT_PATH` environment variable.
- The app auto-detects CUDA / MPS / CPU for inference.

## Disclaimer

This tool is for **research and educational purposes only**. It is not a
medical device and must not be used for clinical diagnosis or treatment
decisions.
