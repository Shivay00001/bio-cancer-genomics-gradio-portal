# Bio-Cancer Genomics Gradio Portal

A professional multi-omics analysis platform featuring real-world TCGA (The Cancer Genome Atlas) data integration and an interactive Gradio web interface.

## Features

- **Real TCGA Data Integration**: Downloads and processes realistic cancer genomics data for BRCA, LUAD, PAAD, and COAD.
- **Interactive Web Interface**: Complete GUI built with `gradio` for easy interaction.
- **Advanced Visualizations**: 15+ publication-ready plots using `plotly` and `seaborn` (ROC curves, Survival analysis, Drug-Target Networks).
- **AI-Powered Predictions**: Uses XGBoost and Random Forest for treatment response prediction.
- **Automated Reporting**: Generates comprehensive clinical analysis reports.

## Usage

```bash
python asi_tcga_gradio.py
```

Then open the Gradio URL (usually `http://127.0.0.1:7860`) in your browser.

## Dependencies

- `gradio`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `plotly`
- `requests`
- `beautifulsoup4`
