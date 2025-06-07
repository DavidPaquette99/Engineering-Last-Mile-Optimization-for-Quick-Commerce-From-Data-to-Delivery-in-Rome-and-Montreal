# From Data to Delivery: Data-Driven Engineering of Last-Mile Optimization for Quick Commerce in Rome and Montreal

This repository contains the full code, data, and model artifacts for my master's thesis project at LUISS Guido Carli. The project focuses on engineering and benchmarking advanced courier assignment strategies for last-mile delivery in quick commerce, using data-driven and machine learning approaches.

---

## Project Structure

Code/ # All simulation engines, assignment strategies, model training scripts, and utils
Data/ # Datasets and trained model files for reproducibility
/Montreal # Cleaned Montreal datasets
/Rome # Cleaned Rome datasets
/ANFIS # Trained ANFIS models, scalers, and transformers
Maps/ # GIS and visualization outputs


---

## Getting Started

1. **Clone this repository**
    ```bash
    git clone https://github.com/DavidPaquette99/From-Data-to-Delivery-Data-Driven-Engineering-of-Last-Mile-Optimization-for-Quick-Commerce-in-Rome-.git
    cd From-Data-to-Delivery-Data-Driven-Engineering-of-Last-Mile-Optimization-for-Quick-Commerce-in-Rome-
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run a sample simulation**
    ```bash
    cd Code/simulation
    python Run_Simulation.py
    ```
    *(You may need to adapt paths depending on your use case. See `Code/README.md` for details on modules.)*

---

## Data

- All cleaned and processed data for **Montreal** and **Rome** are provided in `Data/Montreal/` and `Data/Rome/`.
- Trained ANFIS models and scalers are in `Data/ANFIS/` and used for courier assignment and regression models.

*No sensitive or private data is included. For full raw datasets or Google API keys, please contact the author.*

---

## Folder Guide

- `Code/` - Main simulation code, assignment strategies, training pipelines, utilities
- `Data/Montreal/` - Cleaned and engineered data for Montreal
- `Data/Rome/` - Cleaned and engineered data for Rome
- `Data/ANFIS/` - Trained ANFIS models, scalers, and transformers
- `Maps/` - Optional: map visualizations, GIS outputs

---

## Reproducibility

- All code and data required to reproduce core results are included.
- To retrain models, see scripts in `Code/model_training/`.
- Pretrained models are loaded from `Data/ANFIS/` (default paths are absolute, but can be adapted).

---

## Dependencies

Major dependencies (see `requirements.txt`):
- Python 3.8+
- numpy
- pandas
- scikit-learn
- joblib
- osmnx
- matplotlib

---

## Citation

If you use this code or data, please cite:

> Paquette, D. (2025). *From Data to Delivery: Data-Driven Engineering of Last-Mile Optimization for Quick Commerce in Rome and Montreal*. Master’s Thesis, LUISS Guido Carli.

---

## License

MIT License (or specify another if you prefer)

---

## Contact

Questions, feedback, or collaboration proposals welcome!  
**David Paquette**  
d.paquette@studenti.luiss.it

