# Electron density reconstruction

## Abstract
Diagnostic methods play a critical role in understanding the properties of fast particle beams, which is essential for experimentally validating theoretical studies in laser-plasma interactions. Two-screen magnetic spectrometers are commonly used to simultaneously measure both the energy and angular distribution of electrons. In a magnetic field, electrons are deflected according to their energy, resulting in light signals detected by a scintillator. However, the analysis of the obtained data often involves solving complex multi-parameter problems, which typically require heuristic approaches and manual intervention. In this work, we propose a method for reconstructing the electron distribution using a deep neural network. Unlike existing methods, the proposed approach enables the automatic and simultaneous reconstruction of both the energy and angular electron distribution. Since labeled experimental data is often unavailable, synthetic data generated through numerical simulations combined with data augmentation techniques are used for training the neural network. The results show that the network, trained on synthetic data, achieved an accuracy of 0.78 in cosine similarity between experimental data and data derived from numerical simulations based on the predicted distribution. Although the network did not achieve perfect accuracy due to the presence of significant noise in the data, it successfully reconstructed key features that are sufficient for practical applications.

## Data

The data is available via the [link](https://cloud.unn.ru/s/T37Z5Zc3MKBoBes)

## Project Structure

```markdown
├── datasets                # Folder for datasets (download experimental data and generate synthetic data)
│   ├── numerical_dataset   # Synthetic data (Numerical data) 
│   └── real_dataset        # Data from the experimental facility
├── module                  # Project modules
├── module_torch            # Modules using PyTorch
├── src                     # Source code
│   ├── data_gen            # Data generation
│   ├── inference           # Inference and evaluate
│   ├── metric              # Metrics module
│   └── preprocessing       # Data preprocessing module
│   └── train_electron_density_reconstruction  # Training model for electron density reconstruction
├── README.md               # Project description
└── requirements.txt        # Project dependencies (python=3.11)
```

## Paper
