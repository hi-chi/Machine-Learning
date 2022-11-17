## ML-Based Analysis of Particle Distributions in High-Intensity Laser Experiments: Role of Binning Strategy

We considered the problem of reconstructing the field amplitude using the distribution of electrons using machine-learning methods.

An ultra-intense laser pulse propagating through a counter-propagating monoenergetic electron bunch. To quantify electron spectra in a form suitable for the machine-learning problem, the full energy range from zero to initial energy is split into several bins, and the number of electrons in each bin is calculated.

We considered the effect of the binning strategy on the accuracy of several machine-learning models. We varied the size of bins used for the construction of the input vector from the energy spectra that can presumably be measured with high resolution. We compared various machine-learning methods (principal component analysis, gradient boosting of decision trees, neural networks, and support vector machines).

## Data

The data is available via the [link](https://cloud.unn.ru/s/KbX6aR8Pxq7Pn2J)

## Paper

Rodimkov Y. et al. ML-Based Analysis of Particle Distributions in High-Intensity Laser Experiments: Role of Binning Strategy //Entropy. – 2020. – Т. 23. – №. 1. – С. 21. [https://doi.org/10.3390/e23010021](https://doi.org/10.3390/e23010021)

## Project Structure

```plaintext
├── README.md 
├── generate_data - data generation scripts
├── example.ipynb - training and comparing machine learning models
```