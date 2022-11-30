## Towards ML-based diagnostics of focused laser pulse

We considered the problem of focusing an extremely short laser pulse, taking into account the complex structure of the wavefront.

We numerically simulated the propagation of a tightly focused laser pulse to the focal plane, where the cumulative energy flux was calculated to be used for the ML-model training. In order to make the laser pulse model closer to experimental conditions, we impose a spectral-dependent tilt on its wavefront, and the properties of this tilt act as the latent parameters of the model. The modelling was performed using the [Hi-Chi](https://github.com/hi-chi/pyHiChi) framework.

We demonstrate that with the help of a convolutional neural network, it is possible to obtain a good recovery accuracy of the tilt parameters. To study the generalizing ability of the proposed ML model, we used a technique based on studying the influence of the choice of latent parameters on the accuracy of the ML model. In experiments, it was shown that the network has good generalization ability and can reconstruct the tilt parameters with quite good accuracy even for the latent parameter values not included in the training set.

## Data

The data is available via the [link](https://cloud.unn.ru/s/6K2zgbKR7XG4gZC)

## Paper

Rodimkov Y. R. et al. Towards ML-based diagnostics of focused laser pulse //arXiv preprint arXiv:2209.09959. – 2022.

## Project Structure

```plaintext
├── README.md 
├── dependency.txt - dependency
├── scripts - scripts for training models
```