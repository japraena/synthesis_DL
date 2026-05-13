# synthesis_DL
# Synthesis of Integrated Bragg Gratings via Deep Learning

[![Under Review](https://img.shields.io/badge/Status-Under%20Review-yellow)](https://www.journals.elsevier.com/optics-and-laser-technology)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

Deep learning framework for the inverse design of Integrated Bragg Gratings (IBGs) in silicon-on-insulator platforms using the Effective Refractive Index Transfer Matrix Method (ERI-TMM).

## 📋 Overview

This repository contains the **trained neural network models** for our paper **"Synthesis of Integrated Bragg Gratings via Deep Learning"**, currently under review in *Optics and Laser Technology*.

Our approach uses a Convolutional Neural Network (CNN) to learn the complex mapping from spectral responses (reflectivity and phase) to geometric parameters, providing a fast and accurate alternative to traditional iterative synthesis methods.

### Key Features

- **Dual-channel CNN architecture**: Incorporates both reflectivity and phase information as input
- **ERI-TMM-based dataset generation**: Efficient training data generation without high-performance computing requirements
- **Sub-nanometer precision**: Achieves sub-nanometer accuracy for Bragg period prediction
- **Comprehensive validation**: Validated on 36 hold-out test configurations with excellent spectral reconstruction
- **Multiple design configurations**: Supports uniform, apodized, and chirped gratings

## 🗂️ Repository Contents
```
synthesis_DL/
├── README.md
├── LICENSE
└── models/
    ├── V1_Gain.pth              # Model trained with reflectivity only
    ├── V2_Gain_Phase.pth        # Model trained with reflectivity + phase
    └── V3_Gain_Phase_Chirp.pth  # Model trained with reflectivity + phase + chirp (RECOMMENDED)
```

## 🧠 Available Models

This repository includes **three trained models** corresponding to different stages of our research development. All models use **raised-cosine (squared-cosine) apodization profile**.

### Model 1: **V1_Gain.pth** - Reflectivity-only model
- **Input**: Single-channel reflectivity spectrum R(λ)
  - Shape: `(batch, 1, 501)` - 1 channel, 501 wavelength points from 1500-1600 nm
- **Architecture**: CNN with raised-cosine apodization
- **Use case**: Baseline model for comparison
- **Grating types**: Uniform and apodized (non-chirped)

### Model 2: **V2_Gain_Phase.pth** - Reflectivity + Phase model
- **Input**: Dual-channel input [R(λ), φ(λ)]
  - Shape: `(batch, 2, 501)` - 2 channels, 501 wavelength points
- **Architecture**: CNN with raised-cosine apodization
- **Use case**: Improved accuracy with phase information
- **Grating types**: Uniform and apodized (non-chirped)

### Model 3: **V3_Gain_Phase_Chirp.pth** - Full model with chirp support ⭐ **RECOMMENDED**
- **Input**: Dual-channel input [R(λ), φ(λ)]
  - Shape: `(batch, 2, 501)` - 2 channels, 501 wavelength points
- **Architecture**: CNN with raised-cosine apodization
- **Use case**: Complete inverse design capability
- **Grating types**: Uniform, apodized, and chirped gratings
- **Performance**: Best overall accuracy and versatility

### Model Output Parameters

All models predict four geometric parameters:
- **Λ_B,i**: Initial Bragg period (nm)
- **Λ_B,f**: Final Bragg period (nm)
- **L**: Grating length (number of periods)
- **ΔW_max**: Maximum corrugation width modulation (nm)

**Note**: For uniform gratings, Λ_B,i = Λ_B,f. For chirped gratings, Λ_B,i < Λ_B,f.

## 📊 Dataset and Source Code

### Dataset Availability

The training dataset consists of **31,072 IBG configurations** generated using the ERI-TMM method:
- **Spectral range**: 1500-1600 nm
- **Wavelength points**: 501 samples
- **Platform**: 220 nm SOI with 500 nm waveguide width
- **Generation time**: ~2.3 seconds per configuration (MacBook Pro M4, MATLAB, CPU-only)

⚠️ **Due to the large size of the dataset, it is not included in this repository.**

The complete dataset is **available upon reasonable request**.

### Source Code Availability

The complete source code for:
- ERI-TMM simulation (MATLAB)
- Network architecture and training (Python/PyTorch)
- Validation and analysis scripts

is **available upon reasonable request** to ensure appropriate use and proper citation.

**Please contact**:
- **José Ángel Praena**: japrarod@upo.es

## 🚀 Using the Models

### Prerequisites
```
Python >= 3.8
PyTorch >= 1.12
NumPy >= 1.21
```

### Installation
```bash
pip install torch numpy
```

### Quick Start Example

⚠️ **Note**: The exact loading procedure depends on how the models were saved. The following example assumes the most common approach (state_dict). If you encounter errors loading the model, please contact us for the model architecture definition.
```python
import torch
import numpy as np

# Load the recommended model
# Note: This assumes the model was saved with torch.save(model.state_dict(), ...)
# If loading fails, contact the authors for the model architecture definition
model = torch.load('models/V3_Gain_Phase_Chirp.pth', map_location='cpu')

# If the model was saved as state_dict, you'll need to load it into the architecture:
# from model_architecture import IBG_CNN  # (contact authors for this file)
# model = IBG_CNN()
# model.load_state_dict(torch.load('models/V3_Gain_Phase_Chirp.pth'))

model.eval()  # Set to evaluation mode

# Prepare input: dual-channel spectral response [R(λ), φ(λ)]
# Your reflectivity and phase data should cover 1500-1600 nm with 501 points
reflectivity = np.array([...])  # Shape: (501,) - values between 0 and 1
phase = np.array([...])         # Shape: (501,) - phase in radians

# Stack into dual-channel input
input_spectrum = np.stack([reflectivity, phase], axis=0)  # Shape: (2, 501)
input_spectrum = torch.FloatTensor(input_spectrum).unsqueeze(0)  # Shape: (1, 2, 501)

# Predict geometric parameters
with torch.no_grad():  # No gradient computation needed for inference
    predicted_params = model(input_spectrum)

# Convert to numpy and extract parameters
predicted_params = predicted_params.cpu().numpy()[0]

Lambda_Bi = predicted_params[0]  # Initial Bragg period (nm)
Lambda_Bf = predicted_params[1]  # Final Bragg period (nm)
L = predicted_params[2]          # Length (number of periods)
DW_max = predicted_params[3]     # Max corrugation width modulation (nm)

print(f"Predicted IBG geometry:")
print(f"  Λ_B,i = {Lambda_Bi:.3f} nm")
print(f"  Λ_B,f = {Lambda_Bf:.3f} nm")
print(f"  L = {L:.1f} periods")
print(f"  ΔW_max = {DW_max:.3f} nm")

# Check if grating is chirped
if abs(Lambda_Bf - Lambda_Bi) < 0.1:
    print("  Type: Uniform or apodized (non-chirped)")
else:
    print(f"  Type: Chirped (chirp rate = {Lambda_Bf - Lambda_Bi:.3f} nm)")
```

### Input Requirements

**Wavelength range**: 1500-1600 nm  
**Number of points**: 501 (uniformly spaced)  
**Input shape**: (batch_size, channels, 501)
- For V1_Gain: channels = 1 (reflectivity only)
- For V2_Gain_Phase and V3_Gain_Phase_Chirp: channels = 2 (reflectivity + phase)

**Reflectivity**: Normalized values between 0 and 1  
**Phase**: In radians, typically between -π and π

### Expected Output Ranges

Based on the training dataset:
- **Λ_B,i**: 314-320 nm
- **Λ_B,f**: 314-320 nm
- **L**: 100-2500 periods
- **ΔW_max**: 5-20 nm

Predictions outside these ranges may be less reliable.

## 📖 Method Overview

### ERI-TMM (Effective Refractive Index Transfer Matrix Method)

The training dataset was generated using ERI-TMM, a computationally efficient method for analyzing IBGs:
1. Calculate effective refractive index profile n_eff(z)
2. Discretize into segments with piecewise-constant refractive index
3. Compute transfer matrices for each segment
4. Calculate global reflection and transmission coefficients
5. Extract spectral response R(λ) and φ(λ)

**Advantages over full-wave solvers:**
- ~1000× faster than FDTD
- Standard laptop hardware sufficient
- Experimentally validated for SOI platforms

### CNN Architecture

Our dual-channel CNN processes both reflectivity and phase information:
- **Input layer**: (batch, 2, 501) - 2 channels × 501 wavelength samples
- **Convolutional layers**: Extract spectral features with 1D convolutions
- **Pooling layers**: Downsample feature maps
- **Fully connected layers**: Map extracted features to geometric parameters
- **Output layer**: 4 parameters [Λ_B,i, Λ_B,f, L, ΔW_max]
- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate scheduling

The architecture details are available upon request.

## 📈 Performance

### Prediction Accuracy (36 hold-out test configurations)

**Geometric parameters:**
- **Λ_B,i** (Initial Bragg period):
  - MAE: 0.005 nm
  - RMSE: 0.007 nm
  - Max error: 0.013 nm
  - Relative error: 0.0016%

- **Λ_B,f** (Final Bragg period):
  - MAE: 0.012 nm
  - RMSE: 0.015 nm
  - Max error: 0.034 nm
  - Relative error: 0.0038%

- **L** (Grating length):
  - MAE: 13.10 periods
  - RMSE: 15.99 periods
  - Max error: 37.15 periods
  - Relative error: 1.164%

- **ΔW_max** (Max width modulation):
  - MAE: 0.241 nm
  - RMSE: 0.283 nm
  - Max error: 0.539 nm
  - Relative error: 2.410%

**Spectral reconstruction quality:**
- 75% perfect cases: Virtually indistinguishable from target spectra
- 25% fine cases: Minor deviations within acceptable tolerance
- Excellent agreement across entire 1500-1600 nm bandwidth

**Training performance:**
- Training dataset: 31,072 configurations
- Validation set: 10% of training data
- Test set: 36 independent hold-out configurations
- Hardware: MacBook Pro M4, CPU-only training
- Training time: ~2-3 hours per model

## 📚 Citation

If you use these models in your research, please cite our paper:
```bibtex
@article{praena2024synthesis,
  title={Synthesis of Integrated Bragg Gratings via Deep Learning},
  author={Praena, José Ángel and Gil, Guillermo and Caballero, Fernando and Carballar, Alejandro and Merino, Luis},
  journal={Optics and Laser Technology},
  year={2024},
  note={Under review}
}
```

## 👥 Authors

- **José Ángel Praena** - Universidad Pablo de Olavide & Universidad de Sevilla
  - Email: japrarod@upo.es
  
- **Guillermo Gil** - Universidad Pablo de Olavide
  - Email: ggilgar@upo.es
  
- **Fernando Caballero** - Universidad Pablo de Olavide
  - Email: fcaballero@upo.es
  
- **Alejandro Carballar** - Universidad de Sevilla
  - Email: carballar@us.es
  
- **Luis Merino** - Universidad Pablo de Olavide
  - Email: lmercab@upo.es

## 📧 Contact

For questions, dataset requests, source code requests, model architecture details, or collaborations:

**Corresponding author**: José Ángel Praena  
**Email**: japrarod@upo.es  
**Institution**: Universidad Pablo de Olavide & Universidad de Sevilla

We aim to respond to all inquiries within 1-2 business days.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### You are free to:
- ✅ Use the models for research purposes

### Under the conditions that you:
- ✅ Provide appropriate citation to our paper
- ✅ Acknowledge that the models are provided "as is" without warranty

## 🙏 Acknowledgments

- **Service Robotics Lab**, Universidad Pablo de Olavide
- **Departamento de Ingeniería Electrónica**, Universidad de Sevilla

## 🔗 Related Work

This work builds upon our previous research on ERI-TMM:
- Praena, J.A., et al. (2025). Detailed analysis of integrated Bragg gratings using ERI-TMM.
- Praena, J.A., et al. (2024). Detailed analysis of chirped integrated Bragg gratings using ERI-TMM.

## ⚠️ Known Issues and Limitations

- Models are trained specifically for SOI platforms (220 nm thickness, 500 nm width)
- Performance may degrade for geometries significantly outside training ranges
- Fabrication tolerances (typically >15 nm) may dominate over prediction errors
- Models assume raised-cosine apodization profile

For other platforms or apodization profiles, please contact us for potential collaboration.

## 📝 Version History

- **v1.0** (December 2025): Initial release with three trained models
  - V1_Gain: Reflectivity-only baseline
  - V2_Gain_Phase: Dual-channel without chirp
  - V3_Gain_Phase_Chirp: Full capability (recommended)

---

**Last updated**: December 2025  
**Repository**: https://github.com/japraena/synthesis_DL  
**Paper status**: Under review in *Optics and Laser Technology*  
**Models version**: 1.0  
**Framework**: PyTorch 1.12+
```

---

## 📝 **ARCHIVO LICENSE (MIT):**
```
MIT License

Copyright (c) 2025 José Ángel Praena, Guillermo Gil, Fernando Caballero, Alejandro Carballar, Luis Merino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
