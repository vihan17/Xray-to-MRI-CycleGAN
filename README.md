# Enhancing X-ray Images to Resemble MRI or CT Scans using CycleGAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.6+-blue?logo=python)](https://python.org)

## Overview

Medical imaging plays a crucial role in diagnosing and monitoring health conditions. X-ray is widely used for skeletal structures, but lacks the clarity of MRI and CT scans for soft tissues. This project bridges that gap by enhancing X-ray images to resemble MRI or CT scans using **CycleGAN** — a deep learning model that performs unpaired image-to-image translation.

## How CycleGAN Works

CycleGAN uses two key components: **generator** and **discriminator** networks.

- The **generator** learns to map images from one domain to another (X-ray → MRI/CT)
- The **discriminator** tries to distinguish real images from translated ones

A **cycle consistency loss** ensures that translating an image X → Y → X returns a result similar to the original, enabling bidirectional translation without paired training data.

![CycleGAN Diagram](https://github.com/vihan17/GANsproject/assets/91966446/fec55559-272f-465c-ac1b-c7daea754117)

## Workflow

1. **Data Collection** — Gather X-ray, MRI, and CT images from medical databases covering diverse anatomical regions and conditions.

2. **Data Preprocessing** — Normalize pixel values, resize images, apply augmentation (rotation, flipping) for diversity.

3. **Image Enhancement** — Experiment with histogram equalization, adaptive histogram equalization, denoising, and edge enhancement.

4. **Deep Learning** — Use GANs (CycleGAN, cGANs, CNNs) to learn the X-ray → MRI/CT mapping.

![Workflow Diagram](https://github.com/vihan17/GANsproject/assets/91966446/0d2cf8da-777e-428c-837f-4b6561c34313)

## Results

### Sample X-Ray Images

![Sample X-Ray](https://github.com/vihan17/GANsproject/assets/91966446/fd82d0f1-3b4d-4c3b-b2ee-044acf71b033)

### Histogram Equalization

![Histogram Equalization](https://github.com/vihan17/GANsproject/assets/91966446/5ed5c9c7-914b-44a5-859c-cc2aa13d958c)

### Adaptive Histogram Equalization

![Adaptive Histogram Equalization](https://github.com/vihan17/GANsproject/assets/91966446/bce75a91-df13-4635-8b94-be75b7d0063e)

### Final Results

![Final Results](https://github.com/vihan17/GANsproject/assets/91966446/2395f1a2-3685-4cf7-bee6-8a72b9441d90)

## Project Structure

```
GANsproject/
├── cyclegan_style_transfer.ipynb   # Main notebook with training & inference
├── requirements.txt                 # Python dependencies
├── .gitignore
├── LICENSE                          # MIT License
├── README.md
└── utils/
    ├── image_transformation.py      # Image normalization, resizing, jitter, blur
    ├── image_utils.py               # File I/O utilities for images
    └── model_utils.py               # Generator, discriminator, training loop
```

## Getting Started

### Prerequisites

- Linux, macOS, or Windows
- Python 3.6+
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vihan17/GANsproject.git
   cd GANsproject
   ```

2. Create a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > For GPU support, replace `tensorflow` with `tensorflow-gpu` in `requirements.txt`.

### Running

Start Jupyter and open the notebook:

```bash
jupyter notebook cyclegan_style_transfer.ipynb
```

Run all cells — the notebook will automatically download the dataset and begin training.

## Datasets

- [SHAIP X-Ray Image Datasets](https://www.shaip.com/offerings/x-ray-images-datasets/)
- [Kaggle Chest X-Ray (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## References

- [CycleGAN Paper (arXiv)](https://arxiv.org/pdf/1703.10593.pdf)
- [TensorFlow CycleGAN Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan)
- [Transforming the World into Paintings with CycleGAN](https://medium.com/analytics-vidhya/transforming-the-world-into-paintings-with-cyclegan-6748c0b85632)
- [How to Develop CycleGAN Models from Scratch with Keras](https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/)
- [Style Transfer with GANs on HD Images](https://towardsdatascience.com/style-transfer-with-gans-on-hd-images-88e8efcf3716)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
