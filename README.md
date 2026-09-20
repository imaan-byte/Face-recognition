# Face Recognition with MobileNetV2

A facial-image classification project built with Python, TensorFlow and Keras. The project uses transfer learning with a pretrained MobileNetV2 model to classify images of three enrolled individuals.

## Project Overview

The system processes facial images, trains a neural network and predicts which enrolled class a new image belongs to. MobileNetV2 provides pretrained visual features, while additional neural-network layers adapt the model to this classification task.

## Features

* MobileNetV2 transfer learning with ImageNet weights
* Three-class facial-image classification
* Separate training, validation and test datasets
* Image normalisation and data augmentation
* Dropout regularisation to help reduce overfitting
* Training-performance and accuracy visualisation
* Modular Python structure for data preparation, modelling and training

## Model Architecture

The model contains:

1. A pretrained MobileNetV2 feature extractor
2. A flattened feature representation
3. A dense layer with 128 neurons and ReLU activation
4. A dropout layer with a rate of 0.5
5. A three-class softmax output layer

The MobileNetV2 base is frozen during training. The model uses:

* Optimiser: Adam
* Learning rate: 0.0001
* Loss function: Categorical cross-entropy
* Evaluation metric: Accuracy
* Input image size: 224 × 224 pixels

## Data Preparation

All images are resized to 224 × 224 pixels and rescaled to values between 0 and 1.

The training dataset uses augmentation including:

* Rotation
* Width and height shifts
* Shearing
* Zooming
* Horizontal flipping

Validation and test images are rescaled without augmentation.

The expected dataset structure is:

```text
faces_folder/
├── train/
│   ├── Denzel/
│   ├── Imaan/
│   └── Sandra/
├── validation/
│   ├── Denzel/
│   ├── Imaan/
│   └── Sandra/
└── test/
    ├── Denzel/
    ├── Imaan/
    └── Sandra/
```

Each directory should contain facial images belonging to the corresponding person.

## Project Structure

```text
Face-recognition/
├── main.py
├── data_handling.py
├── model_arc.py
├── training.py
├── graph_accuracy.py
├── last copy.ipynb
└── README.md
```

* `main.py` — coordinates the project workflow
* `data_handling.py` — creates the data generators and performs augmentation
* `model_arc.py` — defines and compiles the MobileNetV2 model
* `training.py` — contains the model-training workflow
* `graph_accuracy.py` — visualises training performance
* `last copy.ipynb` — contains notebook-based development and experimentation

## Installation

Clone the repository:

```bash
git clone https://github.com/imaan-byte/Face-recognition.git
cd Face-recognition
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, use:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install tensorflow numpy matplotlib
```

## Running the Project

Place the authorised image dataset inside the expected folder structure and confirm that the dataset paths in the program are correct.

Run the project with:

```bash
python main.py
```

## Limitations

This is an educational facial-image classification project rather than a production biometric-identification system.

Its performance depends on factors including:

* The size and diversity of the dataset
* Lighting conditions
* Camera angle and image quality
* Differences between training images and new images
* Class imbalance

Because the model is trained on a small set of people, it should not be expected to generalise to individuals outside its enrolled classes.

## Ethical Use

Facial images should only be collected and used with the informed permission of the people represented. This project is not intended for surveillance, security decisions or other high-stakes applications.

## Future Improvements

* Add automatic face detection and cropping
* Evaluate the model using a confusion matrix
* Report precision, recall and F1-score for each class
* Test additional transfer-learning architectures
* Add early stopping and model checkpoints
* Improve documentation of experimental results
* Create a simple interface for uploading and classifying images

## Author

**Imaan Soliman**

Computer Science student interested in artificial intelligence, machine learning and engineering applications.
