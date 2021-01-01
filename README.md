# Computer Vision, Goethe University Frankfurt (Fall 2020)

## General Information
<img align="right" width="300" height="" src="https://upload.wikimedia.org/wikipedia/commons/1/1e/Logo-Goethe-University-Frankfurt-am-Main.svg">

**Instructors:**
* [Prof. Dr. Gemma Roig](http://www.cvai.cs.uni-frankfurt.de/team.html), email: roig@cs.uni-frankfurt.de
* Dr. Iuliia Pliushch
* Kshitij Dwivedi
* Matthias Fulde

**Institutions:**
  * **[Goethe University](http://www.informatik.uni-frankfurt.de/index.php/en/)**
  * **[Computational Vision & Artificial Intelligence](http://www.cvai.cs.uni-frankfurt.de/index.html)**

**Course Description:** 

How can we enable machines to obtain semantic information from image data? How can computers gain a high-level understanding of visual input, which in turn is necessary to solve many elaborate tasks previously reserved for humans?—The objective of this course is to present a modern, data-driven approach to solve these problems.

We will introduce several computer vision problems, ranging from object classification and localization to semantic segmentation and video analysis. The fundamentals of the classical approaches to these problems will be covered, but the main focus will be on deep learning algorithms—especially convolutional neural networks—that are used to hierarchically extract increasingly more abstract features from images, analogous to processes performed by the human visual system.

Besides the theoretical understanding of these algorithms, emphasis is placed on gaining practical experience: There will be weekly exercises accompagnying the lecture in the first half of the semester and a group project spanning the second half, with the final presentation of the group project contributing 30 percent to the course grade.

The assignments are based on the [Stanford CS231n course](http://cs231n.stanford.edu/).

## Assignments ##

### Problem Set 1 (Part 1) 

- 1 Introduction
  - 1.1 Prerequisites
- 2 Setup
  - 2.1 Load OpenCV and NumPy
  - 2.2 Load and configure Matplotlib for use in Jupyter notebook
  - 2.3 Define function to display images
  - 2.4 Define function to measure errors
- 3 Exercises
  - 3.1 Loading Images
  - 3.2 Color Spaces
    - 3.2.1 RGB to Grayscale
    - 3.2.2 Detecting Objects by Color
  - 3.3 Transformations
    - 3.3.1 Scaling
    - 3.3.2 Translation
    - 3.3.3 Rotation
  - 3.4 Image Filtering
    - 3.4.1 Convolution/Cross Correlation
    - 3.4.2 Averaging Filter
    - 3.4.3 Gaussian Blur
    - 3.4.4 Median Filter
  - 3.5 Image Gradient
    - 3.5.1 One Direction
    - 3.5.2 Two Directions
  - 3.6 Histogram Equalization
  - 3.7 Image Thresholding
    - 3.7.1 Estimating Threshold using Image Histograms
    - 3.7.2 Advanced Techniques

### Problem Set 1 (Part 2) 

- 1 Introduction
- 2 Setup
  - 2.1 Import Numpy, OpenCV and other helpers
  - 2.2 Load and configure Matplotlib for use in Jupyter notebook
  - 2.3 Enable auto reload of external modules
  - 2.4 Load the CIFAR-10 dataset
- 3 Exercises
  - 3.1 Preparation
  - 3.2 Distances
  - 3.3 Predicting Labels
  - 3.4 Cross Validation
  - 3.5 Preprocessing

### Problem Set 2 

- 1 Fully Connected Nets
  - Fully-Connected Neural Nets
  - Affine layer: forward
  - Affine layer: backward
  - ReLU activation: forward
  - ReLU activation: backward
  - "Sandwich" layers
  - Loss layers: Softmax and SVM
  - Two-layer network
  - Solver
  - Multilayer network
    - Initial loss and gradient check
- 2 Dropout
  - Dropout forward pass
  - Dropout backward pass
  - Fully-connected nets with Dropout
  - Regularization experiment
- 3 Convolutional Networks
  - Convolution: Naive forward pass
  - Aside: Image processing via convolutions
  - Convolution: Naive backward pass
  - Max-Pooling: Naive forward
  - Max-Pooling: Naive backward
  - Fast layers
  - Convolutional "sandwich" layers
  - Three-layer ConvNet
   - Sanity check loss
   - Gradient check
   - Overfit small data
   - Train the net
   - Visualize Filters
- 4 PyTorch
  - Introduction into how to train neural networks with PyTorch
  - What's this PyTorch business
  - Part I. Preparation
  - Part II. Barebones PyTorch
    - PyTorch Tensors: Flatten Function
    - BareBones PyTorch: Two-Layer Network
    - BareBones PyTorch: Three-Layer ConvNet
    - BareBones PyTorch: Initialization
    - BareBones PyTorch: Check Accuracy
    - BareBones PyTorch: Training Loop
    - BareBones PyTorch: Train a Two-Layer Network
    - BareBones PyTorch: Training a ConvNet
  - Part III. PyTorch Module API
    - Module API: Two-Layer Network
    - Module API: Three-Layer ConvNet
    - Module API: Check Accuracy
    - Module API: Training Loop
    - Module API: Train a Two-Layer Network
    - Module API: Train a Three-Layer ConvNet
  - Part IV. PyTorch Sequential API
    - Sequential API: Two-Layer Network
    - Sequential API: Three-Layer ConvNet
  - Part V. CIFAR-10 open-ended challenge
  
### Problem Set 3 

## Tools ## 
* Python 3
* Pytorch Framework
* OpenCV Framework
* Numpy, Matplotlib Framework

