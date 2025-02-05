# virtual_try_on_SIC

# AI-Based Virtual Try-On with VITON-HD Dataset

## Overview

This project implements an AI-powered virtual try-on system based on deep learning techniques, designed to simulate how different clothing items will appear on a person. Using the **VITON-HD dataset**, we create a high-resolution virtual try-on system that transfers a target clothing item onto a model image in a realistic and visually appealing way, synthesizing outputs at a resolution of **1024x768** pixels.

The system addresses key challenges in virtual try-on tasks, such as handling misalignments between clothing and body, and maintaining the texture and detail of the garments. It leverages the alignment-aware techniques introduced in VITON-HD (ALIAS Normalization and ALIAS Generator) to ensure high-quality, artifact-free results.

## Dataset

We use the **VITON-HD dataset**, which is specifically designed for high-resolution virtual try-on tasks. This dataset includes images of human models and clothing items, segmentation maps for the body and garments, and annotations for pose estimation.

### Dataset Structure:

```
VITON-HD/
│
├── images/
│   ├── model_images/          # Images of human models (1024x768 resolution)
│   ├── clothing_images/       # Images of clothing items (1024x768 resolution)
│
├── segmentation_maps/
│   ├── model_segmentations/   # Segmentation maps of the models, identifying body parts
│   ├── clothing_segmentations/ # Segmentation maps of the clothing items
│
├── annotations/
│   ├── pose_annotations/      # Keypoint annotations for human body pose estimation
│   ├── bounding_boxes/        # Bounding box annotations for models and garments
│
└── README.md                  # Information about the dataset
```

### Dataset Features:
- **High-Resolution Images**: 1024x768 images for both models and garments.
- **Detailed Segmentation Maps**: Provided for precise body part and garment region identification.
- **Pose Annotations**: Keypoints for human poses, helping with accurate clothing alignment.

To learn more about the dataset, please refer to the VITON-HD [repository link](https://github.com/viton-hd).

## Key Features of the AI-Based Virtual Try-On System

1. **High-Resolution Image Synthesis**: Generates try-on images at a resolution of **1024x768**, providing detailed and realistic virtual fitting experiences.
   
2. **ALIAS Normalization and Generator**: Utilizes alignment-aware techniques from VITON-HD to handle misalignment challenges and ensure the final output is free from artifacts.

3. **Texture Preservation**: Ensures that fine garment details, such as patterns and textures, are maintained during the virtual try-on process.

4. **Pose-Aware Synthesis**: Leverages pose annotations to align the clothing accurately with the body, regardless of pose or posture.

5. **Flexible Clothing Transfer**: The system works with different garment types (e.g., tops, dresses) and can be extended to support additional categories with further training.

## Installation

To set up and run the virtual try-on system, follow these steps:

1. **Clone the Repository**:
   ```
   git clone https://github.com/your-repo/virtual-try-on.git
   cd virtual-try-on
   ```

2. **Install Dependencies**:
   Ensure you have Python installed (>= 3.7), and then install the required dependencies.
   ```
   pip install -r requirements.txt
   ```

3. **Download the VITON-HD Dataset**:
   Download the **VITON-HD** dataset from (https://www.kaggle.com/datasets/marquis03/high-resolution-viton-zalando-dataset)
   ```
   data/
   ├── VITON-HD/
       ├── images/
       ├── segmentation_maps/
       ├── annotations/
   ```

4. **Training the Model**:
   You can train the virtual try-on model using the following command:
   ```
   python train.py --dataset_dir data/VITON-HD/ --output_dir models/
   ```

5. **Running Inference**:
   To run the virtual try-on system and generate results using the pre-trained model:
   ```
   python tryon.py --input_dir data/VITON-HD/images/ --model_dir models/pretrained/
   ```

6. **View Results**:
   The synthesized try-on images will be saved in the `results/` directory.

## Usage

This system allows you to perform virtual try-on by selecting a human model and a clothing item from the dataset. The output will be a synthesized image where the clothing item is realistically transferred onto the model, retaining the garment’s texture and fit.

Example usage:
```bash
python tryon.py --model_image data/VITON-HD/images/model_images/0001.jpg --clothing_image data/VITON-HD/images/clothing_images/0001.jpg --output results/0001_tryon.jpg
```

## Results and Output

- **Input**: A model image and a target clothing item.
- **Output**: A high-resolution image (1024x768) of the model wearing the target clothing.
  
Examples of generated try-on images can be found in the `results/` folder.

## Models and Training

The system supports the training of virtual try-on models using VITON-HD dataset. You can modify the hyperparameters in `train.py` to experiment with different architectures and techniques.

- **Pre-trained Models**: Pre-trained models are available in the `models/pretrained/` folder for direct inference.
- **Training Custom Models**: You can train custom models with different configurations by modifying the dataset, learning rate, or architecture.





