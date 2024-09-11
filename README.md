# Lung Segmentation from Chest X-ray Images

## Project Overview
This project implements a deep learning-based approach for lung segmentation from chest X-ray images. The dataset used consists of 800 X-ray images and 704 masks. The main objective is to match masks with corresponding images to train a model for segmenting lungs in new chest X-ray images.

## Dataset
- **Image Path**: `/home/Desktop/igihozo/archive/data/Lung Segmentation/CXR_png`
- **Mask Path**: `/home/Desktop/igihozo/archive/data/Lung Segmentation/masks`
  
The dataset contains a disparity between the number of images and masks, with 800 images and 704 masks. To address this, the script makes a 1-1 correspondence between the masks and images.

## Key Dependencies
- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- tqdm (for progress tracking)

### Python Libraries
```bash
pip install numpy tensorflow pandas opencv-python tqdm matplotlib
```

## Code Structure

1. **Data Preprocessing**:
    - The notebook processes image and mask files by splitting filenames to ensure each mask corresponds to its respective image.
    - Images are read using OpenCV, and some preprocessing steps (like CLAHE) may be applied for contrast enhancement.

2. **Model Architecture**:
    - A Convolutional Neural Network (CNN) is likely employed for segmentation, though specific details would depend on the model defined later in the notebook.

3. **Training**:
    - Once the data is prepared, a model is trained on the dataset to learn the lung segmentation task. (Training code would likely include compiling the model, setting loss functions, and defining evaluation metrics.)

4. **Evaluation**:
    - The model's performance can be evaluated on a test set, using metrics such as IoU (Intersection over Union) to measure segmentation accuracy.

## How to Run the Notebook

1. Clone the repository and ensure all dependencies are installed.
2. Place the dataset in the correct directory structure as expected by the notebook.
3. Run the notebook cell by cell to preprocess the data, build the model, and train it on the dataset.
4. Evaluate the model performance using the test dataset provided.

## Results
After training, the model should be capable of segmenting lung regions from chest X-ray images with reasonable accuracy, based on the quality of the dataset and the model architecture used.

## Conclusion
This project demonstrates the application of deep learning techniques for medical image segmentation, specifically for lungs in chest X-rays, a critical task in diagnosing respiratory conditions.
