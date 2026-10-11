# Coin Vision Mini Project

An automated computer vision and digital image processing pipeline built with Python and OpenCV for coin counting, restoration, boundary segmentation, and physical classification.

## Key Pipeline Features
- **Task 2: Image Restoration:** Removal of Gaussian, Salt & Pepper, and Periodic noise using Gaussian Blur, Median Filtering, and 2D FFT Notch Reject Filtering.
- **Task 3: Linear Filtering & Edges:** Sobel gradient analysis, Canny hysteresis edge detection, and Hough Line Transform.
- **Task 4: Segmentation:** Comparison of Global, Otsu, and Adaptive Gaussian thresholding; separation of touching coin boundaries via Euclidean Distance Transform and Marker-Controlled Watershed.
- **Task 5: Model-Based Vision:** Extraction of morphological and chromatic features (Area, Perimeter, Circularity, HSV color moments) and classification using K-Nearest Neighbors (KNN).
- **Task 6: Stereo Vision:** Dense disparity map generation via Semi-Global Block Matching (StereoSGBM) and depth triangulation.

## Datasets & Attribution
- [Brazilian Coin Image Dataset (Kaggle)](https://www.kaggle.com/datasets/balabaskar/count-coins-image-dataset) — Reference multi-coin scene layout and counting benchmark.
- [Roman Republican Coin Dataset (Kaggle)](https://www.kaggle.com/datasets/mexwell/coin-image-dataset) — Numismatic physical wear and circularity reference.

## How to Run
```bash
pip install opencv-python numpy matplotlib scipy scikit-learn
python compvis.py
