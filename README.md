# Kaggle-Workout-Video-Classifier-PyTorch

## Overview

This project implements a deep learning pipeline for classifying workout types from video data. The system leverages a 3D Convolutional Neural Network (CNN) architecture to capture spatio-temporal features from workout videos and accurately predict the workout type.


---

## Project Structure

```
.
├── CNN3D.py               
├── VideoEncoder.py       
├── classification_layer.py 
├── gymdata.py            
├── run.py                 
├── solution.ipynb         
├── Video Classifier Assignment.pdf  
```

---

## Dataset

- **Source:** [Kaggle - Workout Fitness Video Dataset](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video/data)
- **Description:** Contains short video clips categorized by workout type (e.g., push-ups, squats, lunges, etc.) - 22 classes, organized in folders by class name.

---

## Features

- **3D CNN Architecture:** Extracts spatial and temporal information from video frames.
- **Custom Dataset Loader:** Efficient loading and preprocessing of video files.
- **End-to-End Training Pipeline:** Includes data loading, model training, and evaluation.
- **Reproducible Results:** All code and explanation are provided in the `solution.ipynb` notebook.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Typical dependencies (update as needed):

- torch
- torchvision
- numpy
- opencv-python

### 3. Download the dataset

- Download and unzip the dataset from [Kaggle](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video/data).
- Place the dataset in a folder named `data/` at the root of the project or update the path in `gymdata.py`.

### 4. Train and evaluate the model

You can either run the scripts directly:

```bash
python run.py
```

Or follow the analysis and step-by-step code in `solution.ipynb`:

```bash
jupyter notebook solution.ipynb
```

---

## Files Description

- `CNN3D.py`: Defines the 3D CNN feature extractor.
- `VideoEncoder.py`: Handles reading and encoding videos for input into the model.
- `classification_layer.py`: Final classification head used for workout type prediction.
- `gymdata.py`: Custom PyTorch Dataset class for loading workout videos and labels.
- `run.py`: Trains and evaluates the model; entry point for running experiments.
- `solution.ipynb`: Full project notebook with explanations, results, and insights.
- `Video Classifier Assignment.pdf`: Assignment brief and instructions.


---

## Results

You can find model performance accuracy metric in `solution.ipynb` notebook. 

---

## Citation

If you use this code or ideas from this repository, please cite the original dataset and consider mentioning this project.

---

## License

Please see [Kaggle's dataset terms of use](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video/data) for dataset licensing.

---

## Contact

For any questions or suggestions, feel free to open an issue or contact [Ohad Cohen](mailto\:ohad338@gmail.com).

---

Let me know if you want to add your actual name/email or any specific result/highlight from your notebook! I can also generate a requirements.txt if you need.


