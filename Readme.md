# Air Quality Prediction: Deep Learning with PyTorch

This project implements a Deep Neural Network to predict **Benzene (C6H6)** concentrations using various environmental sensor data. The model was built using **PyTorch**, leveraging the UCI Air Quality Dataset.

## 🚀 Project Overview
The goal was to create a "Virtual Sensor." Real-world Benzene sensors are expensive and difficult to maintain. By utilizing cheaper, more common sensors like Temperature, Humidity, and CO, this model provides high-value air quality insights at a fraction of the hardware cost.

## 📊 The Data Pipeline
* **Cleaning:** Replaced missing values (represented as -200) with `NaN` and performed linear interpolation to maintain time-series trends.
* **Quality Control:** Opted to drop remaining null values after interpolation to ensure high data integrity over quantity.
* **Preprocessing:** Implemented `StandardScaler` to normalize feature distributions, preventing gradient explosion and improving convergence speed.

## 🧠 Model Architecture & Hyperparameter Tuning
A systematic **Grid Search** was conducted across 10 different architectures to find the "Sweet Spot" between complexity and generalization.

| Num Hidden Layers | Units per Layer | Epochs | Train Loss (MSE) | Test Loss (MSE) |
| :--- | :--- | :--- | :--- | :--- |
| **1 (Selected)** | **256** | **200** | **0.0329** | **0.0402** |
| 1 | 128 | 200 | 0.0753 | 0.1112 |
| 2 | 256 | 200 | 0.0308 | 0.0471 |
| 2 | 128 | 200 | 0.0628 | 0.0863 |

**Decision Logic:** The 1-layer 256-unit model was selected as the final architecture. While deeper models showed slightly lower training loss, this configuration provided the best **Test Loss**, indicating superior robustness and less risk of overfitting on unseen data.

## 📈 Results
* **Final $R^2$ Score:** **0.9988** (99.88% of variance explained).
* **Residual Analysis:** Error distribution is tightly centered at zero, with the majority of predictions falling within **±1 mg/m³** of the ground truth.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib

## 📂 Project Structure
* `AirQuality_Analysis.ipynb`: Full data cleaning, EDA, and model training pipeline.
* `benzene_model_v1.pth`: Saved PyTorch state dictionary for the final model.
* `README.md`: Project documentation and results summary.

## 🏁 How to Use
1. Clone the repository.
2. Initialize the `RegressionModelv1` class with 11 input features and 1 output.
3. Load the pre-trained weights:
```python
import torch
# Initialize model architecture
model = RegressionModelv1(input_features=11, output_features=1)
# Load saved weights
model.load_state_dict(torch.load('benzene_model_v1.pth'))
model.eval()