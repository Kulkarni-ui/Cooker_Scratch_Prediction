🍳 Cooker Scratch Detection using Deep Learning
This project implements a **Deep Learning model** using **Keras/TensorFlow** to detect and classify **surface defects (Scratches)** on metal cooker surfaces. The aim is to automate defect detection in manufacturing and quality assurance.  

## Features  
- End-to-end training pipeline in Jupyter Notebook.  
- Preprocessing and augmentation of image dataset.  
- CNN-based defect classification model.  
- Model saved in `.h5` format for deployment.  
- Prediction output includes **class label + confidence score**.  

## Dataset  
The dataset used is the **NEU Metal Surface Defects Dataset**, consisting of images categorized into various defect types such as:  
- Scratch  
- Inclusion  
- Patches
- Pitted  
- Rolled-in Scale  
- etc.  

## ⚙️ Tech Stack  
- Python  
- TensorFlow / Keras  
- NumPy, Pandas, Matplotlib, Seaborn  
- Scikit-learn  

## Project Workflow  
1. **Data Preparation**  
   - Load and split dataset into train/test.  
   - Apply normalization and augmentation.  

2. **Model Building**  
   - Convolutional Neural Network (CNN).  
   - Trained with categorical crossentropy & Adam optimizer.  

3. **Evaluation**  
   - Accuracy and loss plots.  
   - Confusion Matrix & Classification Report.  

4. **Prediction**  
   - Example:  
     ```
     Prediction: Scratch (Confidence:0.9997)
     ```  

## Results

- Test Accuracy: **0.99**
- Precision (No Scratch): **0.99**
- Recall (No Scratch): **1.00**
- F1-score (No Scratch): **0.99**
- Precision (Scratch): **1.00**
- Recall (Scratch): **0.93**
- F1-score (Scratch): **0.96**

### Confusion Matrix (Summary)
|               | Predicted No Scratch | Predicted Scratch |
|---------------|----------------------|-------------------|
| **Actual No Scratch** | 144 | 0 |
| **Actual Scratch**    | 2   | 27 |

### Overall
- **Macro Avg F1-score:** 0.98  
- **Weighted Avg F1-score:** 0.99  

## How to Run  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Cooker-Scratch-Detection.git
   cd Cooker-Scratch-Detection
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Open Jupyter Notebook:  
   ```bash
   jupyter notebook Cooker_Scratch.ipynb
   ```
4. Train the model or use the pre-trained `.h5` file for predictions.  

## Future Scope  
- Extend to multi-defect classification.  
- Deploy as a web app (Flask/Streamlit).  
- Integrate with manufacturing pipelines for real-time defect detection.  

## License  
This project is licensed under the MIT License.  

## Acknowledgements  
- NEU Metal Surface Defects Dataset.  
- TensorFlow & Keras Documentation.  
