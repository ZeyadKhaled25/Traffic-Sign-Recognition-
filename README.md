# **🚗 Traffic Sign Recognition System**

## **📝 Overview**

This project acts as the "eyes" of an autonomous vehicle, enabling it to see and interpret road signs in real-time. We built a robust Deep Learning model (CNN) capable of classifying **43 distinct classes** of traffic signs (e.g., Speed Limits, Stop, Yield) with high precision.  
The system processes live video feeds, filters out noise, and stabilizes predictions to ensure safety-critical reliability.  
**Dataset Used:** [GTSRB \- German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## **🌟 Key Features**

* **🧠 High Accuracy CNN:** A custom Sequential Convolutional Neural Network achieving **96.35% accuracy** on test data.  
* **⚡ Real-Time Inference:** Optimized OpenCV pipeline for processing live webcam streams with minimal latency.  
* **🎨 Color Space Handling:** Specific preprocessing to fix RGB/BGR mismatch issues, ensuring red-border signs (like Speed Limits) are detected correctly.  
* **🛡️ Prediction Stabilization:** Uses a Deque buffer to smooth out predictions over the last 10 frames, eliminating flickering classifications.  
* **🚫 Noise Filtering:** Includes a strict **Confidence Threshold (\>80%)**; the system ignores uncertain inputs to prevent false positives.

## **🛠️ Architecture & Tech Stack**

* **Language:** Python 3.x  
* **Frameworks:** TensorFlow, Keras  
* **Computer Vision:** OpenCV (cv2), PIL  
* **Data Manipulation:** NumPy, Pandas, Scikit-learn

### **Model Architecture (CNN)**

The model consists of a deep sequential network optimized for image classification:

1. **Conv2D Layers (x4):** For feature extraction (shapes, edges, patterns).  
2. **MaxPooling Layers:** To reduce spatial dimensions and computation.  
3. **Dropout (0.25):** To prevent overfitting during training.  
4. **Dense Layers:** For high-level reasoning.  
5. **Softmax Output:** To classify the image into one of the 43 probability classes.

## **🚀 Installation & Usage**

### **1\. Clone the Repository**

git clone \[https://github.com/ZeyadKhaled25/Traffic-Sign-Recognition-\](https://github.com/ZeyadKhaled25/Traffic-Sign-Recognition-)  
cd Traffic-Sign-Recognition-

### **2\. Install Dependencies**

pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn

### **3\. Train the Model (Optional)**

If you want to retrain the network from scratch:

1. Download the **GTSRB** dataset from Kaggle.  
2. Update the DATA\_DIR path in the notebook.  
3. Run the Jupyter Notebook:  
   jupyter notebook german-traffic-signs-classification-97.ipynb

   Or use the colab link:

4. This will save the trained model as traffic\_sign\_model.h5.

### **4\. Run Live Detection**

Ensure you have a webcam connected and run the main script:  
python traffic\_sign\_clean.py

*Press q to quit the video feed.*

## **📊 Performance**

| Metric | Score |
| :---- | :---- |
| **Training Accuracy** | \~98% |
| **Validation Accuracy** | \~98% |
| **Test Accuracy** | **96.35%** |

### **Challenges Solved**

1. **Color Space Mismatch:** Fixed discrepancies between OpenCV (BGR) and Training Data (RGB) which previously caused confusion between Red and Blue borders.  
2. **Video Jitter:** Implemented a smoothing buffer to prevent the class label from jumping between similar signs (e.g., Speed 80 vs Speed 100).

## **👥 Authors**

This project was developed by:

* **Zeyad Khaled Ehab** \- [GitHub Profile](https://www.google.com/search?q=https://github.com/ZeyadKhaled25)  
* **Omar Abdeltawab**

## **📜 License**

This project is licensed under the MIT License \- see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.  
*This project is a prototype for Autonomous Vehicle Perception Systems.*