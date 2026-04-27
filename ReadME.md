# 🫁 Lung Cancer Detection System

An AI-powered web application that detects lung cancer from CT scan images using a Convolutional Neural Network (CNN). The system provides multi-class predictions, risk assessment, and generates a professional medical report in PDF format.

---

## 🚀 Features

* 📤 Upload CT scan images
* 🧠 CNN-based multi-class prediction:

  * Benign
  * Malignant
  * Normal
* 📊 Confidence score visualization (line graph)
* ⚠️ Risk level classification (Low / Medium / High)
* 🧾 Auto-generated professional PDF report
* 👤 Patient details form (Name, Age, Sex)
* 🖼️ CT scan image included in report

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **Model:** TensorFlow / Keras (CNN)
* **Libraries Used:**

  * NumPy
  * Pillow
  * Matplotlib
  * FPDF

---

## 🧠 Model Overview

* Input: CT Scan Image (224 × 224)
* Architecture:

  * 3 Convolutional Layers
  * MaxPooling Layers
  * Fully Connected Dense Layers
  * Dropout for regularization
* Output: 3-class Softmax (Benign, Malignant, Normal)

---

## 📂 Project Structure

```
Lung-Cancer-Detection/
│── app.py
│── lung_cancer_model.h5
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation & Setup

1. Clone the repository:

```
git clone https://github.com/your-username/lung-cancer-detection.git
cd lung-cancer-detection
```

2. Create virtual environment (optional but recommended):

```
python -m venv venv
venv\Scripts\activate   # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
streamlit run app.py
```

---

## 🌐 Deployment

This app can be deployed using **Streamlit Community Cloud**:

1. Push project to GitHub
2. Go to Streamlit Cloud
3. Connect repository
4. Deploy `app.py`

---

## 📊 Output

* Prediction label (Benign / Malignant / Normal)
* Confidence score
* Risk level
* Visualization graph
* Downloadable PDF medical report

---

## ⚠️ Disclaimer

This system is developed for **educational purposes only**.
It is not intended to replace professional medical diagnosis.
Always consult a qualified healthcare professional for medical advice.

---

## 👩‍💻 Author

**Rakshanda Noor**
B.Tech Project – Lung Cancer Detection using Deep Learning

---

## ⭐ Acknowledgment

Dataset sourced from Kaggle.
Model trained using CNN architecture for medical image classification.

---
