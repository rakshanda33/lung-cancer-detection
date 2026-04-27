import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from fpdf import FPDF
import tempfile
import os
import datetime
import matplotlib.pyplot as plt
import gdown

# -------------------------------
# Download Model (ONLY ONCE)
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1HMLvxxL95RVFUcu7w39dyuex4ywRbrMx"
MODEL_PATH = "lung_cancer_model.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# -------------------------------
# Load Model
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["Benign", "Malignant", "Normal"]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Lung Cancer Detection", layout="wide")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🫁 Lung Cancer System")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])

# -------------------------------
# Home
# -------------------------------
if page == "Home":
    st.title("🏥 Lung Cancer Detection System")
    st.markdown("""
    CNN-based system to detect lung cancer from CT scans.

    ✔ Upload CT image  
    ✔ Multi-class prediction  
    ✔ Risk level classification  
    ✔ Download PDF report  
    """)

# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Prediction":

    st.title("🔍 Cancer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        uploaded_file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])

    def predict(image):
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)[0]
        class_index = np.argmax(preds)
        confidence = preds[class_index]

        return class_index, confidence, preds

    def get_risk(label, confidence):
        if label == "Malignant":
            if confidence > 0.7:
                return "High", "red"
            else:
                return "Medium", "orange"
        else:
            return "Low", "green"

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="CT Scan", width="stretch")

            if st.button("Predict"):

                class_index, confidence, preds = predict(image)
                label = class_names[class_index]
                risk, color = get_risk(label, confidence)

                st.markdown("### 🧾 Result Summary")

                st.markdown(f"""
                **Prediction:** :{color}[{label}]  
                **Confidence:** {confidence:.2f}  
                **Risk Level:** :{color}[{risk}]
                """)

                # Line Graph
                fig, ax = plt.subplots()
                ax.plot(class_names, preds, marker='o')
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Confidence")
                st.pyplot(fig)

                # Save image temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)
                    temp_image_path = tmp_file.name

                # PDF
                def generate_pdf():
                    pdf = FPDF()
                    pdf.add_page()

                    pdf.set_font("Arial", "B", 18)
                    pdf.cell(200, 10, "AI Diagnostic Center", ln=True, align="C")

                    pdf.set_font("Arial", "", 12)
                    pdf.cell(200, 8, "Lung Cancer Detection Report", ln=True, align="C")

                    pdf.ln(5)
                    pdf.cell(200, 0, "", ln=True, border="T")

                    pdf.ln(8)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 10, "Patient Information", ln=True)

                    pdf.set_font("Arial", "", 12)
                    pdf.cell(100, 8, f"Name: {name}", ln=False)
                    pdf.cell(100, 8, f"Age: {age}", ln=True)
                    pdf.cell(100, 8, f"Sex: {sex}", ln=False)
                    pdf.cell(100, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)

                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 10, "Diagnosis Summary", ln=True)

                    if risk == "High":
                        pdf.set_text_color(255, 0, 0)
                    elif risk == "Medium":
                        pdf.set_text_color(255, 140, 0)
                    else:
                        pdf.set_text_color(0, 128, 0)

                    pdf.cell(200, 10, f"Prediction: {label}", ln=True)
                    pdf.cell(200, 10, f"Risk Level: {risk}", ln=True)

                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(200, 8, f"Confidence: {confidence:.2f}", ln=True)

                    pdf.ln(5)
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(200, 10, "CT Scan Image", ln=True)
                    pdf.image(temp_image_path, x=30, w=150)

                    pdf.ln(85)
                    pdf.set_font("Arial", "I", 10)
                    pdf.set_text_color(100, 100, 100)

                    pdf.multi_cell(
                        0, 6,
                        "Disclaimer: This report is AI-generated and for educational use only. Consult a doctor."
                    )

                    pdf_path = "report.pdf"
                    pdf.output(pdf_path)
                    return pdf_path

                pdf_path = generate_pdf()

                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download Report",
                        f,
                        file_name="Lung_Cancer_Report.pdf"
                    )

                os.remove(temp_image_path)