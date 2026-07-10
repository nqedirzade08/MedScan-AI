# MedScan AI

MedScan AI is an AI-powered healthcare platform that assists users and healthcare professionals by analyzing medical images and laboratory reports. The platform provides disease risk assessment, AI-generated medical insights, easy-to-understand explanations, and personalized health recommendations.

## ✨ Features
- 🩻 Chest X-ray Analysis *(Pneumonia, Cardiomegaly, Pleural Effusion)*
- 🦴 Bone X-ray Analysis *(Fracture Detection)*
- 🧠 Brain MRI Analysis *(Glioma, Meningioma, Pituitary Tumor, No Tumor)*
- 🩸 Blood Test Analysis *(CBC – Rule-based Expert System)*
- 🧪 Urine Test Analysis *(Rule-based Expert System)*
- 🩹 Skin Disease Analysis *(23 Disease Categories)*
- 🦴 Scoliosis Analysis *(X-ray & Back Photo-based)*
- 🔥 AI Attention Heatmap *(Grad-CAM Visualization)*
- 📄 AI Medical Report Generation
- 📊 Disease Risk Assessment
- ⚖️ Before & After Analysis Comparison
- 🤖 AI Health Assistant *(Rule-based Chatbot)*
- 📋 Medical History & Analysis Tracking

## 🚀 Tech Stack
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Backend:** FastAPI (Python)
- **AI/ML:** PyTorch, EfficientNet-B0, EfficientNet-B2, Transfer Learning, Grad-CAM
- **Database:** SQLite (SQLAlchemy ORM)
- **Authentication:** JWT (python-jose)
- **Model Training:** Kaggle (Tesla T4 GPU)
- **AI Models:** 6 Custom-Trained EfficientNet Models

## 📊 Model Performance
| Model | Architecture | Accuracy | Dataset Size |
|--------|--------------|----------|--------------|
| Chest X-ray | EfficientNet-B0 | **76.0%** | 3,322 Images |
| Bone Fracture | EfficientNet-B0 | **95.7.0%** | 10,581 Images |
| Brain MRI | EfficientNet-B0 | **96.1%** | 7,200 Images |
| Skin Disease | EfficientNet-B2 | **71.2%** | 19,559 Images |
| Scoliosis (X-ray) | EfficientNet-B0 | **98.3%** | 338 Images |
| Scoliosis (Back Photo) | EfficientNet-B0 | **97.5%** | 323 Images |

## 🎯 Vision
Our mission is to make AI-powered healthcare more accessible by providing fast, accurate, and user-friendly medical analysis tools that support healthcare professionals rather than replace them.

## ⚠️ Disclaimer
MedScan AI is intended for educational, research, and informational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional before making medical decisions.
