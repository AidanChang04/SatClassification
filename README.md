# 🛰️ Satellite Image Classifier

AI-powered land use classification system using deep learning for satellite imagery analysis.

## 🌟 Features
- Classifies 10 different land use types
- FastAPI backend with TensorFlow model
- Modern responsive web interface
- Real-time image analysis

## 📋 Land Use Classes
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea/Lake

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload