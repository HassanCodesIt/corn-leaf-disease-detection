![image](https://github.com/user-attachments/assets/ec30c191-95c0-4ef9-a69a-0ddb5ac4f63a)


---

# 🌽 Corn Leaf Disease Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python\&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-green?logo=github\&logoColor=white)
![Roboflow](https://img.shields.io/badge/Dataset-Roboflow-orange?logo=data\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Detect and classify corn leaf diseases in real-time using **YOLOv8n**—identifying Brown Spot, Corn Rust, Corn Smut, Downy Mildew, Grey Leaf Spot, Leaf Blight, and healthy leaves.

---

## ⚙️ What’s Inside

* **`corn_leaf_diseases.ipynb`** – Comprehensive training notebook using Roboflow‑sourced data.
* **`corn_dataset.txt`** – YOLO‑formatted dataset info: 4,924 images, 7 classes + healthy, train/val/test splits, class IDs.
* **`corn_Leaf_model.pt`** – Trained YOLOv8n model weights ready for inference.
* **`test_model_with_image.py`** – Script to detect diseases on a static image: loads model, runs detection, displays, and saves annotated image.
* **`test_model_with_webcam.py`** – Real-time webcam inference demo.
* **`app/`** – FastAPI web application for corn leaf disease detection.
* **`static/`** – JavaScript frontend for uploading images via the browser.

---

## 🎯 Features

* Detects **7 corn leaf classes** + healthy class with bounding boxes.
* Quick and lightweight via YOLOv8n for real-time use.
* Full training pipeline included: from data download to model tuning.
* Inference scripts streamline testing on images or webcams.
* **Web application** with FastAPI backend and JavaScript frontend for easy image uploads.

---

## 🧑‍💻 Getting Started

### 1. Clone repo

```bash
git clone https://github.com/HassanCodesIt/corn-leaf-disease-detection.git  
cd corn-leaf-disease-detection  
```

### 2. Install dependencies

```bash
pip install ultralytics roboflow opencv-python-headless  
```

### 3. Download dataset via Roboflow (in notebook)

Inside `corn_leaf_diseases.ipynb`, you'll find:

```python
from roboflow import Roboflow  
rf = Roboflow(api_key="YOUR_API_KEY")  
project = rf.workspace("corn-leaf-disease").project("corn_disease-vsprz")  
version = project.version(4)  
dataset = version.download("yolov8")  
```

### 4. Train your own model

Open the notebook and run:

```bash
!yolo train model=yolov8n.pt data=path/to/data.yaml epochs=50 imgsz=640  
```

### 5. Inference on images

```bash
python test_model_with_image.py  
```

### 6. Real-time inference

```bash
python test_model_with_webcam.py  
```

### 7. Web Application

Run the FastAPI web application:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open your browser and navigate to `http://localhost:8000` to use the web interface.

---

## 🌐 Web Application

The project includes a modern web application built with **FastAPI** (backend) and **JavaScript** (frontend) that allows users to upload corn leaf images directly through their browser.

### Screenshot

![Corn Leaf Disease Detection Web App](https://github.com/user-attachments/assets/49477b2c-9ed4-46ef-b4ce-652349a18663)

### Features

* **Drag & Drop** – Simply drag and drop your corn leaf images onto the upload area.
* **Click to Browse** – Or click the button to select images from your device.
* **Real-time Detection** – Get instant disease detection results with confidence scores.
* **Annotated Images** – View the original image alongside the annotated detection result.
* **Disease Legend** – Easy-to-understand color-coded legend for all detectable diseases.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web frontend |
| `/health` | GET | Health check endpoint |
| `/predict` | POST | Returns JSON predictions for uploaded image |
| `/predict/annotated` | POST | Returns annotated image with bounding boxes |

---

## 🧠 Dataset Details

* **Images**: 4,924 labeled corn leaf images
* **Classes**:
  0 – Brown Spot
  1 – Corn Rust
  2 – Corn Smut
  3 – Downy Mildew
  4 – Grey Leaf Spot
  5 – Healthy
  6 – Leaf Blight
* **Split**: Train (90%), Val (7%), Test (3%)
* **Annotations**: YOLO bounding boxes with class IDs.

---

## 🧭 Why YOLOv8n?

YOLOv8n offers a compact model ideal for real-time applications. Your setup supports speedy detection of multiple diseases in the field.

---

## 🚀 Next Steps

* Tune hyperparameters for better mAP and recall.
* Add more disease categories or cross-crop detection.
* Deploy to edge devices or integrate into mobile/IoT apps.

---

## 🎓 Credits

* Dataset powered by Roboflow’s **corn\_disease-vsprz**, version 4.
* Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).

---

## 📝 License

This project is licensed under the **MIT License**.

---

### TL;DR

> Quick setup inference with YOLOv8n, train on a \~5K image dataset, detect 7 diseases + healthy leaves in real-time. Easy to scale, deploy & build on.

---


