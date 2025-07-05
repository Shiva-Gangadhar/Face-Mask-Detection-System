# 😷 Real-Time Face Mask Detection System

This project uses a deep learning model to detect whether a person is wearing a face mask or not in **real time** using a webcam feed. It can be deployed in public places such as offices, malls, hospitals, etc., to ensure safety and compliance.

## 📌 Features

- Real-time face detection and classification
- Uses Keras-trained CNN model
- Uses OpenCV for video capture and Haarcascade for face detection
- Clean UI with bounding boxes and prediction labels
- High accuracy (~96%) on validation set

---

## 📂 Project Structure

```
├── mask_model.h5                    # Trained model (Keras)
├── RealTime_Face_Mask.ipynb         # Jupyter notebook for real-time detection
├── Face Mask Detection.ipynb        # Model training notebook (Google Colab)
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
```

---

## 🧠 Model Training (Google Colab)

- Dataset: Custom dataset with `with_mask` and `without_mask` folders
- Image size: 128x128
- Model: CNN with Conv2D, BatchNorm, MaxPooling, Dense, Dropout
- Accuracy: **~96%** on test data

### ✨ Model Architecture (Keras)
```python
Conv2D(32, kernel_size=19) → BatchNorm → MaxPool
Conv2D(64, kernel_size=19) → BatchNorm → MaxPool
Flatten → Dense(128) → Dropout(0.5) → Dense(64) → Dropout(0.5) → Dense(2)
```

---

## 💻 How to Run Real-Time Detection

### 🛠️ 1. Install Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow opencv-python numpy
```

### 📸 2. Run the Notebook

Open the notebook:
```bash
RealTime_Face_Mask.ipynb
```

Make sure your webcam is connected and the file `mask_model.h5` is in the same directory.

Press **`q`** to quit the webcam window.

---

## 📊 Output

- Green box → Person **with mask**
- Red box → Person **without mask**
- Label includes class + confidence score

---

## 🧪 Known Issues & Fixes

- If incorrect predictions occur:
  - Improve face crop quality
  - Use RGB instead of BGR
  - Apply threshold tuning (`if prob[1] > 0.9`)
  - Add more diverse images to the dataset

---

## 🔄 Future Improvements

- Deploy on edge devices (e.g., Raspberry Pi, Jetson Nano)
- Integrate attendance or access control systems
- Use Mediapipe or YOLOv8 for better face detection
- Add GUI or alert system for non-compliance

---

## 🤝 Credits

- Developed by: **Shiv Gangadhar**
- Libraries used: TensorFlow/Keras, OpenCV, NumPy
- Dataset: Combined open-source and custom-curated images

---

## 🛡️ License

This project is open source under the [MIT License](LICENSE).
