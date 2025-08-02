
# 😷 Real-Time Face Mask Detection System (PyTorch)

This project uses a PyTorch deep learning model to detect whether a person is wearing a face mask or not in **real time** using a webcam feed. It is built using a custom CNN-to-MLP architecture and can be deployed in public environments like offices, malls, and hospitals to ensure safety and mask compliance.

---

## 📌 Features

- Real-time face detection and classification via webcam
- Custom PyTorch model using CNN feature extractor and MLP classifier
- Uses OpenCV for video stream and Haarcascade for face detection
- Clean UI with live bounding boxes and labels
- High accuracy (~96%) on validation dataset

---

## 📂 Project Structure

```
├── cnn_to_mlp.py            # CNNtoMLP model architecture (PyTorch)
├── model_weights.pth              # Trained model weights
├── Realtime_Pytorch.ipy          # Real-time mask detection scriptnb
├── Face_Mask_Detection_Pytorch.ipynb                 # Model training script
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 🧠 Model Training (PyTorch)

- **Dataset**: Custom dataset with `with_mask` and `without_mask` image folders
- **Image size**: 128x128
- **Model**: CNNtoMLP  
- **Accuracy**: Achieved ~96% accuracy on validation set

---

### 🔧 Hyperparameter Tuning

To optimize the performance of the face mask classification model, hyperparameter tuning was performed to find the best configuration for the model. This improved the model’s accuracy and generalization ability.

**Tuned Parameters:**
- `num_hidden_layers`: Number of fully connected layers in the MLP (e.g., 2, 3, 4)
- `neurons_per_layer`: Number of neurons per hidden layer (e.g., 64, 112, 128)
- `dropout_rate`: Dropout value to prevent overfitting (e.g., 0.3, 0.5)
- `learning_rate`: Learning rate for the optimizer (e.g., 0.001, 0.0005)
- `batch_size`: Number of samples per gradient update (e.g., 16, 32)
- `optimizer`: Optimizer used during training (e.g., Adam, SGD)

**Tuning Approach:**
- Various combinations of hyperparameters were evaluated using validation accuracy as the primary metric.
- Each model was trained on the training set and evaluated on the validation set.
- The configuration with the highest validation accuracy was selected and the corresponding model was saved as `.pth`.

**Best Hyperparameters Found:**
- `num_hidden_layers`: 2
- `neurons_per_layer`: 112
- `dropout_rate`: 0.1
- `learning_rate`: 0.001
- `optimizer`: RMSprop

> ⚠️ Note: Accuracy obtained during training and after loading the saved model may slightly differ due to randomness in initialization, dropout, or data shuffling. However, the saved `.pth` file contains the trained model with the best performance.


## ✨ Model Architecture (CNNtoMLP - PyTorch)

```python
Conv2D → BatchNorm → MaxPool  
Conv2D → BatchNorm → MaxPool  
Flatten → Linear(112) → ReLU → Dropout(0.1)  
Linear(112) → ReLU → Dropout(0.1)  
Linear(2) → logits
```

---

## 💻 How to Run Real-Time Detection

### 🛠️ 1. Install Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision opencv-python numpy
```

### 📸 2. Run the Detection Script

```bash
python Realtime_Pytorch.ipynb
```

- Make sure your webcam is connected
- Ensure `model_weights.pth` is in the same folder
- Press `q` to quit the webcam window

---

## 📊 Output

- ✅ Green Box → Person with **Mask**
- ❌ Red Box → Person **without Mask**
- Label shows prediction class and confidence score

---

## 🧪 Known Issues & Fixes

- False predictions?
  - Improve face cropping logic
  - Normalize pixel values or convert BGR to RGB
  - Add a confidence threshold (e.g., `if prob[1] > 0.9`)
  - Train on more diverse faces and mask types

---

## 🔄 Future Improvements

- Deploy to edge devices (e.g., Raspberry Pi, Jetson Nano)
- Add GUI or buzzer alert system for `no-mask` cases
- Use YOLOv8, Mediapipe, or DLib for better face detection
- Export to ONNX / TorchScript for mobile deployment
- Integrate access control systems for real-world applications

---

## 🤝 Credits

- **Developed by**: Shiv Gangadhar  
- **Libraries Used**: PyTorch, OpenCV, NumPy  
- **Dataset**: Custom curated dataset from open-source and personal data

---

## 🛡️ License

This project is open source under the **MIT License**.
