# 😷 Real-Time Face Mask Detection using PyTorch

A real-time face mask detection application using a CNN-MLP hybrid model implemented in PyTorch. This system classifies whether a person is wearing a mask or not through live webcam feed.

---

## 🚀 Features

- Real-time detection using your device's webcam.
- Lightweight model using CNN feature extraction and MLP classification.
- Visual output with OpenCV: displays "Mask" or "No Mask" on screen.
- Trained on custom or public dataset (e.g., Face Mask Detection dataset).

---

## 🧠 Model Architecture

The model used is a custom CNNtoMLP architecture which:
- Extracts features from facial images using CNN layers.
- Passes flattened features through multiple MLP (fully-connected) layers.
- Uses `ReLU`, `Dropout`, and `BatchNorm` for regularization and better generalization.

You can find the model defined in `model_definition.py`.

---

## 🗂️ Project Structure

face-mask-detector/
│
├── model_definition.py # Contains CNNtoMLP model architecture
├── model_weights.pth # Trained PyTorch model weights
├── detect_from_camera.py # Script to run real-time detection
├── train_model.py # Optional: your training pipeline
├── requirements.txt # Required Python packages
├── README.md # Project documentation

yaml
Copy
Edit

---

## 🔧 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-mask-detector.git
cd face-mask-detector
2. Install dependencies
Create a virtual environment (optional) and install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
3. Download Trained Weights
Place your trained model file (model_weights.pth) inside the root folder if not already there.

🎥 Running Real-Time Detection
bash
Copy
Edit
python detect_from_camera.py
This will:

Launch your webcam.

Detect faces using Haar Cascades (or a different detector if implemented).

Classify each face as Mask or No Mask.

Overlay predictions on the video frame.

🧪 Evaluation (Optional)
If you want to evaluate your model on a test dataset:

python
Copy
Edit
# Inside test_model.py or notebook
accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
📦 Dependencies
Python 3.8+

PyTorch

OpenCV

NumPy

torchvision

You can install them via:

bash
Copy
Edit
pip install torch torchvision opencv-python numpy
📸 Example Output


✅ To-Do
 Improve face detection with SSD or MTCNN

 Add GUI using Streamlit or PyQt

 Export to ONNX or TensorFlow Lite for mobile use

📝 License
This project is licensed under the MIT License. See LICENSE for details.

🙌 Acknowledgements
PyTorch

OpenCV

Datasets like Face Mask Detection
