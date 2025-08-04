import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

# ---------- 1. Load your PyTorch model ----------
# Make sure this matches the architecture used when training
from model import CNNLSTMModel  # Replace with actual model class

model = CNNLSTMModel()
model.load_state_dict(torch.load("gesture_model.pth", map_location=torch.device('cpu')))
model.eval()

# ---------- 2. Export PyTorch model to ONNX ----------
dummy_input = torch.randn(1, 200, 6)   # Update input shape as needed
onnx_model_path = "gesture_model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path,
                  input_names=['input'], output_names=['output'],
                  opset_version=11)

# ---------- 3. Convert ONNX to TensorFlow ----------
onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_model_path = "tf_model"
if not os.path.exists(tf_model_path):
    os.makedirs(tf_model_path)
tf_rep.export_graph(tf_model_path)

# ---------- 4. Convert TensorFlow to TFLite ----------
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
# ✅ Add supported ops fallback
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: quantization
tflite_model = converter.convert()

# ---------- 5. Save the TFLite model ----------
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)

print("🎉 All done! 'gesture_model.tflite' created successfully.")
