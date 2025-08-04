import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="gesture_model.tflite")
interpreter.allocate_tensors()

print("Input details:")
print(interpreter.get_input_details())

print("Output details:")
print(interpreter.get_output_details())
