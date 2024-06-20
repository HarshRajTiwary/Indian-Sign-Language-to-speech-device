import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:/Users/harsh/OneDrive/Desktop/Task2/temp_codes/tf_lite_quant_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the video capture object
vc = cv2.VideoCapture(0)

# Loop until the user presses the 'q' key
while True:
    # Capture a frame from the webcam
    ret, frame = vc.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to 40x40
    resized_frame = cv2.resize(gray_frame, (40, 40))

    # Normalize the pixel values
    normalized_frame = resized_frame / 127.5 - 1.0

    # Add a batch dimension and ensure it has the right shape
    input_data = np.expand_dims(normalized_frame, axis=(0, -1)).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the interpreter
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_class = np.argmax(output_data)

    # Print the predicted class
    print("Predicted gesture:", predicted_class)

    # Display the frame
    cv2.imshow('Webcam_Gesture_Recognition', gray_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
vc.release()

# Close all windows
cv2.destroyAllWindows()
