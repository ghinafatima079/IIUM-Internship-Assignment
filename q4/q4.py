import cv2  # OpenCV library for computer vision tasks
import datetime  # Used to generate timestamp for recorded files

# ===============================
# Q4. Face Blurring in Video Feeds
# ===============================

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam (0 = default webcam, change to RTSP/HTTP link for CCTV feeds)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("❌ Cannot open camera")  # Print error if camera is not available
    exit()  # Exit program

# Set up video writer parameters (only used when recording starts)
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec (XVID format)
out = None  # Placeholder for VideoWriter object
recording = False  # Flag to track recording state

print("🎥 Press 'r' to start/stop recording, 'q' to quit.")

# Infinite loop to read frames from webcam
while True:
    ret, frame = cap.read()  # Read one frame from the camera
    if not ret:  # If frame is not captured properly
        print("❌ Failed to grab frame")  # Print error
        break  # Exit loop

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale for face detection

    # Detect faces → returns list of rectangles (x, y, w, h)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]  # Extract region of interest (ROI) = face area
        roi_blur = cv2.GaussianBlur(roi, (99, 99), 30)  # Apply Gaussian blur to ROI
        frame[y:y+h, x:x+w] = roi_blur  # Replace original face area with blurred version

    cv2.imshow("Blurred Face Feed", frame)  # Show video frame with blurred faces

    # If recording mode is ON and writer is initialized → save current frame
    if recording and out is not None:
        out.write(frame)

    # Wait for 1ms for a key press (bitwise AND ensures correct key detection)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):  # If "q" is pressed → quit program
        break
    elif key == ord("r"):  # If "r" is pressed → toggle recording
        import os  # Import OS module for file handling
        if not recording:
            # Start recording
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Current time for unique filename
            output_dir = os.path.join(os.path.dirname(__file__), "output")  # Output folder path
            os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn’t exist
            out_path = os.path.join(output_dir, f"blurred_{timestamp}.avi")  # Full file path for video
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))  # Initialize writer
            recording = True  # Set flag ON
            print(f"🔴 Recording started... Saving to {out_path}")  # Print status
        else:
            # Stop recording
            recording = False  # Set flag OFF
            out.release()  # Release video writer resource
            out = None  # Reset writer variable
            print("✅ Recording saved and stopped.")  # Print status

# Cleanup section (runs after quitting loop)
cap.release()  # Release webcam resource
if out is not None:  # If writer was still active
    out.release()  # Release it
cv2.destroyAllWindows()  # Close all OpenCV windows
