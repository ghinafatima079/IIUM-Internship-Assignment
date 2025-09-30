import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

class FaceDetectionApp:
    def __init__(self, root, input_dir="data/input/"):
        self.root = root
        self.root.title("Face Detection & Feature Localization")

        # Folder containing input images
        self.input_dir = input_dir
        # Collect all supported image files
        self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        self.current_index = 0

        # If no images found, stop the app
        if not self.image_files:
            messagebox.showerror("Error", f"No images found in {input_dir}")
            root.destroy()
            return

        # Navigation buttons
        self.prev_btn = tk.Button(root, text="⬅ Previous", command=self.show_previous)
        self.prev_btn.pack(side="left", padx=5)

        self.next_btn = tk.Button(root, text="Next ➡", command=self.show_next)
        self.next_btn.pack(side="right", padx=5)

        # Save output button (disabled until an image is processed)
        self.save_btn = tk.Button(root, text="💾 Save Output", command=self.save_output, state=tk.DISABLED)
        self.save_btn.pack(pady=5)

        # Input and output labels/canvases
        self.input_label = tk.Label(root, text="Input Image")
        self.input_label.pack()
        self.input_canvas = tk.Label(root)
        self.input_canvas.pack(side="left", padx=10)

        self.output_label = tk.Label(root, text="Output Image")
        self.output_label.pack()
        self.output_canvas = tk.Label(root)
        self.output_canvas.pack(side="right", padx=10)

        # Store processed image
        self.processed_image = None
        # Show first image on startup
        self.show_image()

    def show_image(self):
        # Load current image
        img_path = os.path.join(self.input_dir, self.image_files[self.current_index])
        image = cv2.imread(img_path)

        if image is None:
            messagebox.showerror("Error", f"Could not open {img_path}")
            return

        # Show original image
        self.display_image(image, self.input_canvas, max_size=300)

        # Processed output (faces + features marked)
        processed = self.detect_features(image.copy())
        self.processed_image = processed

        # Show processed image
        self.display_image(processed, self.output_canvas, max_size=300)

        # Enable save button
        self.save_btn.config(state=tk.NORMAL)

        # Update window title with image filename
        self.root.title(f"Face Detection – {self.image_files[self.current_index]}")

    def show_next(self):
        # Go to next image
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image()

    def show_previous(self):
        # Go to previous image
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def detect_features(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # ROI for face
            roi_gray = gray[y:y+h, x:x+w]

            # Use upper half of face for eye detection
            upper_half = roi_gray[0:h//2, :]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(upper_half)

            # Mark first 2 eyes with green dots
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2
                cv2.circle(image, (eye_center_x, eye_center_y), 3, (0, 255, 0), -1)

            # Approximate nose tip (red dot at 60% face height, center x)
            nose_x, nose_y = x + w // 2, y + int(h * 0.6)
            cv2.circle(image, (nose_x, nose_y), 3, (0, 0, 255), -1)

        return image

    def display_image(self, cv_img, canvas, max_size=300):
        # Convert OpenCV image to RGB
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((max_size, max_size))  # Resize for display
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.config(image=tk_img)
        canvas.image = tk_img  # keep reference to avoid garbage collection

    def save_output(self):
        # Save processed image
        if self.processed_image is None:
            messagebox.showerror("Error", "No output image to save!")
            return

        # Create output folder if missing
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        # Save with "_output" suffix
        input_filename = self.image_files[self.current_index]
        name, ext = os.path.splitext(input_filename)
        output_filename = f"{name}_output{ext}"
        save_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(save_path, self.processed_image)
        messagebox.showinfo("Saved", f"✅ Output saved at {save_path}")

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root, input_dir="q3/data/input/")  # change folder here
    root.mainloop()
    # GUI loop runs until window closed
