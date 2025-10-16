import cv2, os, tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

class FaceDetectionApp:
    def __init__(self, root, input_dir="data/input/"):
        self.root = root
        self.root.title("Face Detection")
        self.input_dir = input_dir
        self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
        if not self.image_files:
            messagebox.showerror("Error", f"No images in {input_dir}"); root.destroy(); return
        self.current_index = 0
        self.processed_image = None

        # GUI
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        self.prev_btn = tk.Button(btn_frame, text="⬅ Previous", command=lambda: self.change_image(-1)); self.prev_btn.pack(side="left", padx=5)
        self.next_btn = tk.Button(btn_frame, text="Next ➡", command=lambda: self.change_image(1)); self.next_btn.pack(side="right", padx=5)
        self.save_btn = tk.Button(root, text="💾 Save Output", command=self.save_output, state=tk.DISABLED); self.save_btn.pack()

        self.input_canvas = tk.Label(root); self.input_canvas.pack(side="left", padx=10)
        self.output_canvas = tk.Label(root); self.output_canvas.pack(side="right", padx=10)

        self.show_image()

    def change_image(self, offset):
        idx = self.current_index + offset
        if 0 <= idx < len(self.image_files):
            self.current_index = idx
            self.show_image()

    def show_image(self):
        path = os.path.join(self.input_dir, self.image_files[self.current_index])
        img = cv2.imread(path)
        if img is None: messagebox.showerror("Error", f"Cannot open {path}"); return

        self.display_image(img, self.input_canvas)
        self.processed_image = self.detect_features(img.copy())
        self.display_image(self.processed_image, self.output_canvas)
        self.save_btn.config(state=tk.NORMAL)
        self.root.title(f"Face Detection – {self.image_files[self.current_index]}")

    def detect_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.3, 5):
            roi_gray = gray[y:y+h//2, x:x+w]
            for (ex, ey, ew, eh) in eye_cascade.detectMultiScale(roi_gray)[:2]:
                cv2.circle(img, (x+ex+ew//2, y+ey+eh//2), 3, (0,255,0), -1)
            cv2.circle(img, (x+w//2, y+int(h*0.6)), 3, (0,0,255), -1)
        return img

    def display_image(self, cv_img, canvas, max_size=300):
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        pil_img.thumbnail((max_size,max_size))
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.config(image=tk_img); canvas.image = tk_img

    def save_output(self):
        if self.processed_image is None: messagebox.showerror("Error", "No output"); return
        out_dir = os.path.join(os.path.dirname(__file__), "Q3_output")
        os.makedirs(out_dir, exist_ok=True)
        name, ext = os.path.splitext(self.image_files[self.current_index])
        save_path = os.path.join(out_dir, f"{name}_output{ext}")
        cv2.imwrite(save_path, self.processed_image)
        messagebox.showinfo("Saved", f"✅ Saved at {save_path}")

if __name__=="__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root, input_dir="q3/Q3_input_images/")
    root.mainloop()
