from tkinter import Tk, Frame, Label, filedialog, messagebox, Canvas, Text, ttk, StringVar
from PIL import Image, ImageTk
import threading
import os
import cv2
from ultralytics import YOLO

MODEL_PATH = "model/Pothole_Model.pt"

class ObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Pothole Detection App - Group 1 F2F")
        master.geometry("1200x700")
        master.config(bg="#181A20")

        # Rounded buttons and progress bar
        style = ttk.Style(master)
        style.theme_use("clam")
        style.configure("Rounded.TButton", font=("Segoe UI", 14, "bold"), padding=10, borderwidth=0, relief="flat", foreground="#000", background="#fff")
        style.map("Rounded.TButton",
                  background=[("active", "#e0e0e0"), ("disabled", "#cccccc")])
        style.configure("TProgressbar", thickness=20, borderwidth=0, troughcolor="#23272f", background="#2196F3", relief="flat")

        self.model = YOLO(MODEL_PATH)
        self.file_path = None
        self.result_image = None
        self.result_video_path = None
        self.total_frames = 1
        self.stop_processing = False  # For stopping processing
        self.previewing_video = False  # For stopping video preview

        # Main layout frames
        self.container = Frame(master, bg="#181A20")
        self.container.pack(expand=True, fill="both")

        self.left_frame = Frame(self.container, bg="#23272f", bd=0)
        self.left_frame.pack(side="left", expand=True, fill="both", padx=20, pady=20)

        self.right_frame = Frame(self.container, bg="#23272f", bd=0, width=320)
        self.right_frame.pack(side="right", fill="y", padx=10, pady=20)

        # Preview area
        self.status_overlay = Label(self.left_frame, text="Live Processing", bg="#23272f", fg="#bfc9d1", font=("Segoe UI", 16, "bold"))
        self.status_overlay.pack(padx=20, pady=(10, 0), anchor="n")
        self.preview_canvas = Canvas(self.left_frame, bg="#23272f", bd=0, highlightthickness=0)
        self.preview_canvas.pack(expand=True, fill="both", padx=20, pady=(10, 10))
        self.preview_canvas.bind("<Configure>", self.on_canvas_resize)

        # Controls
        self.controls_frame = Frame(self.left_frame, bg="#23272f")
        self.controls_frame.pack(fill="x", padx=20, pady=10)
        self.upload_button = ttk.Button(self.controls_frame, text="Upload", command=self.upload_file, style="Rounded.TButton")
        self.upload_button.pack(side="left", padx=20)
        self.process_button = ttk.Button(self.controls_frame, text="Process", command=self.process_file, style="Rounded.TButton")
        self.process_button.pack(side="left", padx=20)
        self.stop_button = ttk.Button(self.controls_frame, text="Stop", command=self.stop_process, style="Rounded.TButton", state="disabled")
        self.stop_button.pack(side="left", padx=20)
        self.export_button = ttk.Button(self.controls_frame, text="Export", command=self.export_result, style="Rounded.TButton")
        self.export_button.pack(side="left", padx=20)
        self.preview_video_button = ttk.Button(self.controls_frame, text="Preview Video", command=self.preview_video, style="Rounded.TButton", state="disabled")
        self.preview_video_button.pack(side="left", padx=20)
        self.stop_preview_button = ttk.Button(self.controls_frame, text="Stop Preview", command=self.stop_preview_video, style="Rounded.TButton", state="disabled")
        self.stop_preview_button.pack(side="left", padx=20)

        # Progress bar & status
        self.progress_frame = Frame(self.left_frame, bg="#23272f")
        self.progress_frame.pack(fill="x", padx=20, pady=10)
        self.progress = ttk.Progressbar(self.progress_frame, orient="horizontal", mode="determinate", style="TProgressbar")
        self.progress.pack(side="left", fill="x", expand=True, pady=10)
        self.progress_var = StringVar(value="0%")
        self.progress_label = Label(self.progress_frame, textvariable=self.progress_var, bg="#23272f", fg="#bfc9d1", font=("Segoe UI", 12, "bold"))
        self.progress_label.pack(side="left", padx=10)
        self.status_label = Label(self.progress_frame, text="Processing Status", bg="#23272f", fg="#bfc9d1", font=("Segoe UI", 12))
        self.status_label.pack(anchor="w")

        # Processing logs (right panel) - NO SCROLLBAR
        Label(self.right_frame, text="Processing Logs", bg="#23272f", fg="#bfc9d1", font=("Segoe UI", 14, "bold")).pack(pady=10)
        self.log_text = Text(self.right_frame, bg="#181A20", fg="#bfc9d1", font=("Segoe UI", 11), bd=0, wrap="word")
        self.log_text.pack(expand=True, fill="both", padx=10, pady=10)
        self.log_text.config(state="disabled")

        # Internal state
        self.preview_imgtk = None

    def log(self, message, warning=False):
        self.log_text.config(state="normal")
        tag = "warn" if warning else "info"
        self.log_text.insert("end", f"{message}\n", tag)
        self.log_text.tag_config("info", foreground="#bfc9d1")
        self.log_text.tag_config("warn", foreground="#e67e22")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def upload_file(self):
        self.previewing_video = False  # Stop any running video preview
        self.stop_preview_button.config(state="disabled")
        self.file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png"), ("Video files", "*.mp4;*.avi")]
        )
        if self.file_path:
            self.status_label.config(text=f"Selected: {os.path.basename(self.file_path)}")
            self.result_image = None
            self.result_video_path = None
            self.preview_canvas.delete("all")
            self.preview_video_button.config(state="disabled")
            self.stop_preview_button.config(state="disabled")
            ext = os.path.splitext(self.file_path)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                img = cv2.imread(self.file_path)
                if img is not None:
                    self.display_image(img)
            self.log(f"Media \"{os.path.basename(self.file_path)}\" uploaded.")
        else:
            messagebox.showerror("Error", "Invalid file format. Please upload an image or video.")

    def process_file(self):
        self.previewing_video = False  # Stop any running video preview
        self.stop_preview_button.config(state="disabled")
        if self.file_path:
            self.status_label.config(text="Initializing object detection model...")
            self.progress["value"] = 0
            self.progress_var.set("0%")
            self.upload_button.config(state="disabled")
            self.process_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.preview_video_button.config(state="disabled")
            self.stop_preview_button.config(state="disabled")
            self.stop_processing = False
            threading.Thread(target=self.run_processing).start()
        else:
            messagebox.showwarning("Warning", "Please upload a file first.")

    def stop_process(self):
        self.stop_processing = True
        self.previewing_video = False  # Stop video preview if running
        self.stop_preview_button.config(state="disabled")
        self.status_label.config(text="Processing stopped by user.")
        self.log("Processing stopped by user.", warning=True)
        self.upload_button.config(state="normal")
        self.process_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.preview_video_button.config(state="normal" if self.result_video_path else "disabled")

    def run_processing(self):
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            self.log("Processing image...")
            results = self.model.predict(source=self.file_path, save=False)
            annotated = results[0].plot()
            self.result_image = annotated
            self.display_image(annotated)
            self.status_label.config(text="Processing complete.")
            self.progress["value"] = 100
            self.progress_var.set("100%")
            self.log("Image processed.")
            self.preview_video_button.config(state="disabled")
            self.stop_preview_button.config(state="disabled")
        elif ext in [".mp4", ".avi"]:
            cap = cv2.VideoCapture(self.file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out_path = "processed_video.mp4"
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            frame_count = 0
            self.log(f"Processing video ({total_frames} frames)...")
            while True:
                if self.stop_processing:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                results = self.model.predict(source=frame, save=False)
                annotated = results[0].plot()
                out.write(annotated)
                frame_count += 1
                if frame_count % max(1, total_frames // 30) == 0:
                    self.result_image = annotated
                    self.display_image(annotated)
                percent = int((frame_count / total_frames) * 100)
                self.progress["value"] = percent
                self.progress_var.set(f"{percent}%")
                self.status_label.config(text=f"Analyzing frame {frame_count}/{total_frames} - Detecting objects...")
                self.master.update_idletasks()
                if frame_count % 50 == 0:
                    self.log(f"Processing frame {frame_count}/{total_frames}.")
            cap.release()
            out.release()
            if self.stop_processing:
                self.status_label.config(text="Processing stopped by user.")
                self.progress_var.set("0%")
                self.log("Video processing stopped by user.", warning=True)
                self.preview_video_button.config(state="normal" if os.path.exists(out_path) else "disabled")
                self.stop_preview_button.config(state="disabled")
            else:
                self.result_video_path = out_path
                self.status_label.config(text="Processing complete.")
                self.progress["value"] = 100
                self.progress_var.set("100%")
                self.log("Video processed.")
                self.preview_video_button.config(state="normal")
                self.stop_preview_button.config(state="normal")
        else:
            self.status_label.config(text="Unsupported file type.")
            self.log("Unsupported file type.", warning=True)
            self.preview_video_button.config(state="disabled")
            self.stop_preview_button.config(state="disabled")
        self.upload_button.config(state="normal")
        self.process_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def display_image(self, img_cv):
        # Get current canvas size
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 760, 420 # Default size before rendering
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((canvas_width, canvas_height))
        self.preview_imgtk = ImageTk.PhotoImage(img_pil)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_width//2, canvas_height//2, anchor="center", image=self.preview_imgtk)

    def on_canvas_resize(self, event):
        # Redraw preview image to fit new canvas size
        if self.result_image is not None:
            self.display_image(self.result_image)

    def preview_video(self):
        if self.result_video_path and os.path.exists(self.result_video_path):
            self.previewing_video = True
            self.stop_preview_button.config(state="normal")
            cap = cv2.VideoCapture(self.result_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            while cap.isOpened() and self.previewing_video:
                ret, frame = cap.read()
                if not ret or not self.previewing_video:
                    break
                self.display_image(frame)
                self.master.update()
                cv2.waitKey(int(1000 / max(1, fps)))
            cap.release()
            self.status_label.config(text="Video preview finished.")
            self.stop_preview_button.config(state="disabled")
            self.previewing_video = False
        else:
            messagebox.showwarning("Preview", "No processed video to preview.")

    def stop_preview_video(self):
        self.previewing_video = False
        self.stop_preview_button.config(state="disabled")
        self.status_label.config(text="Video preview stopped.")

    def export_result(self):
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"] and self.result_image is not None:
            export_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if export_path:
                cv2.imwrite(export_path, self.result_image)
                messagebox.showinfo("Export", f"Image exported to {export_path}")
                self.log(f"Image exported to {export_path}.")
        elif ext in [".mp4", ".avi"] and self.result_video_path is not None and os.path.exists(self.result_video_path):
            export_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
            if export_path:
                os.rename(self.result_video_path, export_path)
                messagebox.showinfo("Export", f"Video exported to {export_path}")
                self.log(f"Video exported to {export_path}.")
        else:
            messagebox.showwarning("Export", "No processed result to export.")

if __name__ == "__main__":
    root = Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()