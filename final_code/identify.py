import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import tkinter.ttk as ttk
import threading
import zipfile
import shutil
import os
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
VALID_IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_VIDEOS = {".mp4", ".avi", ".mov", ".mkv"}

class SmartLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Traffic/Street Auto-Labeler")
        self.root.geometry("700x900")
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.home() / "SmartLabelerOutput"))
        
        self.gate_model_path = tk.StringVar()
        self.traffic_model_path = tk.StringVar()
        self.normal_model_path = tk.StringVar()
        
        # State flags
        self.is_processing = False
        self.stop_requested = False
        
        self.build_ui()

    def build_ui(self):
        # --- 1. Model Configuration ---
        grp_models = tk.LabelFrame(self.root, text="1. Model Configuration", padx=10, pady=10)
        grp_models.pack(fill="x", padx=10, pady=5)
        
        self.create_file_select(grp_models, "Gate Model (.pt):", self.gate_model_path)
        self.create_file_select(grp_models, "Traffic Camera Model (.pt):", self.traffic_model_path)
        self.create_file_select(grp_models, "High Resolution Model (.pt):", self.normal_model_path)

        # --- 2. Input & Output ---
        grp_io = tk.LabelFrame(self.root, text="2. Input & Output", padx=10, pady=10)
        grp_io.pack(fill="x", padx=10, pady=5)
        
        tk.Label(grp_io, text="Input File (Image, Video, or Zip):").pack(anchor="w")
        frame_in = tk.Frame(grp_io)
        frame_in.pack(fill="x", pady=(0, 5))
        tk.Entry(frame_in, textvariable=self.input_path).pack(side="left", fill="x", expand=True)
        tk.Button(frame_in, text="Browse...", command=self.browse_input).pack(side="right", padx=5)
        
        tk.Label(grp_io, text="Output Directory:").pack(anchor="w")
        frame_out = tk.Frame(grp_io)
        frame_out.pack(fill="x", pady=(0, 5))
        tk.Entry(frame_out, textvariable=self.output_dir).pack(side="left", fill="x", expand=True)
        tk.Button(frame_out, text="Browse...", command=self.browse_output).pack(side="right", padx=5)

        # --- 3. Progress & Actions ---
        grp_prog = tk.LabelFrame(self.root, text="3. Progress", padx=10, pady=10)
        grp_prog.pack(fill="x", padx=10, pady=5)

        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(grp_prog, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=(0, 5))
        
        # Status Labels
        self.lbl_status = tk.Label(grp_prog, text="Ready", anchor="w", fg="blue")
        self.lbl_status.pack(fill="x")
        self.lbl_time = tk.Label(grp_prog, text="", anchor="w", fg="gray")
        self.lbl_time.pack(fill="x")

        # Buttons
        btn_frame = tk.Frame(grp_prog)
        btn_frame.pack(fill="x", pady=10)
        
        self.btn_run = tk.Button(btn_frame, text="START PROCESSING", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), command=self.start_processing_thread)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_stop = tk.Button(btn_frame, text="STOP", bg="#f44336", fg="white", font=("Arial", 12, "bold"), state="disabled", command=self.request_stop)
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # --- 4. Logs ---
        grp_log = tk.Frame(self.root, padx=10, pady=5)
        grp_log.pack(fill="both", expand=True)
        tk.Label(grp_log, text="Detailed Log:").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(grp_log, height=10, state='disabled')
        self.log_area.pack(fill="both", expand=True)

    # --- UI Helpers ---
    def create_file_select(self, parent, label_text, var):
        f = tk.Frame(parent)
        f.pack(fill="x", pady=2)
        # Increased width from 20 to 30 and added padx=(0, 10) for spacing
        tk.Label(f, text=label_text, width=30, anchor="w").pack(side="left", padx=(0, 10))
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)
        tk.Button(f, text="Select", command=lambda: self.browse_pt(var)).pack(side="right", padx=5)

    def log(self, msg):
        self.root.after(0, self._log_thread_safe, msg)

    def _log_thread_safe(self, msg):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def update_status(self, msg, progress=None, time_msg=""):
        self.root.after(0, self._status_thread_safe, msg, progress, time_msg)

    def _status_thread_safe(self, msg, progress, time_msg):
        self.lbl_status.config(text=msg)
        if progress is not None:
            self.progress_var.set(progress)
        if time_msg:
            self.lbl_time.config(text=time_msg)

    def browse_pt(self, var):
        f = filedialog.askopenfilename(filetypes=[("YOLO Models", "*.pt")])
        if f: var.set(f)

    def browse_input(self):
        f = filedialog.askopenfilename()
        if f: self.input_path.set(f)

    def browse_output(self):
        d = filedialog.askdirectory()
        if d: self.output_dir.set(d)

    # --- Control Logic ---
    def request_stop(self):
        if self.is_processing:
            self.stop_requested = True
            self.btn_stop.config(text="Stopping...", state="disabled")
            self.log("\n[STOP REQUESTED] Halting process (partial files will be kept)...")

    def start_processing_thread(self):
        if self.is_processing: return
        
        # Validation
        if not all([self.gate_model_path.get(), self.traffic_model_path.get(), self.normal_model_path.get()]):
            messagebox.showerror("Error", "Please select all three model weights.")
            return
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file.")
            return
            
        self.is_processing = True
        self.stop_requested = False
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal", text="STOP")
        self.log_area.config(state='normal')
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state='disabled')
        self.progress_var.set(0)
        
        threading.Thread(target=self.run_process, daemon=True).start()

    # --- Main Worker ---
    def run_process(self):
        try:
            self.log("--- Loading Models... ---")
            gate = YOLO(self.gate_model_path.get())
            det_traffic = YOLO(self.traffic_model_path.get())
            det_normal = YOLO(self.normal_model_path.get())
            
            input_p = Path(self.input_path.get())
            output_d = Path(self.output_dir.get())
            output_d.mkdir(parents=True, exist_ok=True)

            # Determine Workload
            files_to_process = []
            temp_dir = None

            if input_p.suffix.lower() == ".zip":
                self.update_status("Extracting ZIP...", 0)
                temp_dir = output_d / "temp_extracted"
                if temp_dir.exists(): shutil.rmtree(temp_dir)
                temp_dir.mkdir()
                
                with zipfile.ZipFile(input_p, 'r') as z:
                    z.extractall(temp_dir)
                
                for root, _, files in os.walk(temp_dir):
                    for f in files:
                        if Path(f).suffix.lower() in VALID_IMAGES | VALID_VIDEOS:
                            files_to_process.append(Path(root) / f)
            else:
                files_to_process = [input_p]

            total_files = len(files_to_process)
            self.log(f"Found {total_files} files to process.")
            
            start_time = time.time()
            
            for i, file_path in enumerate(files_to_process):
                if self.stop_requested: break
                
                # Initial status for this file
                pct = (i / total_files) * 100
                status_msg = f"Processing {i+1}/{total_files}: {file_path.name}"
                
                # For the very first file, we don't have past stats yet
                if i == 0:
                    time_str = "Estimating..."
                else:
                    # Fallback to file-based ETA if video streaming doesn't update it
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remain = avg_time * (total_files - i)
                    time_str = f"Elapsed: {int(elapsed)}s | ETA: {int(remain)}s"

                self.update_status(status_msg, pct, time_str)
                
                # Route and Process
                if file_path.suffix.lower() in VALID_IMAGES:
                    self.route_image(file_path, output_d, gate, det_traffic, det_normal)
                elif file_path.suffix.lower() in VALID_VIDEOS:
                    # Video streaming handles its own ETA updates for the frame loop
                    self.route_video_stream(file_path, output_d, gate, det_traffic, det_normal, 
                                            file_idx=i, total_files=total_files, status_prefix=status_msg)

            if self.stop_requested:
                self.log("\n[ABORTED] Processing stopped by user.")
                self.update_status("Stopped", 0)
            else:
                self.update_status("Completed", 100)
                self.log("\n=== ALL TASKS COMPLETED ===")
                messagebox.showinfo("Success", f"Processed {total_files} files.\nSaved to: {output_d}")

            # Cleanup temp
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)

        except Exception as e:
            self.log(f"\n[CRITICAL ERROR]: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.is_processing = False
            self.btn_run.config(state="normal")
            self.btn_stop.config(state="disabled")

    # --- Core Logic ---
    def route_image(self, img_path, out_dir, gate, det_traffic, det_normal):
        # Gatekeeper
        r = gate.predict(img_path, imgsz=224, verbose=False)[0]
        
        # Find class
        idx = 0
        for k, v in r.names.items():
            if "traffic" in v.lower() and "cam" in v.lower():
                idx = k; break
        
        p_traffic = float(r.probs.data[idx])
        model, domain = (det_traffic, "traffic_cam") if p_traffic >= 0.6 else (det_normal, "normal_view")
        
        self.log(f"  [{domain}] {img_path.name} ({p_traffic:.0%})")
        model.predict(img_path, save=True, project=str(out_dir), name=domain, exist_ok=True, verbose=False)

    def route_video_stream(self, video_path, out_dir, gate, det_traffic, det_normal, file_idx, total_files, status_prefix):
        """
        Uses manual streaming to allow Stop button and progress updates during long video.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1. Smart Sampling (first 15 frames)
        traffic_votes = 0
        samples_needed = 15
        step = max(int(fps), 1)
        
        self.log(f"  Sampling {video_path.name}...")
        
        for _ in range(samples_needed):
            if self.stop_requested: 
                cap.release()
                return
            
            grabbed = cap.grab()
            if not grabbed: break
            
            # Retrieve only on step
            ret, frame = cap.retrieve()
            if not ret: break
            
            # Predict
            r = gate.predict(frame, imgsz=224, verbose=False)[0]
            idx = 0
            for k, v in r.names.items():
                if "traffic" in v.lower() and "cam" in v.lower():
                    idx = k; break
            
            if float(r.probs.data[idx]) >= 0.6:
                traffic_votes += 1
                
            # Skip ahead
            for _ in range(step - 1): cap.grab()

        cap.release() # Close reader to reset or reopen
        
        # Decision
        domain = "traffic_cam" if traffic_votes > (samples_needed / 2) else "normal_view"
        model = det_traffic if domain == "traffic_cam" else det_normal
        self.log(f"  > Decision: {domain}. Processing full video...")

        # 2. Full Processing with Progress & Stop
        save_dir = out_dir / domain
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / video_path.name
        
        # Create Writer
        # VideoWriter overwrites existing files by default if the filename matches
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        processed_frames = 0
        video_start_time = time.time()
        
        # STREAM PREDICTION (Allows interruption)
        results_gen = model.predict(source=str(video_path), stream=True, verbose=False)
        
        try:
            for r in results_gen:
                if self.stop_requested:
                    self.log("  [Aborted Video] Stopping stream...")
                    break
                
                # Write Annotated Frame
                annotated_frame = r.plot()
                writer.write(annotated_frame)
                
                # Update Progress Bar & ETA
                processed_frames += 1
                
                if processed_frames % 10 == 0: # Update UI every 10 frames
                    # Progress calc
                    file_pct = (processed_frames / total_frames)
                    global_pct = ((file_idx + file_pct) / total_files) * 100
                    
                    # ETA Calc (Frame-based)
                    elapsed_sec = time.time() - video_start_time
                    if elapsed_sec > 0:
                        fps_proc = processed_frames / elapsed_sec
                        frames_remain = total_frames - processed_frames
                        sec_remain = frames_remain / fps_proc
                        
                        # Format nicely
                        time_str = f"Speed: {fps_proc:.1f}fps | ETA: {int(sec_remain)}s"
                        
                        # Update UI (Update both progress bar and ETA text)
                        self.update_status(status_prefix, global_pct, time_str)
                        self.root.update_idletasks()

        except Exception as e:
            self.log(f"  [Error in video stream] {e}")
        finally:
            writer.release()
            # If stopped, we now KEEP the partial file instead of deleting it.
            if self.stop_requested and save_path.exists():
                self.log(f"  [Info] Partial video saved to: {save_path.name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartLabelerApp(root)
    root.mainloop()