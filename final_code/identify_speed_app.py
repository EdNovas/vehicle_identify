import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import os
import threading
import time

# --- LOGIC HELPERS ---
def get_homography_matrix(source_pts, real_w, real_h):
    src = np.float32(source_pts)
    # Map to: TL(0,0), TR(w,0), BR(w,h), BL(0,h)
    dst = np.float32([[0, 0], [real_w, 0], [real_w, real_h], [0, real_h]])
    return cv2.getPerspectiveTransform(src, dst)

def transform_point(point, matrix):
    pt_array = np.float32([[point]])
    transformed = cv2.perspectiveTransform(pt_array, matrix)
    return transformed[0][0]

# --- CALIBRATION UI HELPERS ---
def draw_calibration_ui(img, points, dragging_idx, hover_idx, real_w, real_h):
    # 1. Draw Polygon Lines
    cv2.polylines(img, [np.array(points, np.int32)], True, (0, 255, 0), 2)
    
    # 2. Draw Dimension Labels
    mx_w = int((points[0][0] + points[1][0]) / 2)
    my_w = int((points[0][1] + points[1][1]) / 2)
    cv2.putText(img, f"WIDTH ({real_w}m)", (mx_w - 40, my_w - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    mx_l = int((points[1][0] + points[2][0]) / 2)
    my_l = int((points[1][1] + points[2][1]) / 2)
    cv2.putText(img, f"LENGTH ({real_h}m)", (mx_l + 10, my_l), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 3. Draw Corners
    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(points):
        if i == dragging_idx:
            color = (0, 0, 255) # Red (Dragging)
            radius = 14
        elif i == hover_idx:
            color = (0, 255, 255) # Yellow (Hover)
            radius = 14
        else:
            color = (0, 255, 0) # Green (Idle)
            radius = 10
            
        cv2.circle(img, pt, radius, color, -1)
        cv2.putText(img, labels[i], (pt[0]+15, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 4. Transparent Instructions Overlay
    overlay = img.copy()
    h, w = img.shape[:2]
    
    box_w, box_h = 600, 230
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    lines = [
        "--- CALIBRATION GUIDE ---",
        "1. FRAME SELECTION: Use the SLIDER below to find a clear frame.",
        "2. ALIGN THE GREEN BOX:",
        f"   - Top Edge: Must span the Lane Width ({real_w}m)",
        f"   - Side Edge: Must span the Segment Length ({real_h}m)",
        "3. CONTROLS:",
        "   - Hover (Yellow) -> Click & Drag (Red) to move corners.",
        "   - Press 'r' to Reset box shape.",
        "4. START:",
        "   - Press SPACE to Confirm & Begin Analysis."
    ]
    
    y0, dy = 30, 22
    for i, line in enumerate(lines):
        c = (0, 255, 255) if "Width" in line or "Length" in line or "SPACE" in line else (255, 255, 255)
        scale = 0.6 if i == 0 else 0.5
        thickness = 2 if i == 0 else 1
        cv2.putText(img, line, (20, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, scale, c, thickness)

# --- MOUSE CALLBACKS ---
g_points = []
g_dragging_idx = -1
g_hover_idx = -1
g_scale = 0.7

def mouse_callback(event, x, y, flags, param):
    global g_dragging_idx, g_hover_idx, g_points, g_scale
    
    # Adjust coordinates for display scale
    real_x = int(x / g_scale)
    real_y = int(y / g_scale)
    radius = int(20 / g_scale)

    found = -1
    for i, pt in enumerate(g_points):
        if np.linalg.norm(np.array(pt) - np.array((real_x, real_y))) < radius:
            found = i; break
    g_hover_idx = found

    if event == cv2.EVENT_LBUTTONDOWN and g_hover_idx != -1:
        g_dragging_idx = g_hover_idx
    elif event == cv2.EVENT_MOUSEMOVE and g_dragging_idx != -1:
        g_points[g_dragging_idx] = (real_x, real_y)
    elif event == cv2.EVENT_LBUTTONUP:
        g_dragging_idx = -1

g_current_frame_idx = 0
g_frame_dirty = True 

def trackbar_callback(val):
    global g_current_frame_idx, g_frame_dirty
    g_current_frame_idx = val
    g_frame_dirty = True

# --- MAIN APP CLASS ---
class UnifiedTrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Traffic Analyzer (Identify + Speed)")
        self.root.geometry("750x850")
        
        # Variables
        self.gate_model = tk.StringVar()
        self.traffic_model = tk.StringVar()
        self.street_model = tk.StringVar()
        self.video_path = tk.StringVar()
        
        self.real_width = tk.DoubleVar(value=3.5)
        self.real_length = tk.DoubleVar(value=10.0)
        self.ego_speed = tk.DoubleVar(value=0.0)
        self.disp_scale = tk.DoubleVar(value=0.7)
        
        self.is_running = False
        
        # Check for embedded models
        self.check_embedded_models()
        
        self.build_ui()

    def check_embedded_models(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        gate_pt = os.path.join(models_dir, "gate.pt")
        traffic_pt = os.path.join(models_dir, "traffic.pt")
        street_pt = os.path.join(models_dir, "street.pt")
        
        if os.path.exists(gate_pt): self.gate_model.set(gate_pt)
        if os.path.exists(traffic_pt): self.traffic_model.set(traffic_pt)
        if os.path.exists(street_pt): self.street_model.set(street_pt)

    def build_ui(self):
        # 1. Models
        grp_mod = tk.LabelFrame(self.root, text="1. AI Models", padx=10, pady=10)
        grp_mod.pack(fill="x", padx=10, pady=5)
        self.add_file(grp_mod, "Gate Model (.pt):", self.gate_model)
        self.add_file(grp_mod, "Traffic Expert (.pt):", self.traffic_model)
        self.add_file(grp_mod, "Street Expert (.pt):", self.street_model)

        # 2. Input
        grp_in = tk.LabelFrame(self.root, text="2. Input Video", padx=10, pady=10)
        grp_in.pack(fill="x", padx=10, pady=5)
        self.add_file(grp_in, "Video File:", self.video_path, is_video=True)

        # 3. Speed Calibration
        grp_spd = tk.LabelFrame(self.root, text="3. Speed Calibration", padx=10, pady=10)
        grp_spd.pack(fill="x", padx=10, pady=5)
        
        f1 = tk.Frame(grp_spd); f1.pack(fill="x", pady=2)
        tk.Label(f1, text="Lane Width (m):", width=20, anchor="w").pack(side="left")
        tk.Entry(f1, textvariable=self.real_width).pack(side="left", fill="x", expand=True)
        
        f2 = tk.Frame(grp_spd); f2.pack(fill="x", pady=2)
        tk.Label(f2, text="Segment Length (m):", width=20, anchor="w").pack(side="left")
        tk.Entry(f2, textvariable=self.real_length).pack(side="left", fill="x", expand=True)
        
        f3 = tk.Frame(grp_spd); f3.pack(fill="x", pady=2)
        tk.Label(f3, text="Camera Speed (km/h):", width=20, anchor="w").pack(side="left")
        tk.Entry(f3, textvariable=self.ego_speed).pack(side="left", fill="x", expand=True)
        tk.Label(f3, text="(Set 0 for Static Cam)", fg="gray").pack(side="right")

        # 4. Settings
        grp_set = tk.LabelFrame(self.root, text="4. Display Settings", padx=10, pady=10)
        grp_set.pack(fill="x", padx=10, pady=5)
        self.add_entry(grp_set, "Window Scale (0.1-1.0):", self.disp_scale)

        # 5. Actions
        f_act = tk.Frame(self.root, pady=15)
        f_act.pack(fill="x", padx=20)
        self.btn_run = tk.Button(f_act, text="START ANALYZER", bg="green", fg="white", font=("Arial", 12, "bold"), command=self.start_thread)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=5)
        self.btn_stop = tk.Button(f_act, text="STOP", bg="red", fg="white", font=("Arial", 12, "bold"), command=self.stop, state="disabled")
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=5)
        
        self.log_area = tk.Text(self.root, height=8, state="disabled")
        self.log_area.pack(fill="both", expand=True, padx=10, pady=5)

    def add_file(self, parent, label, var, is_video=False):
        f = tk.Frame(parent)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, width=20, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        types = [("Videos", "*.mp4 *.avi *.mov")] if is_video else [("Models", "*.pt")]
        tk.Button(f, text="Browse", command=lambda: self.browse(var, types)).pack(side="right")

    def add_entry(self, parent, label, var):
        f = tk.Frame(parent)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, width=20, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)

    def browse(self, var, types):
        f = filedialog.askopenfilename(filetypes=types)
        if f: var.set(f)

    def log(self, msg):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, msg + "\n")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")

    def start_thread(self):
        if not all([self.gate_model.get(), self.traffic_model.get(), self.street_model.get(), self.video_path.get()]):
            messagebox.showerror("Error", "Please select all models and the video file.")
            return
        self.is_running = True
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self.run_process, daemon=True).start()

    def stop(self):
        self.is_running = False

    def run_process(self):
        try:
            self.log("--- Loading Gate Model... ---")
            gate = YOLO(self.gate_model.get())
            
            vid_path = self.video_path.get()
            rw = self.real_width.get()
            rl = self.real_length.get()
            ego = self.ego_speed.get()
            scale = self.disp_scale.get()

            # 1. CALIBRATION PHASE
            self.log("Starting Calibration...")
            H, calib_frame = self.run_calibration(vid_path, rw, rl, scale)
            if H is None: 
                self.log("Calibration cancelled.")
                self.cleanup()
                return

            # --- LOADING SCREEN ---
            self.show_loading_screen(calib_frame, scale)

            # 2. IDENTIFICATION PHASE
            self.log("Identifying Domain (Sampling first 15 frames)...")
            domain, confidence = self.identify_domain(gate, vid_path)
            self.log(f"Result: {domain} (Conf: {confidence:.2f})")
            
            # 3. LOAD EXPERT MODEL
            expert_path = self.traffic_model.get() if domain == "traffic_cam" else self.street_model.get()
            self.log(f"Loading Expert: {os.path.basename(expert_path)}...")
            expert = YOLO(expert_path)

            # 4. PROCESSING LOOP
            self.log("Starting Analysis Loop...")
            self.run_analysis(expert, vid_path, H, ego, scale, domain)

        except Exception as e:
            self.log(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        self.is_running = False
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        cv2.destroyAllWindows()

    def show_loading_screen(self, frame, scale):
        if frame is None: return
        loading_img = frame.copy()
        h, w = loading_img.shape[:2]
        
        overlay = loading_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, loading_img, 0.3, 0, loading_img)
        
        text = "LOADING AI MODELS..."
        font = cv2.FONT_HERSHEY_SIMPLEX
        t_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (w - t_size[0]) // 2
        text_y = (h + t_size[1]) // 2
        
        cv2.putText(loading_img, text, (text_x, text_y), font, 1.5, (0, 255, 255), 3)
        cv2.putText(loading_img, "Please Wait", (text_x + 50, text_y + 60), font, 1.0, (255, 255, 255), 2)
        
        disp_h, disp_w = int(h * scale), int(w * scale)
        cv2.imshow("Calibration", cv2.resize(loading_img, (disp_w, disp_h)))
        cv2.waitKey(1)

    def run_calibration(self, vid_path, rw, rl, scale):
        global g_points, g_dragging_idx, g_hover_idx, g_scale, g_cap_calib, g_frame_dirty
        g_scale = scale
        
        g_cap_calib = cv2.VideoCapture(vid_path)
        w = int(g_cap_calib.get(3))
        h = int(g_cap_calib.get(4))
        total_frames = int(g_cap_calib.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cx, cy = w//2, h//2
        dx, dy = 150, 200
        g_points = [(cx-dx, cy-dy), (cx+dx, cy-dy), (cx+dx, cy+dy), (cx-dx, cy+dy)]
        g_dragging_idx = -1
        g_frame_dirty = True

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        cv2.createTrackbar("Frame", "Calibration", 0, total_frames-1, trackbar_callback)

        H_matrix = None
        current_frame = None
        
        while self.is_running:
            if cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE) < 1:
                break

            if g_frame_dirty:
                g_cap_calib.set(cv2.CAP_PROP_POS_FRAMES, g_current_frame_idx)
                ret, frame = g_cap_calib.read()
                if ret:
                    current_frame = frame
                g_frame_dirty = False

            if current_frame is not None:
                disp = current_frame.copy()
                draw_calibration_ui(disp, g_points, g_dragging_idx, g_hover_idx, rw, rl)
                disp_h, disp_w = int(h * scale), int(w * scale)
                cv2.imshow("Calibration", cv2.resize(disp, (disp_w, disp_h)))
            
            k = cv2.waitKey(10)
            if k == ord(' '): 
                H_matrix = get_homography_matrix(g_points, rw, rl)
                break
            elif k == ord('r'):
                g_points = [(cx-dx, cy-dy), (cx+dx, cy-dy), (cx+dx, cy+dy), (cx-dx, cy+dy)]
            elif k == ord('q'):
                self.is_running = False
        
        g_cap_calib.release()
        return H_matrix, current_frame

    def identify_domain(self, gate_model, vid_path):
        cap = cv2.VideoCapture(vid_path)
        votes = {"traffic_cam": 0, "normal_view": 0}
        frames_checked = 0
        
        while frames_checked < 15:
            ret, frame = cap.read()
            if not ret: break
            
            res = gate_model(frame, verbose=False)[0]
            probs = res.probs
            top1 = probs.top1
            name = res.names[top1]
            
            if "traffic" in name.lower():
                votes["traffic_cam"] += 1
            else:
                votes["normal_view"] += 1
            frames_checked += 1
            
        cap.release()
        
        if votes["traffic_cam"] > votes["normal_view"]:
            return "traffic_cam", votes["traffic_cam"]/frames_checked
        else:
            return "normal_view", votes["normal_view"]/frames_checked

    def run_analysis(self, model, vid_path, H, ego_speed, scale, domain):
        cv2.destroyWindow("Calibration")
        
        cap = cv2.VideoCapture(vid_path)
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out_path = os.path.splitext(vid_path)[0] + "_analyzed.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        track_hist = defaultdict(lambda: deque(maxlen=20))
        speeds = {}
        
        # --- NEW: Class Counting ---
        # Map Track ID -> Class Name (e.g. {1: "Car", 2: "Truck"})
        vehicle_registry = {} 
        
        cv2.namedWindow("Analysis")
        
        while self.is_running and cap.isOpened():
            if cv2.getWindowProperty("Analysis", cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret: break
            
            results = model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                clss = results[0].boxes.cls.int().cpu().numpy()
                
                for box, tid, cls in zip(boxes, ids, clss):
                    x, y, wb, hb = box
                    cls_name = model.names[cls]
                    
                    # Store unique ID and its Class
                    if tid not in vehicle_registry:
                        vehicle_registry[tid] = cls_name
                    
                    # Speed Logic
                    pt_curr = (float(x), float(y + hb/2))
                    track_hist[tid].append(pt_curr)
                    
                    spd_val = 0
                    if len(track_hist[tid]) > 2:
                        gap = min(5, len(track_hist[tid]) - 1)
                        m_curr = transform_point(track_hist[tid][-1], H)
                        m_prev = transform_point(track_hist[tid][-1-gap], H)
                        
                        dist = np.linalg.norm(m_curr - m_prev)
                        rel_spd = (dist / (gap/fps)) * 3.6
                        
                        # --- STATIC vs DASHCAM LOGIC ---
                        if ego_speed == 0:
                            spd_val = rel_spd
                        else:
                            if m_curr[1] < m_prev[1]: # Moving Up
                                spd_val = ego_speed + rel_spd
                            else: # Moving Down
                                spd_val = max(0, ego_speed - rel_spd)
                        
                        if tid in speeds:
                            speeds[tid] = 0.9 * speeds[tid] + 0.1 * spd_val
                        else:
                            speeds[tid] = spd_val
                    
                    # DRAWING
                    x1, y1 = int(x - wb/2), int(y - hb/2)
                    x2, y2 = int(x + wb/2), int(y + hb/2)
                    color = ((int(tid)*50)%255, (int(tid)*100)%255, (int(tid)*200)%255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    spd_str = f"{int(speeds[tid])}km/h" if tid in speeds else "..."
                    label = f"{cls_name} | {spd_str}"
                    
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- DRAW UI OVERLAY ---
            # 1. Domain
            cv2.putText(frame, f"DOMAIN: {domain.upper()}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 2. Total Count
            total_count = len(vehicle_registry)
            cv2.putText(frame, f"TOTAL: {total_count}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 3. Class Breakdown (Car: 5, Truck: 2)
            # Count occurrences of each class in registry
            counts = defaultdict(int)
            for _, c_name in vehicle_registry.items():
                counts[c_name] += 1
            
            # Format string: "Car: 5 | Truck: 2"
            breakdown_str = " | ".join([f"{k}: {v}" for k, v in counts.items()])
            cv2.putText(frame, breakdown_str, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Draw Calibration Box
            cv2.polylines(frame, [np.array(g_points, np.int32)], True, (255, 0, 0), 1)

            h_disp, w_disp = int(h * scale), int(w * scale)
            cv2.imshow("Analysis", cv2.resize(frame, (w_disp, h_disp)))
            writer.write(frame)
            
            if cv2.waitKey(1) == ord('q'): break
            
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        # Log Final Counts
        final_str = ", ".join([f"{k}: {v}" for k, v in counts.items()])
        self.log(f"Analysis Complete.")
        self.log(f"Total: {total_count}")
        self.log(f"Breakdown: {final_str}")
        
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = UnifiedTrafficApp(root)
    root.mainloop()