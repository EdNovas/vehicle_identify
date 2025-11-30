import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import os
import pandas as pd # Requires pip install pandas

# --- LOGIC HELPERS ---
def get_homography_matrix(source_pts, real_w, real_h):
    src = np.float32(source_pts)
    # Map to: TL(0,0), TR(w,0), BR(w,h), BL(0,h)
    # Y=0 is FAR, Y=H is CLOSE
    dst = np.float32([[0, 0], [real_w, 0], [real_w, real_h], [0, real_h]])
    return cv2.getPerspectiveTransform(src, dst)

def transform_point(point, matrix):
    pt_array = np.float32([[point]])
    transformed = cv2.perspectiveTransform(pt_array, matrix)
    return transformed[0][0]

def draw_calibration_ui(img, points, dragging_idx, hover_idx, real_w, real_h):
    """Draws the calibration box, labels, and instructions."""
    overlay = img.copy()
    
    cv2.polylines(img, [np.array(points, np.int32)], True, (0, 255, 0), 2)
    
    # Labels
    mx_w = int((points[0][0] + points[1][0]) / 2)
    my_w = int((points[0][1] + points[1][1]) / 2)
    cv2.putText(img, f"WIDTH ({real_w}m)", (mx_w - 40, my_w - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    mx_l = int((points[1][0] + points[2][0]) / 2)
    my_l = int((points[1][1] + points[2][1]) / 2)
    cv2.putText(img, f"LENGTH ({real_h}m)", (mx_l + 10, my_l), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    labels = ["TL", "TR", "BR", "BL"]
    for i, pt in enumerate(points):
        color = (0, 0, 255) if i == dragging_idx else ((0, 255, 255) if i == hover_idx else (0, 255, 0))
        cv2.circle(img, pt, 12 if i in [dragging_idx, hover_idx] else 8, color, -1)
        cv2.putText(img, labels[i], (pt[0]+15, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Instructions
    cv2.rectangle(overlay, (0, 0), (450, 185), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    lines = [
        "--- INSTRUCTIONS ---",
        "1. MATCH GREEN BOX TO ROAD (Top=Far, Bottom=Close)",
        f"   - Top Edge = Width ({real_w}m)",
        f"   - Side Edge = Length ({real_h}m)",
        "2. Click & Drag circles to adjust",
        "3. ADJUST SLIDER to see road markings",
        "4. Press 'r' to Reset Box",
        "5. Press SPACE to Start"
    ]
    
    y0, dy = 30, 20
    for i, line in enumerate(lines):
        c = (0, 255, 255) if "WIDTH" in line or "LENGTH" in line else (255, 255, 255)
        cv2.putText(img, line, (20, y0 + i*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

# --- GLOBAL VARS FOR CV2 CALLBACK ---
g_points = []
g_dragging_idx = -1
g_hover_idx = -1
g_current_frame_idx = 0 
g_cap = None
g_current_frame = None
g_display_scale = 0.7 # Default scale factor

def mouse_callback(event, x, y, flags, param):
    global g_dragging_idx, g_hover_idx, g_points, g_display_scale
    
    # Scale mouse input UP to match real resolution
    real_x = int(x / g_display_scale)
    real_y = int(y / g_display_scale)
    
    radius = int(20 / g_display_scale) # Scale radius too
    found_hover = -1
    for i, pt in enumerate(g_points):
        if np.linalg.norm(np.array(pt) - np.array((real_x, real_y))) < radius:
            found_hover = i
            break
    g_hover_idx = found_hover

    if event == cv2.EVENT_LBUTTONDOWN and g_hover_idx != -1:
        g_dragging_idx = g_hover_idx
    elif event == cv2.EVENT_MOUSEMOVE and g_dragging_idx != -1:
        g_points[g_dragging_idx] = (real_x, real_y)
    elif event == cv2.EVENT_LBUTTONUP:
        g_dragging_idx = -1

def trackbar_callback(val):
    global g_current_frame_idx, g_cap, g_current_frame
    g_current_frame_idx = val
    g_cap.set(cv2.CAP_PROP_POS_FRAMES, g_current_frame_idx)
    ret, frame = g_cap.read()
    if ret: g_current_frame = frame

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Speed Estimator (Generic/Dashcam)")
        self.root.geometry("600x650")

        self.model_path = tk.StringVar()
        self.video_path = tk.StringVar()
        self.csv_path = tk.StringVar()
        self.real_width = tk.DoubleVar(value=3.5)
        self.real_length = tk.DoubleVar(value=10.0)
        self.const_speed = tk.DoubleVar(value=0.0)
        self.display_scale = tk.DoubleVar(value=0.7) # New Scale Var
        
        self.is_running = False  # Flag to control processing loop

        self.create_ui()

    def create_ui(self):
        # 1. Files
        grp_files = tk.LabelFrame(self.root, text="1. Files", padx=10, pady=10)
        grp_files.pack(fill="x", padx=10, pady=10)
        self.add_file_selector(grp_files, "YOLO Model (.pt):", self.model_path, "*.pt")
        self.add_file_selector(grp_files, "Video File:", self.video_path, "*.mp4 *.avi *.mov")
        
        # 2. Dimensions
        grp_dims = tk.LabelFrame(self.root, text="2. Road Calibration", padx=10, pady=10)
        grp_dims.pack(fill="x", padx=10, pady=5)
        self.add_entry(grp_dims, "Lane Width (m):", self.real_width, "(Top Edge)")
        self.add_entry(grp_dims, "Segment Length (m):", self.real_length, "(Side Edge)")

        # 3. Camera Settings
        grp_dash = tk.LabelFrame(self.root, text="3. Camera Motion Compensation", padx=10, pady=10)
        grp_dash.pack(fill="x", padx=10, pady=5)
        
        tk.Label(grp_dash, text="If camera itself is moving (e.g. Dashcam), provide speed:", font=("Arial", 9, "bold")).pack(anchor="w")
        
        # Option A: Constant
        f_const = tk.Frame(grp_dash)
        f_const.pack(fill="x", pady=5)
        tk.Label(f_const, text="Option A: Constant Speed (km/h):", width=30, anchor="w").pack(side="left")
        tk.Entry(f_const, textvariable=self.const_speed).pack(side="left", fill="x", expand=True)

        # Option B: CSV
        self.add_file_selector(grp_dash, "Option B: Speed CSV:", self.csv_path, "*.csv")
        tk.Label(grp_dash, text="* CSV Format: frame_number, speed_kmh (No Header)", fg="gray", font=("Arial", 8)).pack(anchor="w")

        # 4. Display Settings (New)
        grp_disp = tk.LabelFrame(self.root, text="4. Display Settings", padx=10, pady=10)
        grp_disp.pack(fill="x", padx=10, pady=5)
        self.add_entry(grp_disp, "Window Scale (0.1 - 1.0):", self.display_scale, "(e.g. 0.5 for 50%)")

        # 5. Action Buttons
        f_btns = tk.Frame(self.root, pady=20)
        f_btns.pack(fill="x", padx=20)
        
        self.btn_start = tk.Button(f_btns, text="LAUNCH ESTIMATOR", bg="#007bff", fg="white", 
                              font=("Arial", 14, "bold"), command=self.start_process)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_stop = tk.Button(f_btns, text="STOP", bg="#dc3545", fg="white", 
                             font=("Arial", 14, "bold"), command=self.stop_process, state="disabled")
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(5, 0))

    def add_file_selector(self, parent, label, var, filetypes):
        f = tk.Frame(parent)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, width=20, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(f, text="Browse", command=lambda: self.browse_file(var, filetypes)).pack(side="right")

    def add_entry(self, parent, label, var, note):
        f = tk.Frame(parent)
        f.pack(fill="x", pady=2)
        tk.Label(f, text=label, width=20, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)
        tk.Label(f, text=note, fg="gray").pack(side="right", padx=5)

    def browse_file(self, var, patterns):
        f = filedialog.askopenfilename(filetypes=[("Files", patterns)])
        if f: var.set(f)

    def start_process(self):
        m_path = self.model_path.get()
        v_path = self.video_path.get()
        scale = self.display_scale.get()
        
        if not m_path or not v_path:
            messagebox.showerror("Error", "Select Model and Video")
            return
            
        # Parse CSV if provided
        speed_map = {}
        if self.csv_path.get():
            try:
                df = pd.read_csv(self.csv_path.get(), header=None)
                # Assumes Col 0 = Frame, Col 1 = Speed
                speed_map = dict(zip(df[0], df[1]))
            except Exception as e:
                messagebox.showerror("CSV Error", str(e))
                return
        
        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        
        try:
            self.run_logic(m_path, v_path, self.real_width.get(), self.real_length.get(), self.const_speed.get(), speed_map, scale)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.is_running = False
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

    def stop_process(self):
        self.is_running = False

    def run_logic(self, m_path, v_path, rw, rl, const_spd, speed_map, scale):
        global g_points, g_dragging_idx, g_hover_idx, g_cap, g_total_frames, g_current_frame, g_current_frame_idx, g_display_scale
        
        g_display_scale = scale
        model = YOLO(m_path)
        g_cap = cv2.VideoCapture(v_path)
        w, h = int(g_cap.get(3)), int(g_cap.get(4))
        fps = g_cap.get(cv2.CAP_PROP_FPS)
        g_total_frames = int(g_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- CALIBRATION UI ---
        ret, frame = g_cap.read()
        if not ret: return
        g_current_frame = frame; g_current_frame_idx = 0
        
        cx, cy = w // 2, h // 2
        dx, dy = 150, 200
        g_points = [(cx-dx, cy-dy), (cx+dx, cy-dy), (cx+dx, cy+dy), (cx-dx, cy+dy)]
        g_dragging_idx = -1

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)
        cv2.createTrackbar("Frame", "Calibration", 0, g_total_frames-1, trackbar_callback)

        while True:
            # Check if window was closed manually
            if cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE) < 1:
                g_cap.release()
                cv2.destroyAllWindows()
                return

            # Check stop flag
            if not self.is_running:
                g_cap.release()
                cv2.destroyAllWindows()
                return

            if g_current_frame is not None:
                disp = g_current_frame.copy()
                draw_calibration_ui(disp, g_points, g_dragging_idx, g_hover_idx, rw, rl)
                
                # RESIZE for display
                disp_h, disp_w = int(h * scale), int(w * scale)
                disp_resized = cv2.resize(disp, (disp_w, disp_h))
                
                cv2.imshow("Calibration", disp_resized)
            k = cv2.waitKey(1)
            if k == ord(' '): break
            elif k == ord('r'): g_points = [(cx-dx, cy-dy), (cx+dx, cy-dy), (cx+dx, cy+dy), (cx-dx, cy+dy)]
            elif k == ord('q'): g_cap.release(); cv2.destroyAllWindows(); return

        cv2.destroyWindow("Calibration")
        H = get_homography_matrix(g_points, rw, rl)

        # --- PROCESSING ---
        # Re-open or reset capture to ensure clean state
        g_cap.release()
        g_cap = cv2.VideoCapture(v_path)
        
        out_name = os.path.splitext(v_path)[0] + "_processed_speed.mp4"
        out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        track_hist = defaultdict(lambda: deque(maxlen=20))
        display_speeds = {}
        
        frame_cnt = 0
        
        # Force window creation before loop to ensure it exists
        cv2.namedWindow("Vehicle Speed Estimator")
        
        while g_cap.isOpened():
            # Check if window was closed manually
            if cv2.getWindowProperty("Vehicle Speed Estimator", cv2.WND_PROP_VISIBLE) < 1:
                break

            # Check stop flag
            if not self.is_running:
                break

            ret, frame = g_cap.read()
            if not ret: break
            frame_cnt += 1
            
            # Determine Ego Speed (Camera Speed)
            ego_speed = speed_map.get(frame_cnt, const_spd)

            results = model.track(frame, persist=True, verbose=False)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()

                for box, tid in zip(boxes, ids):
                    x, y, w_b, h_b = box
                    # Use bottom center
                    pt_curr = (float(x), float(y + h_b/2))
                    track_hist[tid].append(pt_curr)

                    if len(track_hist[tid]) > 2:
                        gap = min(5, len(track_hist[tid]) - 1)
                        # Transform to meters
                        # Recall: Y=0 is FAR, Y=H is CLOSE
                        m_curr = transform_point(track_hist[tid][-1], H)
                        m_prev = transform_point(track_hist[tid][-1-gap], H)
                        
                        # Relative Speed
                        dist = np.linalg.norm(m_curr - m_prev)
                        rel_spd = (dist / (gap/fps)) * 3.6
                        
                        # Direction Check: 
                        # If Y increases (prev < curr), object moved 'down' (closer/slower than cam)
                        # If Y decreases (prev > curr), object moved 'up' (away/faster than cam)
                        # NOTE: This assumes standard perspective where Top is Far.
                        y_prev_m, y_curr_m = m_prev[1], m_curr[1]
                        
                        if y_curr_m < y_prev_m: 
                            # Moved 'Up' (Away towards 0) -> Faster than us
                            abs_spd = ego_speed + rel_spd
                        else:
                            # Moved 'Down' (Closer towards H) -> Slower than us
                            abs_spd = ego_speed - rel_spd

                        # Sanity check (no negative speeds)
                        abs_spd = max(0, abs_spd)

                        # Smooth
                        if tid in display_speeds:
                            display_speeds[tid] = 0.8 * display_speeds[tid] + 0.2 * abs_spd
                        else:
                            display_speeds[tid] = abs_spd
                    
                    if tid in display_speeds:
                        x1, y1 = int(x - w_b/2), int(y - h_b/2)
                        x2, y2 = int(x + w_b/2), int(y + h_b/2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Show both speeds for debug? No, just final.
                        label = f"{int(display_speeds[tid])} km/h"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Draw calibration overlay lightly
            cv2.polylines(frame, [np.array(g_points, np.int32)], True, (255, 0, 0), 1)
            # Draw Ego Speed
            cv2.putText(frame, f"CAM SPEED: {ego_speed:.1f} km/h", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # RESIZE for display
            disp_h, disp_w = int(h * scale), int(w * scale)
            frame_resized = cv2.resize(frame, (disp_w, disp_h))

            cv2.imshow("Vehicle Speed Estimator", frame_resized)
            out.write(frame)
            if cv2.waitKey(1) == ord('q'): break

        g_cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()