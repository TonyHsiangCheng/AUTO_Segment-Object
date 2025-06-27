import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageGrab
from segment_anything import sam_model_registry, SamPredictor
import os
import time
import pyautogui  # 用於模擬滑鼠動作

SAM_MODEL_PATH = 'models/sam_vit_h_4b8939.pth'
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=SAM_MODEL_PATH).to(device)
predictor = SamPredictor(sam)

class SAMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM GUI Viewer")

        self.image_bgr = None
        self.image_rgb = None
        self.tk_image = None
        self.image_path = None
        self.masks = []
        self.points = []
        self.labels = []
        self.contours = []
        self.saved_contours = []  # (label, contour)
        self.visible_saved_indices = set()
        self.highlight_saved_index = None
        self.selected_contour_indices = set()
        self.scale_ratio = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.default_center = None  # 儲存預設的中心點 (僅作 SAM 增加點用途)
        self.intersections = []    # 儲存中心水平線與垂直線與遮罩輪廓的交點

        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top_frame, text="載入圖片", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="拍照", command=self.take_screenshot).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="儲存標註結果", command=self.save_annotated_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="清除全部", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="儲存所選曲線", command=self.save_selected_contours).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="儲存二值遮罩", command=self.save_binary_mask).pack(side=tk.LEFT, padx=5)
        # 新增 AI打點 按鈕
        tk.Button(top_frame, text="AI打點", command=self.ai_click_button).pack(side=tk.LEFT, padx=5)

        self.mode = tk.StringVar(value="add")
        ttk.Radiobutton(top_frame, text="增加點", variable=self.mode, value="add").pack(side=tk.LEFT)
        ttk.Radiobutton(top_frame, text="刪除點", variable=self.mode, value="remove").pack(side=tk.LEFT)
        self.info_label = tk.Label(top_frame, text="圖片資訊：")
        self.info_label.pack(side=tk.LEFT, padx=10)

        canvas_frame = tk.Frame(root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, cursor="cross", width=800, height=600, bg="black")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_click)

        self.detail_canvas = tk.Canvas(canvas_frame, width=400, height=400, bg="white")
        self.detail_canvas.pack(side=tk.LEFT, padx=10)

        self.coord_listbox = tk.Listbox(root, width=80, selectmode=tk.EXTENDED)
        self.coord_listbox.pack(side=tk.BOTTOM, fill=tk.X)
        self.coord_listbox.bind("<<ListboxSelect>>", self.on_select_contour)

        self.saved_listbox = tk.Listbox(root, width=80, selectmode=tk.MULTIPLE)
        self.saved_listbox.pack(side=tk.BOTTOM, fill=tk.X)
        self.saved_listbox.bind("<<ListboxSelect>>", self.on_saved_selection)

    def ai_click_button(self):
        """按下 AI打點 按鈕後等待2秒再開始動作"""
        self.root.after(2000, self.ai_click)

    def ai_click(self):
        """執行AI打點：模擬點擊 icon，再將你選取的曲線點進行分群取100個代表點，
        並在擷取區域內模擬點擊，過濾掉靠近照片邊緣的點（margin=5 像素）。
        """
        # Step 1: 模擬點擊 icon
        pyautogui.moveTo(780,650) #舊版(780,650)新版(460, 62)
        pyautogui.click()
        
        pyautogui.moveTo(1000,650) #舊版(1000,650)新版(990, 62)
        pyautogui.click()
        

        # Step 2: 檢查是否有選取的曲線
        if not self.selected_contour_indices:
            messagebox.showerror("錯誤", "請先點選要使用的曲線")
            return

        # 使用使用者點選的曲線 (取第一個選取的曲線)
        selected_index = list(self.selected_contour_indices)[0]
        contour_points = self.contours[selected_index]
        
        pts = np.float32(contour_points.reshape(-1, 2))
        if len(pts) < 20:
            messagebox.showerror("錯誤", "所選曲線邊緣點數量不足以分成20群")
            return

        # 使用 OpenCV k-means 將點分成100群
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 30
        ret, labels, centers = cv2.kmeans(pts, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = centers.astype(np.int32)

        # 擷取區域的左上角偏移 (固定為 新版(388,218)；舊版(1118,430))
        captured_offset = (1118,430)
        # 取得圖片尺寸（原圖座標的有效範圍）
        h_img, w_img = self.image_rgb.shape[:2]
        for center in centers:
            cx, cy = int(center[0]), int(center[1])
            # 檢查該點是否在圖片有效區域內（隱藏靠近邊緣的點）
            if not self.is_inside_margin((cx, cy), w_img, h_img):
                continue
            screen_x = captured_offset[0] + cx
            screen_y = captured_offset[1] + cy
            print(f"點擊座標: 原圖({cx}, {cy}) -> 螢幕({screen_x}, {screen_y})")
            pyautogui.moveTo(screen_x, screen_y)
            pyautogui.click()
        print("AI打點完成")


    def take_screenshot(self):
        """擷取指定螢幕區域並作為圖片來源，並以圖片中心作為 SAM 增加點，
           接著計算中心水平線與垂直線與遮罩輪廓交點"""
        try:
            # 擷取螢幕區域: 舊版左上 (1118,430), 右下 (1919,1031)
            # 擷取螢幕區域: 新版左上 (388, 218), 右下 (1408, 982)
            screenshot = ImageGrab.grab(bbox=(1118,430, 1919,1031)) #新版點位
        except Exception as e:
            messagebox.showerror("錯誤", f"拍照失敗: {e}")
            return

        self.image_rgb = np.array(screenshot)
        self.image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)

        h, w = self.image_rgb.shape[:2]
        mpix = (w * h) / 1e6
        file_size = 0
        self.info_label.config(text=f"圖片尺寸：{w}x{h}，{mpix:.2f} MPixel，檔案大小：{file_size:.2f} MB (拍照)")

        self.points.clear()
        self.labels.clear()
        self.masks.clear()
        self.contours.clear()
        self.selected_contour_indices.clear()
        self.coord_listbox.delete(0, tk.END)
        self.saved_contours.clear()
        self.visible_saved_indices.clear()
        self.highlight_saved_index = None
        self.saved_listbox.delete(0, tk.END)
        self.default_center = None
        self.intersections = []

        predictor.set_image(self.image_rgb)

        center_x = w // 2
        center_y = h // 2
        self.default_center = (center_x, center_y)
        self.points.append([center_x, center_y])
        self.labels.append(1)
        print(f"預設中心點: ({center_x}, {center_y})")

        input_point = np.array(self.points)
        input_label = np.array(self.labels)
        start_time = time.time()
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        print(f"SAM 耗時: {time.time() - start_time:.2f} 秒")
        mask = masks[0].astype(np.uint8)
        self.masks = [mask]

        self.extract_contours(mask)
        self.intersections = self.compute_intersections(self.default_center)
        self.redraw()

    def compute_intersections(self, center):
        intersections = []
        cx, cy = center
        for contour in self.contours:
            pts = contour.reshape(-1, 2)
            for i in range(1, len(pts)):
                p1 = pts[i - 1]
                p2 = pts[i]
                if (p1[1] - cy) * (p2[1] - cy) <= 0 and p1[1] != p2[1]:
                    t = (cy - p1[1]) / (p2[1] - p1[1])
                    ix = p1[0] + t * (p2[0] - p1[0])
                    intersections.append((int(round(ix)), cy))
                if (p1[0] - cx) * (p2[0] - cx) <= 0 and p1[0] != p2[0]:
                    t = (cx - p1[0]) / (p2[0] - p1[0])
                    iy = p1[1] + t * (p2[1] - p1[1])
                    intersections.append((cx, int(round(iy))))
        return intersections

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return

        self.image_path = path
        try:
            with open(path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                self.image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except:
            self.image_bgr = None

        if self.image_bgr is None:
            messagebox.showerror("錯誤", "圖片讀取失敗，請檢查檔案是否存在或為有效圖檔格式")
            return

        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)

        h, w = self.image_rgb.shape[:2]
        mpix = (w * h) / 1e6
        file_size = os.path.getsize(path) / 1024 / 1024
        self.info_label.config(text=f"圖片尺寸：{w}x{h}，{mpix:.2f} MPixel，檔案大小：{file_size:.2f} MB")

        self.points.clear()
        self.labels.clear()
        self.masks.clear()
        self.contours.clear()
        self.selected_contour_indices.clear()
        self.coord_listbox.delete(0, tk.END)
        self.saved_contours.clear()
        self.visible_saved_indices.clear()
        self.highlight_saved_index = None
        self.saved_listbox.delete(0, tk.END)
        self.default_center = None
        self.intersections = []

        predictor.set_image(self.image_rgb)
        self.redraw()

    def on_click(self, event):
        if self.image_rgb is None:
            return
        x = int((event.x - self.offset_x) / self.scale_ratio)
        y = int((event.y - self.offset_y) / self.scale_ratio)
        if x < 0 or y < 0:
            return
        label = 1 if self.mode.get() == "add" else 0
        self.points.append([x, y])
        self.labels.append(label)

        input_point = np.array(self.points)
        input_label = np.array(self.labels)
        start_time = time.time()
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        print(f"SAM 耗時: {time.time() - start_time:.2f} 秒")
        mask = masks[0].astype(np.uint8)
        self.masks = [mask]
        self.extract_contours(mask)
        if self.default_center:
            self.intersections = self.compute_intersections(self.default_center)
        self.redraw()

    def extract_contours(self, mask):
        self.contours.clear()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if len(contour) < 2:
                continue
            total_length = sum(np.linalg.norm(contour[i][0] - contour[i - 1][0]) for i in range(1, len(contour)))
            if total_length == 0:
                continue
            step = total_length / 500
            sampled = [contour[0]]
            acc_length = 0
            for i in range(1, len(contour)):
                acc_length += np.linalg.norm(contour[i][0] - contour[i - 1][0])
                if acc_length >= step * len(sampled):
                    sampled.append(contour[i])
                if len(sampled) >= 500:
                    break
            sampled = np.array(sampled).reshape(-1, 2).astype(np.int32)
            self.contours.append(sampled)
        self.coord_listbox.delete(0, tk.END)
        for i, c in enumerate(self.contours):
            self.coord_listbox.insert(tk.END, f"曲線 {i+1} ({len(c)} 點)")
        if self.contours:
            self.selected_contour_indices = {0}
            self.coord_listbox.selection_clear(0, tk.END)
            self.coord_listbox.selection_set(0)

    def save_selected_contours(self):
        for idx in self.selected_contour_indices:
            if idx < len(self.contours):
                label = f"遮罩{len(self.saved_contours)+1} - 曲線 {idx+1}"
                self.saved_contours.append((label, self.contours[idx].copy()))
                self.saved_listbox.insert(tk.END, label)
                self.visible_saved_indices.add(len(self.saved_contours) - 1)
        self.redraw()

    def on_select_contour(self, event):
        self.selected_contour_indices = set(event.widget.curselection())
        self.redraw()

    def save_binary_mask(self):
        if self.image_rgb is None or not self.masks:
            return
        h, w = self.image_rgb.shape[:2]
        mask = self.masks[0]

        # 建立純二值遮罩：mask 為 1 → 255（白色），其餘為 0（黑色）
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        binary_mask[mask != 0] = 255

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 圖片", "*.png")])
        if save_path:
            cv2.imwrite(save_path, binary_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f"✅ 已儲存二值遮罩影像（白=SAM區域）：{save_path}")


    def on_saved_selection(self, event):
        self.visible_saved_indices = set(event.widget.curselection())
        if self.visible_saved_indices:
            self.highlight_saved_index = list(self.visible_saved_indices)[0]
        else:
            self.highlight_saved_index = None
        self.redraw()

    def clear_all(self):
        self.points.clear()
        self.labels.clear()
        self.masks.clear()
        self.contours.clear()
        self.coord_listbox.delete(0, tk.END)
        self.selected_contour_indices.clear()
        self.default_center = None
        self.intersections = []
        self.redraw()

    def draw_open_polyline(self, image, points, color, thickness, w, h):
        last_valid = None
        for pt in points:
            if self.is_inside_margin(pt, w, h):
                if last_valid is not None:
                    cv2.line(image, last_valid, pt, color, thickness)
                last_valid = pt
            else:
                last_valid = None

    def redraw(self):
        if self.image_rgb is None:
            return
        overlay = self.image_rgb.copy()
        h, w = overlay.shape[:2]

        if self.masks:
            mask = self.masks[0]
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            color_mask[mask != 0] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)

        for pt, label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(overlay, tuple(pt), 5, color, -1)

        if self.intersections:
            for (ix, iy) in self.intersections:
                size = 10
                cv2.line(overlay, (ix - size, iy - size), (ix + size, iy + size), (0, 255, 255), 2)
                cv2.line(overlay, (ix - size, iy + size), (ix + size, iy - size), (0, 255, 255), 2)

        for idx, contour in enumerate(self.contours):
            if idx in self.selected_contour_indices:
                self.draw_open_polyline(overlay, contour, (255, 0, 0), 1, w, h)
                if len(contour) > 0:
                    cv2.putText(overlay, str(idx + 1), tuple(contour[0]), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 4)

        for idx, (label, contour) in enumerate(self.saved_contours):
            if idx in self.visible_saved_indices:
                thick = 4 if idx == self.highlight_saved_index else 2
                self.draw_open_polyline(overlay, contour, (0, 255, 255), thick, w, h)
                # if len(contour) > 0:
                #     cv2.putText(overlay, f"S{idx+1}", tuple(contour[0]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 100), 3)

        img = Image.fromarray(overlay)
        img_w, img_h = img.size
        canvas_w, canvas_h = 800, 600
        scale = min(canvas_w / img_w, canvas_h / img_h)
        self.scale_ratio = scale
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)

        self.draw_detail_canvas()

    def draw_detail_canvas(self):
        self.detail_canvas.delete("all")
        width, height = 400, 400
        margin = 20
        contours_to_draw = []

        for idx in self.selected_contour_indices:
            if idx < len(self.contours):
                contours_to_draw.append((self.contours[idx], f"{idx+1}", "#00BFFF"))
        for idx in self.visible_saved_indices:
            if idx < len(self.saved_contours):
                label, contour = self.saved_contours[idx]
                contours_to_draw.append((contour, f"S{idx+1}", "#FFA07A"))
        if not contours_to_draw:
            return

        all_xs = np.concatenate([c[:, 0] for c, _, _ in contours_to_draw])
        all_ys = np.concatenate([c[:, 1] for c, _, _ in contours_to_draw])
        min_x, max_x = all_xs.min(), all_xs.max()
        min_y, max_y = all_ys.min(), all_ys.max()
        range_x = max_x - min_x
        range_y = max_y - min_y
        if range_x == 0 or range_y == 0:
            return
        scale = min((width - 2 * margin) / range_x, (height - 2 * margin) / range_y)
        for contour, label, color in contours_to_draw:
            xs, ys = contour[:, 0], contour[:, 1]
            points = [((x - min_x) * scale + margin, (y - min_y) * scale + margin) for x, y in zip(xs, ys)]
            for i, pt in enumerate(points):
                orig_x, orig_y = contour[i]
                if self.is_inside_margin((orig_x, orig_y), self.image_rgb.shape[1], self.image_rgb.shape[0]):
                    self.detail_canvas.create_oval(pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2, fill=color)

    def is_inside_margin(self, pt, w, h, margin=5):
        x, y = pt
        return margin <= x < w - margin and margin <= y < h - margin

    def save_annotated_image(self):
        if self.image_rgb is None:
            return
        overlay = self.image_rgb.copy()
        h, w = overlay.shape[:2]
        if self.masks:
            mask = self.masks[0]
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            color_mask[mask != 0] = [0, 255, 0]
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.4, 0)
        for idx in self.selected_contour_indices:
            if idx < len(self.contours):
                self.draw_open_polyline(overlay, self.contours[idx], (255, 0, 0), 1, w, h)
        for idx in self.visible_saved_indices:
            if idx < len(self.saved_contours):
                _, contour = self.saved_contours[idx]
                self.draw_open_polyline(overlay, contour, (0, 255, 255), 1, w, h)
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 圖片", "*.png")])
        if save_path:
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, overlay_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__ == '__main__':
    root = tk.Tk()
    app = SAMGUI(root)
    root.geometry("1300x900")
    root.mainloop()
