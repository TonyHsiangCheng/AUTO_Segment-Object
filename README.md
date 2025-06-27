#  Segment Anything Based Object Detection and Measurement Tool

使用 Meta AI 的 Segment Anything Model (SAM)，結合工業相機影像進行物件分割、自動化邊緣擷取與幾何量測的應用專案。

---

##  專案簡介

本專案致力於將深度學習語意分割模型（SAM）應用於精密製造流程中，實現自動化物件分割與邊界提取，進一步應用於量測、缺陷判斷與品質管理。

目前努力達成的方向有:

- 提供廠內品保人員無需技能便能將自動化AI工具套用在樣品量測時需要設定的自動化程序
- 提供客製化專案，使研發人員CFT(major NPI & RD team)無須訓練模型便可以知道物件外型，已實現應用: 1.玻璃網版印刷字體歪斜 2.膠溝內氣泡分布及面積定義
- 提供組內(PMC)進行斷層CT Scan時將肉眼不易辨識、且調整原生躁點還是看不清楚的邊界能藉由機器視覺卷積(CNN)邏輯將結果視覺化呈現

---

##  動機與背景

傳統機器視覺在產品邊緣辨識上常受限於參數設定（如Canny/Sobel），且難以應對不同背景與複雜圖樣。為解決此問題，我結合了：

- Meta 提出的 Segment Anything (SAM) 模型
- 自訂化 GUI 工具並即時蒐集使用者喜好，即時修正Promt精度
- 精密量測需求下的邊緣修補、外接矩形與輪廓抽取邏輯
- 設計Promt Encoder進一步提升自動化模型

應用場景包含：
- 工業產品氣泡缺陷標註
- 黏著區分割與面積分析
- 光學元件邊界量測
- 錶殼或玻璃件中物件中心定位
- 未知且模糊邊緣定位(傳統2.5D在含有拔模角的塑膠件不易對焦)

---

##  技術架構與方法

- **預處裡**：正規化輸入圖片格式 `800*600`
- **模型基礎**：Meta AI 的 `Segment Anything Model (ViT-H)`
- **提示點控制**：支援 Positive / Negative 點標註與即時遮罩疊加
- **邊界強化處理**：Laplacian/Canny/Sobel 預處理作為參考輔助
- **反向傳播**：分類物件，使每次使用者下的Promt都能加入訓練模型，且避免模型偏移，要求門檻值
- **量測輔助**：遮罩面積計算、最小外接矩形、重心提取與比例換算
- **可視化**：Tkinter 介面呈現分割結果與使用者互動選點

---

##  使用方式

```bash
# 安裝必要套件
pip install -r requirements.txt
```
```bash
# Segment Model
請詳閱SAM & SAM2 https://github.com/facebookresearch/segment-anything?tab=readme-ov-file
```
```bash
# API 回傳 mask
def do_segmentation_with_all_hints():
    """
    用 active_positive, active_negative, active_bbox 三項合成一个 Prompt，
    只呼叫一次 API，更新 segmentation_masks 并重繪
    """
    prompt = Prompt(
        positives=active_positive.copy(),
        negatives=active_negative.copy(),
        bbox=active_bbox
    )
    # 只传一个 prompt，期待回一个 mask 列表
    masks = segment(img, [prompt])
    # 覆盖旧的 masks
    segmentation_masks[:] = masks
    redraw_annotations_and_masks()
```
```bash
# 描邊以及功能性選擇 (以Garmin廠內為例: 有描邊及最近點需求)
# polygon 邊界
poly = seg.get("polygon", [])
if poly:
    pts = []
    for px, py in poly:
        pts += [px*scale + offset_x, py*scale + offset_y]
    if len(pts) >= 4:
        canvas.create_polygon(
            pts, outline="red", fill="", width=1,
            tags=("segmentation", f"poly_{seg_idx}")
        )

# 只有在交點模式下才畫最小外接矩形
    if mode_pt == "intersection":
        # 最小外接矩形
        rect_pts = []
        for px, py in seg.get("min_rect_corners", []):
            rect_pts += [float(px)*scale + offset_x, float(py)*scale + offset_y]
        if len(rect_pts) >= 4:
            canvas.create_polygon(
                rect_pts, outline="blue", fill="", width=2,
                tags=("segmentation", f"min_rect_{seg_idx}")
            )

if mode_pt == "intersection":
    # 繪製所有交點 + 紅黃標記
    for pt_idx, pt in enumerate(seg.get("intersection_points", [])):
        x_int = pt[0]*scale + offset_x
        y_int = pt[1]*scale + offset_y
        color = "yellow"
        cp = seg.get("closest_point_info")
        if cp and math.isclose(pt[0], cp[0], rel_tol=1e-6) and math.isclose(pt[1], cp[1], rel_tol=1e-6):
            color = "red"
        canvas.create_oval(
            x_int-4, y_int-4, x_int+4, y_int+4,
            fill=color, outline="black", width=1,
            tags=("segmentation", f"intersection_pt_{seg_idx}_{pt_idx}")
        )
```
- 必要時須將mask校正

---

##  主要貢獻者

- Tony Cheng (PMC Mechanical & AI Engineer)

提供實驗室專案架構，進行智慧實驗室升級

驗證模型及校正遮罩，使SAM回傳之遮罩準確度提升至亞像素等級

設計GUI及後端資料蒐集作為後續模型擴展的主幹

結合AI工具及傳統工具機操作DEMO教學

- Roger Lo (PI Team Leader)

提供AI專案諮詢，並領導旗下成員進行建構

- Elena Hsieh (AI Engineer)

提供解決方案之選擇

- Steven Chan (Data Science Engineer)

提供API串接架構
