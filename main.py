import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import pandas as pd

# --- 設定頁面標題 ---
st.set_page_config(page_title="皮膚偵測比對系統", layout="wide")
st.title("🔍 皮膚偵測與差異分析系統")
st.write("請上傳兩張圖片（例如：治療前與治療後），系統將自動比對偵測目標的數量差異。")

# --- 載入模型 (快取處理) ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 側邊欄設定 ---
st.sidebar.header("全域參數設定")
brightness = st.sidebar.slider("圖片亮度調整", 0.5, 2.0, 1.0, 0.1)
conf_threshold = st.sidebar.slider("AI 信心度門檻", 0.1, 1.0, 0.25, 0.05)

# --- 定義偵測函式 ---
def process_and_detect(uploaded_file, brightness, conf_threshold):
    if uploaded_file is None:
        return None, None, None
    
    # 1. 影像處理
    image = Image.open(uploaded_file)
    enhancer = ImageEnhance.Brightness(image)
    processed_image = enhancer.enhance(brightness)
    
    # 2. 轉換為 OpenCV 格式
    img_array = np.array(processed_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 3. YOLO 偵測
    results = model.predict(source=img_bgr, conf=conf_threshold)
    
    # 4. 取得畫框後的圖片
    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    
    # 5. 統計各類別數量
    counts = {}
    boxes = results[0].boxes
    for box in boxes:
        label = model.names[int(box.cls[0])]
        counts[label] = counts.get(label, 0) + 1
        
    return annotated_img, counts, len(boxes)

# --- 圖片上傳區域 (分兩欄) ---
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.subheader("圖片 A (參照組)")
    file_a = st.file_uploader("選擇第一張照片...", type=["jpg", "jpeg", "png"], key="file_a")

with col_up2:
    st.subheader("圖片 B (對照組)")
    file_b = st.file_uploader("選擇第二張照片...", type=["jpg", "jpeg", "png"], key="file_b")

# --- 執行偵測與比對 ---
if file_a and file_b:
    if st.button("🚀 開始執行雙圖偵測與比對分析", use_container_width=True):
        with st.spinner('AI 分析中...'):
            # 分別偵測兩張圖片
            img_a_res, counts_a, total_a = process_and_detect(file_a, brightness, conf_threshold)
            img_b_res, counts_b, total_b = process_and_detect(file_b, brightness, conf_threshold)
            
            # 顯示偵測結果圖
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.image(img_a_res, caption=f"圖片 A 偵測結果 (總計: {total_a})", use_container_width=True)
            with res_col2:
                st.image(img_b_res, caption=f"圖片 B 偵測結果 (總計: {total_b})", use_container_width=True)
            
            # --- 差異比對邏輯 ---
            st.divider()
            st.subheader("📊 目標差異分析報告")
            
            # 整合所有出現過的類別
            all_labels = set(counts_a.keys()).union(set(counts_b.keys()))
            
            comparison_data = []
            for label in all_labels:
                num_a = counts_a.get(label, 0)
                num_b = counts_b.get(label, 0)
                
                # 計算差異百分比 (以圖片 A 為基準)
                if num_a > 0:
                    diff_pct = ((num_b - num_a) / num_a) * 100
                    diff_str = f"{diff_pct:+.2f}%"
                else:
                    diff_str = "新增目標" if num_b > 0 else "0%"
                
                comparison_data.append({
                    "偵測目標": label,
                    "圖片 A 數量": num_a,
                    "圖片 B 數量": num_b,
                    "差異程度 (B vs A)": diff_str
                })
            
            # 使用表格呈現
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                st.table(df)
                
                # 額外數據總結
                st.info(f"💡 分析總結：圖片 B 相較於圖片 A，總偵測數量由 {total_a} 變更為 {total_b}。")
            else:
                st.warning("兩張圖片皆未偵測到任何目標。")

elif file_a or file_b:
    st.info("💡 請上傳兩張圖片以啟動比對功能。")
