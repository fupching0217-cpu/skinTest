import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import pandas as pd

# --- 1. 頁面初始化與標題 ---
st.set_page_config(page_title="皮膚斑點比對系統", layout="wide")
st.title("🔍 皮膚斑點 AI 偵測與療程比對系統")

# --- 2. 初始化 Session State (紀錄偵測結果以免刷新消失) ---
if 'res_a' not in st.session_state:
    st.session_state.res_a = None  # 儲存參照組結果：(影像, 數據字典, 總數)
if 'res_b' not in st.session_state:
    st.session_state.res_b = None  # 儲存對照組結果

# --- 3. 載入模型 ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- 4. 側邊欄參數 ---
st.sidebar.header("AI 參數設定")
brightness = st.sidebar.slider("圖片亮度調整", 0.5, 2.0, 1.0, 0.1)
conf_threshold = st.sidebar.slider("AI 信心度門檻", 0.1, 1.0, 0.25, 0.05)

# --- 5. 核心偵測函式 ---
def perform_detection(uploaded_file):
    image = Image.open(uploaded_file)
    enhancer = ImageEnhance.Brightness(image)
    processed_image = enhancer.enhance(brightness)
    
    img_array = np.array(processed_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    results = model.predict(source=img_bgr, conf=conf_threshold)
    
    # 畫圖並處理類別名稱 (warts -> 斑)
    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    
    counts = {}
    boxes = results[0].boxes
    for box in boxes:
        raw_label = model.names[int(box.cls[0])]
        # 轉換名稱：如果是 warts 則顯示為 斑
        label = "斑" if raw_label.lower() == "warts" else raw_label
        counts[label] = counts.get(label, 0) + 1
        
    return annotated_img, counts, len(boxes)

# --- 6. 介面佈局：左右兩欄 ---
col1, col2 = st.columns(2)

# --- 第一步：參照組 (Reference) ---
with col1:
    st.header("1. 參照組 (治療前)")
    file_a = st.file_uploader("上傳第一張照片", type=["jpg", "png", "jpeg"], key="up_a")
    
    if file_a:
        if st.button("開始偵測參照組"):
            with st.spinner("分析中..."):
                st.session_state.res_a = perform_detection(file_a)
        
    # 如果已有偵測結果，則持續顯示
    if st.session_state.res_a:
        img_a, counts_a, total_a = st.session_state.res_a
        st.image(img_a, caption=f"參照組結果 (總計: {total_a})", use_container_width=True)
        st.write(f"偵測明細: {counts_a}")

# --- 第二步：對照組 (Comparison) ---
with col2:
    st.header("2. 對照組 (治療後)")
    file_b = st.file_uploader("上傳第二張照片", type=["jpg", "png", "jpeg"], key="up_b")
    
    if file_b:
        if st.button("開始偵測對照組"):
            with st.spinner("分析中..."):
                st.session_state.res_b = perform_detection(file_b)
                
    # 如果已有偵測結果，則持續顯示
    if st.session_state.res_b:
        img_b, counts_b, total_b = st.session_state.res_b
        st.image(img_b, caption=f"對照組結果 (總計: {total_b})", use_container_width=True)
        st.write(f"偵測明細: {counts_b}")

# --- 第三步：比對分析 ---
st.divider()
if st.session_state.res_a and st.session_state.res_b:
    if st.button("📊 執行兩張圖片差異比對", use_container_width=True, type="primary"):
        _, counts_a, _ = st.session_state.res_a
        _, counts_b, _ = st.session_state.res_b
        
        all_labels = set(counts_a.keys()).union(set(counts_b.keys()))
        report = []
        
        for label in all_labels:
            num_a = counts_a.get(label, 0)
            num_b = counts_b.get(label, 0)
            
            # 計算減少百分比 (Reduction Percentage)
            # 公式: ((A - B) / A) * 100
            if num_a > 0:
                reduction = ((num_a - num_b) / num_a) * 100
                # 如果結果是負數，代表不減反增
                if reduction >= 0:
                    red_str = f"減少 {reduction:.1f}%"
                else:
                    red_str = f"增加 {abs(reduction):.1f}%"
            else:
                red_str = "無法計算 (初始值為0)" if num_b == 0 else "新增目標"
            
            report.append({
                "偵測項目": label,
                "參照組數量": num_a,
                "對照組數量": num_b,
                "改善程度 (減少百分比)": red_str
            })
            
        st.subheader("比對分析報告")
        st.table(pd.DataFrame(report))
        
        # 針對 "斑" 進行特別總結
        warts_a = counts_a.get("斑", 0)
        warts_b = counts_b.get("斑", 0)
        if warts_a > 0:
            final_red = ((warts_a - warts_b) / warts_a) * 100
            st.success(f"✨ 療程分析總結：您的「斑」點數量從 {warts_a} 處改善至 {warts_b} 處，整體改善率為 {final_red:.1f}%。")
else:
    st.info("請依序完成「參照組」與「對照組」的偵測，即可進行比對分析。")
