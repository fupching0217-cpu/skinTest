import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from ultralytics import YOLO
import pandas as pd

# --- 1. 頁面初始化 ---
st.set_page_config(page_title="皮膚斑點 AI 分析系統", layout="wide")
st.title("🔍 皮膚斑點 AI 偵測與綜合改善分析")
st.write("本系統透過比對兩次偵測的「數量」與「信心度」變化，計算綜合改善百分比。")

# --- 2. 初始化 Session State ---
if 'res_a' not in st.session_state:
    st.session_state.res_a = None  # 格式: {img, counts, avg_conf, total_num}
if 'res_b' not in st.session_state:
    st.session_state.res_b = None

# --- 3. 載入模型 ---
@st.cache_resource
def load_model():
    # 載入您的 YOLOv8 模型
    return YOLO("best.pt")

model = load_model()

# --- 4. 核心偵測與數據計算函式 ---
def perform_detection(uploaded_file, bright, conf_thresh):
    # 影像處理
    image = Image.open(uploaded_file)
    enhancer = ImageEnhance.Brightness(image)
    processed_image = enhancer.enhance(bright)
    
    img_array = np.array(processed_image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 執行 AI 偵測
    results = model.predict(source=img_bgr, conf=conf_thresh)
    
    # 取得標註圖 (OpenCV 預設不支援中文標籤，若需圖中顯示中文需額外字體檔，此處先維持原圖標註)
    annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    
    # 統計數據
    boxes = results[0].boxes
    total_num = len(boxes)
    
    # 類別對照表 (可在此擴充其他類別)
    label_map = {"warts": "斑"}
    
    class_stats = {} # 結構: { "斑": [conf1, conf2, ...] }
    
    for box in boxes:
        raw_label = model.names[int(box.cls[0])].lower()
        chn_label = label_map.get(raw_label, raw_label) # 若不在對照表則用原名
        conf = float(box.conf[0])
        
        if chn_label not in class_stats:
            class_stats[chn_label] = []
        class_stats[chn_label].append(conf)
    
    # 計算各類別的平均信心度
    final_stats = {}
    for label, conf_list in class_stats.items():
        final_stats[label] = {
            "數量": len(conf_list),
            "平均信心度": np.mean(conf_list) if conf_list else 0
        }
        
    return annotated_img, final_stats, total_num

# --- 5. 側邊欄設定 ---
st.sidebar.header("⚙️ 偵測參數設定")
brightness_val = st.sidebar.slider("圖片亮度調整", 0.5, 2.0, 1.0, 0.1)
conf_val = st.sidebar.slider("AI 信心度門檻", 0.1, 1.0, 0.25, 0.05)

# --- 6. 介面佈局 ---
col1, col2 = st.columns(2)

# --- 參照組區域 ---
with col1:
    st.header("1. 參照組 (療程前)")
    file_a = st.file_uploader("上傳第一張照片", type=["jpg", "png", "jpeg"], key="up_a")
    
    if file_a and st.button("🔍 執行參照組偵測"):
        with st.spinner("AI 分析中..."):
            img_a, stats_a, total_a = perform_detection(file_a, brightness_val, conf_val)
            st.session_state.res_a = {"img": img_a, "stats": stats_a, "total": total_a}
        
    if st.session_state.res_a:
        res = st.session_state.res_a
        st.image(res["img"], caption=f"參照組偵測結果 (總數: {res['total']})", use_container_width=True)
        # 顯示簡易數據小卡
        for label, data in res["stats"].items():
            st.info(f"📍 {label}：數量 {data['數量']} / 信心度 {data['平均信心度']:.2%}")

# --- 對照組區域 ---
with col2:
    st.header("2. 對照組 (療程後)")
    file_b = st.file_uploader("上傳第二張照片", type=["jpg", "png", "jpeg"], key="up_b")
    
    if file_b and st.button("🔍 執行對照組偵測"):
        with st.spinner("AI 分析中..."):
            img_b, stats_b, total_b = perform_detection(file_b, brightness_val, conf_val)
            st.session_state.res_b = {"img": img_b, "stats": stats_b, "total": total_b}
                
    if st.session_state.res_b:
        res = st.session_state.res_b
        st.image(res["img"], caption=f"對照組偵測結果 (總數: {res['total']})", use_container_width=True)
        for label, data in res["stats"].items():
            st.success(f"📍 {label}：數量 {data['數量']} / 信心度 {data['平均信心度']:.2%}")

# --- 7. 綜合比對分析 ---
st.divider()
if st.session_state.res_a and st.session_state.res_b:
    if st.button("📊 執行雙圖數據綜合對比", use_container_width=True, type="primary"):
        stats_a = st.session_state.res_a["stats"]
        stats_b = st.session_state.res_b["stats"]
        
        all_labels = set(stats_a.keys()).union(set(stats_b.keys()))
        report_list = []
        
        for label in all_labels:
            data_a = stats_a.get(label, {"數量": 0, "平均信心度": 0})
            data_b = stats_b.get(label, {"數量": 0, "平均信心度": 0})
            
            q_a, c_a = data_a["數量"], data_a["平均信心度"]
            q_b, c_b = data_b["數量"], data_b["平均信心度"]
            
            # --- 計算邏輯 ---
            # 1. 數量減少率
            q_reduction = ((q_a - q_b) / q_a * 100) if q_a > 0 else (0 if q_b == 0 else -100)
            
            # 2. 信心度減少率 (信心度下降通常代表病灶變淡)
            c_reduction = ((c_a - c_b) / c_a * 100) if c_a > 0 else 0
            
            # 3. 綜合改善率 (兩者相加平均)
            combined_improvement = (q_reduction + c_reduction) / 2
            
            report_list.append({
                "偵測項目": label,
                "參照數量": q_a,
                "對照數量": q_b,
                "參照信心度": f"{c_a:.2%}",
                "對照信心度": f"{c_b:.2%}",
                "綜合改善率": f"{combined_improvement:.2f}%"
            })
            
        st.subheader("📋 雙圖差異對比報表")
        st.table(pd.DataFrame(report_list))
        
        # 針對「斑」的特別總結
        if "斑" in all_labels:
            imp_val = float(next(item["綜合改善率"].replace('%','') for item in report_list if item["偵測項目"] == "斑"))
            if imp_val > 0:
                st.balloons()
                st.success(f"✨ 分析結果：您的「斑」點狀況有明顯改善，綜合改善率達 {imp_val}%！")
            else:
                st.warning(f"💡 分析結果：目前「斑」點的綜合指標未見顯著減少，建議持續觀察。")
else:
    st.info("請依序完成左右兩組影像的 AI 偵測，系統將自動計算改善百分比。")
