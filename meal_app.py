import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --- 1. Professional Page Config ---
st.set_page_config(
    page_title="Meal Quality Meter | مقياس جودة الوجبة",
    page_icon="🥗",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Smart Model Loader ---
@st.cache_resource
def load_model():
    # Using 'yolov8n.pt' for speed (Nano version)
    return YOLO('yolov8n.pt')

model = load_model()

# --- 3. Expanded Food Database ---
FOOD_DB = {
    'apple': {'name': 'تفاح', 'cal': 52, 'status': 'Healthy', 'color': (76, 175, 80)}, # Green
    'pizza': {'name': 'بيتزا', 'cal': 266, 'status': 'Unhealthy', 'color': (244, 67, 54)}, # Red
    'banana': {'name': 'موز', 'cal': 89, 'status': 'Healthy', 'color': (255, 235, 59)}, # Yellow
    'orange': {'name': 'برتقال', 'cal': 47, 'status': 'Healthy', 'color': (255, 152, 0)}, # Orange
    'broccoli': {'name': 'بروكلي', 'cal': 34, 'status': 'Healthy', 'color': (56, 142, 60)}, # Dark Green
    'sandwich': {'name': 'ساندويتش', 'cal': 250, 'status': 'Neutral', 'color': (121, 85, 72)} # Brown
}

# --- 4. Sidebar Interface ---
with st.sidebar:
    st.header("⚙️ الإعدادات Settings")
    source = st.radio("مصدر الصورة / Image Source:", ("تحميل صورة 📁", "الكاميرا 📸"))
    conf_level = st.slider("دقة التعرف (Confidence)", 0.1, 1.0, 0.4)
    st.divider()
    st.info("هذا التطبيق يستخدم الذكاء الاصطناعي لتحليل الوجبات")

# --- 5. Main Content Area ---
st.title("🥗 Meal Quality Meter | مقياس جودة الوجبة")
st.write("قم بتصوير وجبتك للحصول على تقييم فوري للسعرات والجودة")

if source == "تحميل صورة 📁":
    uploaded_file = st.file_uploader("اختر صورة الوجبة", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("التقط صورة للوجبة")

if uploaded_file is not None:
    # Processing state
    with st.spinner('جاري التحليل... Analyzing...'):
        image = Image.open(uploaded_file)
        frame = np.array(image)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run Prediction
        results = model(frame_bgr, conf=conf_level)
        total_calories = 0
        items_detected = []

        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])].lower()
                
                if label in FOOD_DB:
                    info = FOOD_DB[label]
                    total_calories += info['cal']
                    items_detected.append(info)
                    
                    # Drawing logic (Bolder lines and text)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), info['color'], 4)
                    label_text = f"{info['name']} ({info['cal']} cal)"
                    cv2.putText(frame_bgr, label_text, (x1, y1 - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, info['color'], 2)

        # UI Layout for Results
        col_img, col_metrics = st.columns([2, 1])

        with col_img:
            # Convert back to RGB for Streamlit display
            final_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(final_img, caption="تحليل الوجبة / Processed Image", use_container_width=True)

        with col_metrics:
            st.subheader("📊 النتائج Results")
            st.metric("إجمالي السعرات (Total)", f"{total_calories} Kcal")
            
            if items_detected:
                # Logic for status
                status_list = [item['status'] for item in items_detected]
                if "Unhealthy" in status_list:
                    st.warning("⚠️ جودة الوجبة: تحتاج تحسين (غير صحية)")
                elif all(s == "Healthy" for s in status_list):
                    st.success("✅ جودة الوجبة: ممتازة (صحية جداً)")
                else:
                    st.info("⚖️ جودة الوجبة: متوازنة")
                
                # List specific items detected
                st.write("**الأصناف المكتشفة:**")
                for item in items_detected:
                    st.write(f"- {item['name']} ({item['cal']} سعرة)")
            else:
                st.error("لم يتم التعرف على أطعمة معروفة")
