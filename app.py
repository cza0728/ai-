import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. 设置页面标题
# ==========================================
st.set_page_config(page_title="我的 AI 识别助手", page_icon="🤖")
st.title("🤖 CIFAR-10 图像识别小工具")
st.write("上传一张图片（'Plane ✈️', 'Car 🚗', 'Bird 🐦', 'Cat 🐱', 'Deer 🦌', 'Dog 🐶', 'Frog 🐸', 'Horse 🐴', 'Ship 🚢', 'Truck 🚚'），让我来猜猜它是什么！")

# ==========================================
# 2. 准备标签和预处理
# ==========================================
classes = ('Plane ✈️', 'Car 🚗', 'Bird 🐦', 'Cat 🐱', 'Deer 🦌',
           'Dog 🐶', 'Frog 🐸', 'Horse 🐴', 'Ship 🚢', 'Truck 🚚')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. 加载模型 (核心！)
# ==========================================
# @st.cache_resource 是 Streamlit 的黑科技
# 它会把加载好的模型存在缓存里。
# 这样你每次点按钮时，就不用重新花几秒钟去加载模型了，速度飞快！
@st.cache_resource
def load_model():
    # 搭建空壳
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    # 注入灵魂
    # 【注意】确保 resnet18_cifar10.pth 和这个代码在同一个文件夹！
    state_dict = torch.load('./resnet18_cifar10.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 加载模型（这一步在网页打开时只会运行一次）
try:
    model = load_model()
    st.success("✅ 模型加载成功！")
except FileNotFoundError:
    st.error("❌ 找不到模型文件！请确认 'resnet18_cifar10.pth' 在当前目录下。")

# ==========================================
# 4. 网页交互逻辑
# ==========================================
# 创建一个文件上传框
uploaded_file = st.file_uploader("请选择一张图片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. 显示用户上传的图
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='你上传的图片', use_container_width=True)
    
    # 2. 当用户点击“开始识别”按钮
    if st.button('开始识别'):
        # 显示一个转圈圈的加载条
        with st.spinner('AI 正在思考中...'):
            # 预处理
            img_tensor = transform(image).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # 找最大值
                value, index = torch.max(output, 1)
                predicted_class = classes[index.item()]
                confidence = probabilities[index.item()].item()
            
            # 3. 展示结果
            st.markdown(f"### 我觉得它是： **{predicted_class}**")
            
            # 显示置信度进度条
            st.progress(confidence)
            st.write(f"置信度: {confidence*100:.2f}%")
            
            # 如果置信度太低，吐槽一下
            if confidence < 0.5:
                st.warning("🤔 我不太确定，这图是不是有点糊？")
            elif confidence > 0.9:
                st.balloons() # 放个气球庆祝一下！