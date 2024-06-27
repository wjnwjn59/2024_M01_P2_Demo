import os
import base64
import streamlit as st

from utils import save_upload_file, delete_file, download_model
from models.yolov10.detector import inference
from config.model_config import Detector_Config
from components.streamlit_footer import footer

@st.cache_data(max_entries=1000)
def process_and_display_image(image_path):
    result_img = inference(image_path,
                           weight_path=Detector_Config.export_weight_path,
                           yaml_path=Detector_Config.yaml_path)
    st.markdown('**Detection result**')
    st.image(result_img)

def main():
    st.set_page_config(
        page_title="AIO2024 Module01 Project YOLOv10 - AI VIETNAM",
        page_icon='static/aivn_favicon.png',
        layout="wide"
    )

    col1, col2 = st.columns([0.8, 0.2], gap='large')
    
    with col1:
        st.title('AIO2024 - Module01 - Image Project')
        st.title(':sparkles: :blue[YOLOv10] Helmet Safety Detection Demo')
        
    with col2:
        logo_img = open("static/aivn_logo.png", "rb").read()
        logo_base64 = base64.b64encode(logo_img).decode()
        st.markdown(
            f"""
            <a href="https://aivietnam.edu.vn/">
                <img src="data:image/png;base64,{logo_base64}" width="full">
            </a>
            """,
            unsafe_allow_html=True,
        )

    uploaded_img = st.file_uploader('__Input your image__', type=['jpg', 'jpeg', 'png'])
    example_button = st.button('Run example')

    st.divider()

    if example_button:
        process_and_display_image('static/example_img.jpg')

    if uploaded_img:
        uploaded_img_path = save_upload_file(uploaded_img)
        try:
            process_and_display_image(uploaded_img_path)
        finally:
            delete_file(uploaded_img_path)

    footer()


if __name__ == '__main__':
    if not os.path.exists(Detector_Config.origin_weight_path):
        download_model()
    main()