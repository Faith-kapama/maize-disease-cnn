import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "maize_cnn.h5"
IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.7

CLASS_LABELS = ['blight', 'common_rust', 'gray_leaf_spot', 'healthy', 'others']

EXAMPLE_IMAGES = {
    'blight': "data/test/Blight/Corn_Blight (688).JPG",
    'common_rust': "data/test/Common_Rust/Corn_Common_Rust (611).JPG",
    'gray_leaf_spot': "data/test/Gray_Leaf_Spot/augmented_Corn_Gray_Spot (404).JPG",
}

LANGUAGES = {"en": "English", "sw": "Kiswahili"}


@st.cache_data
def load_content(lang):
    return json.loads((BASE / "content" / f"{lang}.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_cnn():
    return load_model(MODEL_PATH)


if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "nav" not in st.session_state:
    st.session_state["nav"] = "Upload"

content = load_content(st.session_state["lang"])
ui = content["ui"]

st.set_page_config(page_title=ui["page_title"], page_icon="🌽", layout="wide")
st.markdown(f"<style>{(BASE / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)

model = load_cnn()


def preprocess_image(uploaded_file):
    try:
        original_img = Image.open(uploaded_file).convert("RGB")
        img = original_img.resize(IMG_SIZE, Image.NEAREST)
        img_array = np.asarray(img, dtype="float32")[np.newaxis] / 255.0
        return img_array, original_img
    except Exception:
        return None, None


def load_example_image(rel_path):
    full_path = BASE / rel_path
    if full_path.exists():
        return Image.open(full_path)
    return None


def confidence_gauge(value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'color': 'white', 'size': 28}},
        title={'text': ui["confidence"], 'font': {'color': '#8899aa', 'size': 13}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#334', 'tickfont': {'color': '#8899aa'}},
            'bar': {'color': color},
            'bgcolor': '#1a1f2e',
            'bordercolor': '#2e3a50',
            'steps': [
                {'range': [0, CONFIDENCE_THRESHOLD * 100], 'color': '#12151e'},
                {'range': [CONFIDENCE_THRESHOLD * 100, 100], 'color': '#1a1f2e'},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40, b=10, l=30, r=30),
        height=200,
        font_color='white',
    )
    return fig


def nav_button(column, page, label):
    active = st.session_state["nav"] == page
    if column.button(label, width="stretch", type="primary" if active else "secondary"):
        st.session_state["nav"] = page
        st.rerun()


st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
col_logo, col_nav, col_lang = st.columns([2, 4, 2])

with col_logo:
    st.markdown(f"### 🌽 {ui['brand']}")

with col_nav:
    n1, n2, n3 = st.columns(3)
    nav_button(n1, "Upload", ui["nav_upload"])
    nav_button(n2, "Results", ui["nav_results"])
    nav_button(n3, "Help", ui["nav_help"])

with col_lang:
    st.radio(
        "Language",
        options=list(LANGUAGES),
        format_func=LANGUAGES.get,
        key="lang",
        horizontal=True,
        label_visibility="collapsed",
    )

st.markdown("<hr style='border-color:#1e2535;margin:8px 0 24px 0'>", unsafe_allow_html=True)
nav = st.session_state["nav"]


if nav == "Upload":
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"## {ui['upload_title']}")
        st.markdown(ui["upload_intro"])

        uploaded_file = st.file_uploader(
            ui["upload_label"], type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        if not uploaded_file:
            st.markdown(f"<div class='upload-hint'>{ui['upload_hint']}</div>", unsafe_allow_html=True)
        else:
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["nav"] = "Results"
            st.rerun()


elif nav == "Results":
    if "uploaded_file" not in st.session_state:
        st.warning(ui["no_upload"])
        st.stop()

    img_array, display_img = preprocess_image(st.session_state["uploaded_file"])
    if img_array is None:
        st.error(ui["invalid_image"])
        st.stop()

    pred = model.predict(img_array, verbose=0)
    pred_class = CLASS_LABELS[int(np.argmax(pred))]
    pred_prob = float(np.max(pred))

    if pred_class == "others" or pred_prob < CONFIDENCE_THRESHOLD:
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.image(display_img, width="stretch")
            if pred_class == "others":
                st.error(ui["not_maize"])
            else:
                st.warning(ui["low_confidence"].format(confidence=f"{pred_prob:.0%}"))
        st.stop()

    is_healthy = pred_class == "healthy"
    label = content["labels"][pred_class]
    card_class = "card-healthy" if is_healthy else "card-disease"
    status_icon = "✅" if is_healthy else "⚠️"
    accent = "#2E7D32" if is_healthy else "#E65100"

    img_col, result_col = st.columns([1, 1], gap="large")

    with img_col:
        st.image(display_img, caption=ui["uploaded_caption"], width=380)

    with result_col:
        st.markdown(f"""
        <div class="{card_class}">
            <div class="card-label">{ui['result_label']}</div>
            <div class="card-value">{status_icon} {label}</div>
            <div class="card-confidence">{ui['confidence']}: {pred_prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(confidence_gauge(pred_prob, accent), width="stretch",
                        config={'displayModeBar': False})

    st.markdown("<hr style='border-color:#1e2535;margin:24px 0'>", unsafe_allow_html=True)

    if is_healthy:
        st.markdown(f"#### {ui['healthy_title']}")
        st.markdown(f"<div class='section-header'>{ui['healthy_header']}</div>", unsafe_allow_html=True)
        for tip in content["healthy_practices"]:
            st.markdown(f"<div class='info-card-green'>🌱 &nbsp;{tip}</div>", unsafe_allow_html=True)

    else:
        info = content["diseases"][pred_class]
        tab1, tab2, tab3, tab4 = st.tabs([
            ui["tab_overview"], ui["tab_prevention"], ui["tab_treatment"], ui["tab_resources"]
        ])

        with tab1:
            desc_col, img_col2 = st.columns([3, 2], gap="large")
            with desc_col:
                st.markdown(info["description"])
                st.markdown(f"<div class='section-header' style='margin-top:16px'>{ui['symptoms_header']}</div>",
                            unsafe_allow_html=True)
                for s in info["symptoms"]:
                    st.markdown(f"<div class='info-card'>• &nbsp;{s}</div>", unsafe_allow_html=True)
            with img_col2:
                example_img = load_example_image(EXAMPLE_IMAGES[pred_class])
                if example_img:
                    st.image(example_img, caption=ui["example_caption"].format(label=label), width="stretch")
                else:
                    st.info(ui["no_example"])

        with tab2:
            st.markdown(f"<div class='section-header'>{ui['prevention_header']}</div>", unsafe_allow_html=True)
            for tip in info["prevention"]:
                st.markdown(f"<div class='info-card'>🌱 &nbsp;{tip}</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown(f"<div class='section-header'>{ui['treatment_header']}</div>", unsafe_allow_html=True)
            for step in info["treatment_steps"]:
                st.markdown(f"<div class='info-card-amber'>💊 &nbsp;{step}</div>", unsafe_allow_html=True)

        with tab4:
            if info["resources"]:
                st.markdown(info["resources"])
            else:
                st.info(ui["no_resources"])


elif nav == "Help":
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        st.markdown(f"## {ui['help_title']}")
        st.markdown(ui["help_body"])
