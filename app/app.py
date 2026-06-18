import os
import tempfile

import streamlit as st
import torch

from audio_io import (
    load_audio_mono,
    pad_or_trim,
    peak_normalize,
    to_torch_1x1xT,
    save_wav,
)
from flow_separator import FlowSeparator, SR as SEP_SR, DATASET_LEN


# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="Selective Hearing", layout="centered")
st.title("Prompt-Conditioned Selective Hearing")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Device: **{DEVICE.upper()}**")


# --------------------------
# Paths
# --------------------------
FLOW_CKPT = st.text_input("Flow separator checkpoint", "../checkpoints/best.pt")


# --------------------------
# Cache model
# --------------------------
@st.cache_resource
def load_separator(ckpt, device):
    return FlowSeparator(ckpt, device)


# --------------------------
# Check checkpoint exists
# --------------------------
if not os.path.exists(FLOW_CKPT):
    st.error(f"Missing Flow checkpoint: {FLOW_CKPT}")
    st.stop()

separator = load_separator(FLOW_CKPT, DEVICE)


# --------------------------
# Upload audio
# --------------------------
st.subheader("Upload mixture audio")
uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])

if uploaded is None:
    st.stop()

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
tmp.write(uploaded.read())
tmp.flush()
audio_path = tmp.name

st.subheader("Original audio")
st.audio(audio_path)


# --------------------------
# Choose what to isolate
# --------------------------
st.subheader("Selective hearing (prompt isolation)")

TRAINED_PROMPTS = ["dog barking", "human speech", "music"]

choice = st.selectbox(
    "Choose what to isolate",
    TRAINED_PROMPTS + ["Custom prompt..."],
)

if choice == "Custom prompt...":
    prompt = st.text_input("Enter a custom prompt", "dog barking")
else:
    prompt = choice

steps = st.slider("Inference steps", 60, 250, 120, 10)


# --------------------------
# Run isolation
# --------------------------
if st.button("Run isolation"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner("Isolating sound... please wait"):
        y = load_audio_mono(audio_path, sr=SEP_SR)
        y = pad_or_trim(y, DATASET_LEN)
        y = peak_normalize(y, 0.95)

        mix_t = to_torch_1x1xT(y, DEVICE)

        out = separator.isolate(mix_t, prompt=prompt, steps=steps)
        out = peak_normalize(out, 0.95)

        out_path = os.path.join(tempfile.gettempdir(), "isolated.wav")
        save_wav(out_path, out, SEP_SR)

        st.session_state["isolated_path"] = out_path
        st.session_state["isolated_prompt"] = prompt


# --------------------------
# Output
# --------------------------
isolated_path = st.session_state.get("isolated_path", None)

if isolated_path and os.path.exists(isolated_path):
    st.subheader("Isolated output")
    st.write("Prompt:", st.session_state.get("isolated_prompt", ""))
    st.audio(isolated_path)

    with open(isolated_path, "rb") as f:
        st.download_button(
            "Download isolated.wav",
            f,
            file_name="isolated.wav",
            mime="audio/wav",
        )