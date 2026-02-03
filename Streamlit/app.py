import streamlit as st
import tempfile
import os

import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# ===============================
# Environment
# ===============================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Speaker Diarization", layout="centered")
st.title("🎙️ Speaker Diarization")
st.caption("Upload a WAV file to see speaker-wise segments")

# ===============================
# WAV-only upload
# ===============================
uploaded_file = st.file_uploader(
    "Upload WAV audio file",
    type=["wav"]
)

# ===============================
# Audio conversion (safety)
# ===============================
def convert_audio_to_16k_mono(input_path, output_path):
    waveform, sample_rate = torchaudio.load(input_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )
        waveform = resampler(waveform)

    torchaudio.save(output_path, waveform, 16000)

# ===============================
# Load diarization pipeline
# ===============================
@st.cache_resource
def load_pipeline():
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_TOKEN
    )

pipeline = load_pipeline()

# ===============================
# Run diarization
# ===============================
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as raw:
        raw.write(uploaded_file.read())
        raw_path = raw.name

    processed_path = raw_path.replace(".wav", "_16k.wav")
    convert_audio_to_16k_mono(raw_path, processed_path)

    # Audio player
    st.audio(processed_path)

    with st.spinner("Running speaker diarization..."):
        diarization = pipeline(
            processed_path,
            min_speakers=1,
            max_speakers=3
        )

    st.success("Diarization complete")

    # ===============================
    # Caption-style speaker timeline
    # ===============================
    st.subheader("📝 Speaker Timeline")

    speaker_map = {}
    speaker_count = 1

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {speaker_count}"
            speaker_count += 1

        label = speaker_map[speaker]
        start = f"{turn.start:.2f}"
        end = f"{turn.end:.2f}"

        st.markdown(
            f"""
            **[{start}s – {end}s]**  
            🔊 {label}
            ---
            """
        )