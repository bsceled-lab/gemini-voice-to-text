import os
import base64

import streamlit as st
from google import genai


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "gemini-3.8-flash"
MAX_AUDIO_BYTES = 20 * 1024 * 1024  # Gemini inline-audio request limit

st.set_page_config(
    page_title="Voice to Text",
    page_icon="🎙️",
    layout="centered",
)

st.title("🎙️ Voice to Text")
st.caption("Record your voice or upload an audio file, then transcribe it with Gemini Flash.")


def get_api_key() -> str | None:
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets may not be configured locally.
        pass

    return os.getenv("GEMINI_API_KEY")


def transcribe_audio(audio_bytes: bytes, mime_type: str, api_key: str) -> str:
    """Send audio to Gemini and return a clean transcript."""
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(
            "This demo accepts audio up to 20 MB. Please use a shorter recording "
            "or a smaller audio file."
        )

    client = genai.Client(api_key=api_key)

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    response = client.interactions.create(
        model=MODEL_NAME,
        input=[
            {
                "type": "text",
                "text": (
                    "Transcribe the speech in this audio exactly and clearly. "
                    "Preserve the speaker's words and meaning. "
                    "Do not summarize, explain, or add commentary. "
                    "Return only the transcript. "
                    "Use normal punctuation and paragraphs."
                ),
            },
            {
                "type": "audio",
                "data": audio_b64,
                "mime_type": mime_type,
            },
        ],
    )

    transcript = (response.output_text or "").strip()

    if not transcript:
        raise RuntimeError("Gemini returned an empty transcript.")

    return transcript


# -----------------------------
# API key check
# -----------------------------
api_key = get_api_key()

if not api_key:
    st.warning(
        "Gemini API key not found. Add GEMINI_API_KEY to your environment "
        "or Streamlit Secrets before transcribing."
    )
    st.stop()


# -----------------------------
# Audio input
# -----------------------------
st.subheader("1. Record audio")

recorded_audio = st.audio_input(
    "Click the microphone button and record",
    sample_rate=16000,
    help="16 kHz is a good sample rate for speech recognition.",
)

st.subheader("2. Or upload audio")

uploaded_audio = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"],
    help="Maximum size for this demo: 20 MB.",
)

# Prefer a newly uploaded file over a recording if both exist.
audio_source = uploaded_audio if uploaded_audio is not None else recorded_audio

if audio_source is not None:
    audio_bytes = audio_source.getvalue()
    mime_type = getattr(audio_source, "type", None) or "audio/wav"

    st.audio(audio_bytes, format=mime_type)

    st.caption(f"Audio size: {len(audio_bytes) / (1024 * 1024):.2f} MB")

    if st.button("✨ Transcribe", type="primary", use_container_width=True):
        with st.spinner("Transcribing with Gemini Flash..."):
            try:
                transcript = transcribe_audio(
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                    api_key=api_key,
                )

                st.success("Transcription complete!")
                st.subheader("Transcript")
                st.text_area(
                    "Generated transcript",
                    value=transcript,
                    height=300,
                    label_visibility="collapsed",
                )

                st.download_button(
                    "⬇️ Download transcript",
                    data=transcript,
                    file_name="transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
else:
    st.info("Record audio or upload an audio file to get started.")

with st.expander("ℹ️ About this app"):
    st.write(
        "Audio is sent to Google's Gemini API for transcription. "
        "The API key is never hard-coded in this application."
    )
