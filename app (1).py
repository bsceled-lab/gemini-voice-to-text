import os
import tempfile

import streamlit as st
from google import genai

MODEL_NAME = "gemini-3.5-transcribe"

st.set_page_config(page_title="Voice to Text", page_icon="🎙️")

st.title("🎙️ Voice to Text")
st.caption("Record or upload audio and transcribe it with Gemini 3.5 Transcribe.")


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key else None


def transcribe_audio(audio_bytes, mime_type, filename, api_key):
    client = genai.Client(api_key=api_key)

    suffix = os.path.splitext(filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        audio_file = client.files.upload(
            file=temp_path,
            config={"mime_type": mime_type},
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[audio_file],
        )

        transcript = (response.text or "").strip()
        if not transcript:
            raise RuntimeError("Gemini returned an empty transcript.")

        return transcript
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


api_key = get_api_key()

if not api_key:
    st.error("Gemini API key not found.")
    st.info(
        'Create ".streamlit/secrets.toml" and add: '
        'GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"'
    )
    st.stop()

st.subheader("1. Record audio")
recorded_audio = st.audio_input(
    "Click the microphone button to record",
    sample_rate=16000,
    help="16 kHz is optimized for speech recognition.",
)

st.subheader("2. Or upload audio")
uploaded_audio = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"],
)

audio_source = uploaded_audio if uploaded_audio is not None else recorded_audio

if audio_source is None:
    st.info("Record something or upload an audio file to begin.")
    st.stop()

audio_bytes = audio_source.getvalue()
mime_type = getattr(audio_source, "type", None) or "audio/wav"
filename = getattr(audio_source, "name", None) or "recording.wav"

st.audio(audio_bytes, format=mime_type)
st.caption(f"File: {filename} | Size: {len(audio_bytes) / 1024 / 1024:.2f} MB")

if st.button("✨ Transcribe", type="primary", use_container_width=True):
    with st.spinner("Uploading audio and transcribing..."):
        try:
            transcript = transcribe_audio(
                audio_bytes, mime_type, filename, api_key
            )

            st.success("Transcription complete!")
            st.subheader("Transcript")
            st.text_area(
                "Transcript",
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
            message = str(exc)

            if "401" in message or "UNAUTHENTICATED" in message:
                st.error(
                    "Gemini rejected the API key. Verify that it is a valid "
                    "Gemini API key and that it is stored correctly."
                )
            else:
                st.error(f"Transcription failed: {message}")


with st.expander("🔐 API key setup"):
    st.markdown(
        """Create `.streamlit/secrets.toml` locally:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Do not commit this file to GitHub.

On Streamlit Community Cloud, add the same value under your app's **Settings → Secrets**.
"""
    )
