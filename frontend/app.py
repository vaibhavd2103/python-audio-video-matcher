import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Video → Audio Recommender", layout="centered")

st.title("🎬 → 🎵 Audio Recommendation System")

uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"]
)

top_k = st.slider("Number of recommendations", 1, 10, 5)

if uploaded_video:
    st.video(uploaded_video)

    if st.button("Recommend Audio"):
        with st.spinner("Analyzing video and ranking audios..."):
            files = {"file": uploaded_video}
            response = requests.post(API_URL, files=files, params={"top_k": top_k})

            if response.status_code == 200:
                results = response.json()
                st.success("Recommendations ready")

                for idx, r in enumerate(results, 1):
                    st.markdown(
                        f"### {idx}. Audio ID: `{r['audio_id']}`  \n"
                        f"Similarity: `{r['score']:.4f}`"
                        )
                    st.audio(r["audio_url"])
            else:
                st.error("API error occurred")
