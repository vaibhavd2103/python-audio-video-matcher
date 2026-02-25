import base64
import streamlit as st
import streamlit.components.v1 as components
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
    video_bytes = uploaded_video.getvalue()
    st.video(video_bytes)
    mute_original = st.checkbox("Mute original video audio", value=True)

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
                    if st.button(f"Play this audio on video #{idx}", key=f"play_audio_{idx}"):
                        st.session_state["selected_audio_url"] = r["audio_url"]
                        st.session_state["selected_audio_id"] = r["audio_id"]
            else:
                st.error("API error occurred")

    if st.session_state.get("selected_audio_url"):
        audio_url = st.session_state["selected_audio_url"]
        audio_id = st.session_state.get("selected_audio_id", "selected")
        video_mime = uploaded_video.type or "video/mp4"
        video_b64 = base64.b64encode(video_bytes).decode("ascii")

        st.markdown(f"## Video with recommended audio: `{audio_id}`")
        html = f"""
        <div style="max-width: 720px; margin: 0 auto;">
          <div style="display:flex; gap:8px; margin-bottom:8px;">
            <button id="avm-play">Play combined</button>
            <button id="avm-pause">Pause</button>
          </div>
          <video id="avm-video" width="100%" controls {"muted" if mute_original else ""}>
            <source src="data:{video_mime};base64,{video_b64}">
          </video>
          <audio id="avm-audio" src="{audio_url}" preload="auto" crossorigin="anonymous"></audio>
        </div>
        <script>
          const v = document.getElementById("avm-video");
          const a = document.getElementById("avm-audio");
          const playBtn = document.getElementById("avm-play");
          const pauseBtn = document.getElementById("avm-pause");
          let syncing = false;
          function syncAudioToVideo() {{
            if (syncing) return;
            syncing = true;
            try {{ a.currentTime = v.currentTime; }} catch (e) {{}}
            syncing = false;
          }}
          playBtn.addEventListener("click", () => {{
            syncAudioToVideo();
            v.play().catch(() => {{}});
            a.play().catch(() => {{}});
          }});
          pauseBtn.addEventListener("click", () => {{
            v.pause();
            a.pause();
          }});
          v.addEventListener("play", () => {{
            syncAudioToVideo();
            a.play().catch(() => {{}});
          }});
          v.addEventListener("pause", () => a.pause());
          v.addEventListener("seeking", syncAudioToVideo);
          v.addEventListener("timeupdate", () => {{
            if (Math.abs(a.currentTime - v.currentTime) > 0.3) {{
              syncAudioToVideo();
            }}
          }});
          a.addEventListener("play", () => {{
            if (v.paused) v.play().catch(() => {{}});
          }});
          a.addEventListener("pause", () => {{
            if (!v.paused) v.pause();
          }});
        </script>
        """
        components.html(html, height=460)
