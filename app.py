# ==========================================================
# 🎵 EmotiSync v4 — Mood-Based Tamil + English Music Recommender
# ==========================================================
# ✅ Uses free APIs (YouTube Music + JioSaavn)
# ✅ Supports Tamil + English songs
# ✅ Hugging Face ViT + DeepFace for emotion
# ✅ No API keys, no expiry
# ==========================================================

import os
import tempfile
import numpy as np
import cv2
import librosa
import requests
import streamlit as st
from PIL import Image
import torch

st.set_page_config(page_title="🎵 EmotiSync — Mood-Based Tamil + English Music", layout="centered")
st.title("🎵 EmotiSync: Mood-Based Tamil + English Music Generator")

st.caption("Uses Hugging Face ViT + DeepFace + Free JioSaavn & YouTube APIs")

# ------------------------------------------------
# ✅ Optional Libraries
# ------------------------------------------------
try:
    from deepface import DeepFace
    USE_DEEPFACE = True
except:
    USE_DEEPFACE = False

try:
    from ytmusicapi import YTMusic
    YT_AVAILABLE = True
except:
    YT_AVAILABLE = False

# ------------------------------------------------
# 🧠 Hugging Face Face Emotion Model
# ------------------------------------------------
from transformers import AutoImageProcessor, AutoModelForImageClassification

USE_HF_FACE_MODEL = False
try:
    processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
    model_hf = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
    USE_HF_FACE_MODEL = True
    print("✅ Hugging Face ViT face model loaded.")
except Exception as e:
    USE_HF_FACE_MODEL = False
    print("⚠️ Hugging Face face emotion model not loaded:", e)

# ------------------------------------------------
# Emotion Setup
# ------------------------------------------------
EMOTIONS = ["angry","disgust","fear","happy","sad","surprise","neutral","calm","energetic","excited"]
DISPLAY = {e: e.capitalize() for e in EMOTIONS}

# ------------------------------------------------
# 🎭 Face Emotion Detection
# ------------------------------------------------
def face_emotion_probs(img_bgr):
    """
    Try Hugging Face first; fallback to DeepFace if unavailable.
    Returns emotion probabilities.
    """
    # --- Hugging Face ViT Model ---
    if USE_HF_FACE_MODEL:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_rgb)
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model_hf(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            labels = model_hf.config.id2label
            results = {labels[i].lower(): float(probs[i]) for i in range(len(labels))}
            s = sum(results.values())
            for k in results:
                results[k] /= s
            return results
        except Exception as e:
            print("⚠️ HF face emotion error:", e)

    # --- DeepFace fallback ---
    if USE_DEEPFACE:
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = DeepFace.analyze(img_rgb, actions=["emotion"], enforce_detection=False)
            if isinstance(res, list):
                res = res[0]
            emo_probs = res.get("emotion", {})
            if not emo_probs:
                return None
            probs = np.array([emo_probs.get(e, emo_probs.get(e.capitalize(), 0.0)) for e in EMOTIONS])
            probs = probs / probs.sum() if probs.sum() else np.ones(len(EMOTIONS)) / len(EMOTIONS)
            return {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
        except Exception as e:
            print("⚠️ DeepFace fallback error:", e)

    return None

# ------------------------------------------------
# 🎧 Audio Emotion Estimation (Librosa)
# ------------------------------------------------
def audio_emotion_probs(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=5.0)
        if len(y) == 0:
            return None
        rms = np.mean(librosa.feature.rms(y=y))
        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        loud = 20*np.log10(rms+1e-6)

        scores = {e:0.01 for e in EMOTIONS}
        if loud > -18 and tempo > 110:
            scores["energetic"] += 3
            scores["excited"] += 2
        if loud < -25 and tempo < 90:
            scores["sad"] += 2
            scores["calm"] += 1.5
        if spec_centroid > 3000:
            scores["happy"] += 2
        if zcr > 0.08:
            scores["angry"] += 1.8
        arr = np.array(list(scores.values()))
        arr = arr / arr.sum()
        return {EMOTIONS[i]: float(arr[i]) for i in range(len(EMOTIONS))}
    except:
        return None

# ------------------------------------------------
# 🔗 Combine Emotion Results
# ------------------------------------------------
def combine_probs(face_probs, audio_probs):
    if face_probs is None and audio_probs is None:
        return {e: 1.0/len(EMOTIONS) for e in EMOTIONS}
    if face_probs is None:
        return audio_probs
    if audio_probs is None:
        return face_probs
    combined = {}
    for e in EMOTIONS:
        combined[e] = (0.55*face_probs.get(e,0.0) + 0.45*audio_probs.get(e,0.0))
    s = sum(combined.values())
    for k in combined:
        combined[k] = combined[k]/s
    return combined

# ------------------------------------------------
# 🎶 Music Search Functions
# ------------------------------------------------
def youtube_music_search(query, limit=8):
    if not YT_AVAILABLE:
        return []
    try:
        ytm = YTMusic()
        res = ytm.search(query, filter="songs")
        songs = []
        for r in res[:limit]:
            title = r.get("title", "Unknown")
            artist = r.get("artists")[0]["name"] if r.get("artists") else ""
            vid = r.get("videoId", "")
            url = f"https://music.youtube.com/watch?v={vid}" if vid else ""
            songs.append({"title": title, "artist": artist, "url": url})
        return songs
    except:
        return []

def jiosaavn_search(query, limit=8):
    try:
        url = f"https://saavn.dev/api/search/songs?query={query}"
        res = requests.get(url, timeout=6)
        res.raise_for_status()
        data = res.json().get("data", {}).get("results", [])
        songs = []
        for s in data[:limit]:
            songs.append({
                "title": s.get("name"),
                "artist": ", ".join(a["name"] for a in s.get("artists", {}).get("primary", [])),
                "url": s.get("url")
            })
        return songs
    except:
        return []

# ------------------------------------------------
# 🎨 Streamlit Interface
# ------------------------------------------------
if USE_HF_FACE_MODEL:
    st.success("✅ Using Hugging Face ViT model for face emotion detection.")
elif USE_DEEPFACE:
    st.info("🧠 Using DeepFace for face emotion detection.")
else:
    st.warning("⚠️ No face emotion model loaded. Please install transformers or deepface.")

col1, col2 = st.columns(2)
with col1:
    image_file = st.file_uploader("📸 Upload Face Image", type=["jpg","jpeg","png"])
with col2:
    audio_file = st.file_uploader("🎤 Upload Voice Clip", type=["wav","mp3","m4a"])

lang = st.selectbox("🎧 Music Language", ["Tamil", "English", "Both"], index=0)

if st.button("🔍 Analyze Emotion"):
    if not image_file and not audio_file:
        st.warning("Please upload at least one input.")
    else:
        face_probs, audio_probs = None, None

        # --- Image ---
        if image_file:
            arr = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            with st.spinner("Analyzing face emotion..."):
                face_probs = face_emotion_probs(img)

        # --- Audio ---
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                tf.write(audio_file.read())
                temp_path = tf.name
            st.audio(temp_path)
            with st.spinner("Analyzing voice emotion..."):
                audio_probs = audio_emotion_probs(temp_path)
            os.remove(temp_path)

        # --- Combine Results ---
        final_probs = combine_probs(face_probs, audio_probs)
        top_emotion = max(final_probs, key=final_probs.get)
        st.success(f"🎭 Detected Emotion: **{DISPLAY[top_emotion]}**")

        # --- Build Search Query ---
        base_query = f"{top_emotion} mood songs"
        if lang == "Tamil":
            query = f"Tamil {base_query}"
        elif lang == "Both":
            query = f"Tamil and English {base_query}"
        else:
            query = f"English {base_query}"

        st.markdown(f"#### 🔎 Searching for: **{query}**")

        # --- Music Search ---
        tracks = youtube_music_search(query)
        if not tracks:
            tracks = jiosaavn_search(query)

        if tracks:
            st.markdown("### 🎶 Suggested Songs:")
            for t in tracks:
                name = t.get("title", "Unknown")
                artist = t.get("artist", "")
                link = t.get("url", "")
                if link:
                    st.markdown(f"- [{name} — {artist}]({link})")
                else:
                    st.write(f"- {name} — {artist}")
        else:
            st.info("No songs found. Try switching language or re-upload clearer inputs.")
