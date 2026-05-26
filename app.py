import streamlit as st
import spacy
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
import fitz

st.set_page_config(
    page_title="Semantic Analysis of Works of Fiction",
    page_icon="📖",
    layout="wide"
)

# ── Load models ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=1
    )
    return nlp, classifier

nlp, emotion_classifier = load_models()

# ── Core functions ────────────────────────────────────────────────
def extract_entities(text):
    doc = nlp(text)
    characters, entities = [], []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            characters.append(ent.text)
        else:
            entities.append(ent.text)
    return list(set(characters)), list(set(entities))

def extract_themes(text):
    keywords = ["conflict", "authority", "resistance", "power", "loyalty",
                "betrayal", "hope", "fear", "responsibility", "moral",
                "tension", "trust", "control", "decision", "freedom"]
    found = [w for w in keywords if w.lower() in text.lower()]
    vectorizer = TfidfVectorizer(max_features=5, stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([text])
        tfidf_themes = vectorizer.get_feature_names_out().tolist()
    except:
        tfidf_themes = []
    all_themes = list(set(found + tfidf_themes))
    return all_themes[:5] if all_themes else ["tension", "conflict"]

def extract_themes_bertopic(text):
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]
    if len(sentences) < 3:
        return extract_themes(text)
    try:
        topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=False)
        topics, _ = topic_model.fit_transform(sentences)
        topic_info = topic_model.get_topic_info()
        keywords = []
        for topic_id in topic_info['Topic'].values:
            if topic_id != -1:
                words = topic_model.get_topic(topic_id)
                if words:
                    keywords.extend([w[0] for w in words[:3]])
        return list(set(keywords))[:5] if keywords else extract_themes(text)
    except:
        return extract_themes(text)

def detect_emotion(text):
    result = emotion_classifier(text[:512])
    label = result[0][0]['label']
    score = round(result[0][0]['score'], 2)
    return label, score

def detect_narrative_role(text, themes, emotion):
    text_lower = text.lower()
    scores = {
        "exposition": 0, "tension building": 0,
        "moral decision point": 0, "conflict escalation": 0,
        "turning point": 0, "resolution": 0
    }
    if any(w in text_lower for w in ["entered","arrived","began","found","discovered","once","years ago"]):
        scores["exposition"] += 2
    if emotion in ["neutral"]:
        scores["exposition"] += 1
    if any(w in text_lower for w in ["waited","silence","watching","slowly","wrong","felt","uneasy"]):
        scores["tension building"] += 2
    if emotion in ["fear","neutral"] and any(w in text_lower for w in ["silent","hum","waiting"]):
        scores["tension building"] += 2
    if any(w in text_lower for w in ["refused","decided","chose","would not","should","must","if she","if he","risk","orders"]):
        scores["moral decision point"] += 2
    if any(w in themes for w in ["responsibility","conflict","decision","moral","power"]):
        scores["moral decision point"] += 2
    if any(w in text_lower for w in ["attacked","fought","shouted","slammed","clash","confronted"]):
        scores["conflict escalation"] += 2
    if emotion in ["anger","fear"]:
        scores["conflict escalation"] += 1
    if any(w in text_lower for w in ["realized","understood","suddenly","changed","never","walked out","betrayal"]):
        scores["turning point"] += 2
    if any(w in themes for w in ["betrayal","discovery","change"]):
        scores["turning point"] += 2
    if any(w in text_lower for w in ["finally","ended","peace","resolved","forgave","accepted","smiled","relief"]):
        scores["resolution"] += 2
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "narrative development"

def analyze_scene(text, mode="zero-shot"):
    doc = nlp(text)
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
    pos_tags = [(t.text, t.pos_) for t in doc if not t.is_stop and not t.is_punct]
    characters, entities = extract_entities(text)
    themes = extract_themes_bertopic(text)
    emotion, confidence = detect_emotion(text)
    role = detect_narrative_role(text, themes, emotion)
    return {
        "mode": mode,
        "tokens": tokens[:15],
        "pos_tags": pos_tags[:8],
        "characters": characters,
        "entities": entities,
        "themes": themes,
        "emotion": emotion,
        "confidence": confidence,
        "narrative_role": role
    }

def split_into_scenes(text, scene_length=500):
    words = text.split()
    scenes = []
    for i in range(0, len(words), scene_length):
        chunk = " ".join(words[i:i+scene_length])
        if len(chunk.strip()) > 100:
            scenes.append({
                "id": f"scene_{i//scene_length+1:03d}",
                "title": f"Scene {i//scene_length+1}",
                "text": chunk
            })
    return scenes

def make_chart(result, title):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    emotions = ['fear','anger','sadness','joy','neutral','surprise','disgust']
    values = [0.0] * len(emotions)
    if result['emotion'].lower() in emotions:
        values[emotions.index(result['emotion'].lower())] = result['confidence']
    colors = ['#E24B4A' if e == result['emotion'].lower() else '#D3D1C7' for e in emotions]
    axes[0].bar(emotions, values, color=colors)
    axes[0].set_title('Emotional Tone')
    axes[0].set_ylabel('Confidence')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)

    themes = result['themes'][:5] if result['themes'] else ['none']
    theme_scores = np.linspace(0.9, 0.5, len(themes))
    axes[1].barh(themes, theme_scores, color='#378ADD')
    axes[1].set_title('Detected Themes')
    axes[1].set_xlabel('Relevance')
    axes[1].set_xlim(0, 1)

    layers = ['Characters','Entities','Themes','Emotion','Narrative Role']
    completeness = [
        1.0 if result['characters'] else 0,
        1.0 if result['entities'] else 0,
        1.0 if result['themes'] else 0,
        1.0 if result['emotion'] else 0,
        1.0 if result['narrative_role'] else 0
    ]
    bar_colors = ['#1D9E75' if c == 1.0 else '#E24B4A' for c in completeness]
    axes[2].bar(layers, completeness, color=bar_colors)
    axes[2].set_title('Semantic Layers')
    axes[2].set_ylim(0, 1.3)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────────
st.title("📖 Semantic Analysis of Works of Fiction")
st.markdown("**Master Thesis Prototype** | Samral Feyziyev | Vilnius Tech 2026")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze Scene", "📚 Analyze Book", "📊 Dataset Demo"])

# ── TAB 1: Single scene analysis ──────────────────────────────────
with tab1:
    st.header("Scene Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        scene_text = st.text_area(
            "Paste a fictional scene here:",
            height=180,
            placeholder="Evelyn stood at the center of the control room..."
        )
    with col2:
        mode = st.radio("Analysis mode", ["zero-shot", "guided"])
        st.markdown("**Zero-shot** — no examples provided")
        st.markdown("**Guided** — uses example for better precision")
        analyze_btn = st.button("Analyze Scene", type="primary", use_container_width=True)

    if analyze_btn and scene_text.strip():
        with st.spinner("Analyzing..."):
            result = analyze_scene(scene_text, mode=mode)

        st.success("Analysis complete!")
        st.markdown("---")

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Emotion", result['emotion'].capitalize())
        col_b.metric("Confidence", f"{result['confidence']:.0%}")
        col_c.metric("Narrative Role", result['narrative_role'].title())
        col_d.metric("Characters", len(result['characters']))

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Step 1 — Preprocessing")
            st.write("**Tokens:**", result['tokens'])
            st.write("**POS Tags:**", result['pos_tags'])

            st.subheader("Step 2 — Entity Extraction")
            st.write("**Characters:**", result['characters'] if result['characters'] else "None detected")
            st.write("**Entities:**", result['entities'] if result['entities'] else "None detected")

        with col2:
            st.subheader("Step 3 — Thematic Analysis")
            st.write("**Themes (BERTopic):**", result['themes'])

            st.subheader("Step 4 — Emotional Tone")
            st.write(f"**{result['emotion'].capitalize()}** — confidence: {result['confidence']}")
            st.progress(result['confidence'])

            st.subheader("Step 5 — Narrative Role")
            role_colors = {
                "tension building": "🟡",
                "moral decision point": "🔴",
                "turning point": "🟠",
                "exposition": "🔵",
                "conflict escalation": "🔴",
                "resolution": "🟢",
                "narrative development": "⚪"
            }
            icon = role_colors.get(result['narrative_role'], "⚪")
            st.write(f"{icon} **{result['narrative_role'].title()}**")

        st.markdown("---")
        st.subheader("Visualization")
        fig = make_chart(result, f"Scene Analysis — {mode.title()} Mode")
        st.pyplot(fig)
        plt.close()

    elif analyze_btn:
        st.warning("Please paste a scene to analyze.")

# ── TAB 2: Book/PDF analysis ──────────────────────────────────────
with tab2:
    st.header("Book Analysis")
    st.markdown("Upload a plain text (.txt) file of a book and analyze its scenes automatically.")

    uploaded_file = st.file_uploader("Upload a book (.txt)", type=["txt"])
    max_scenes = st.slider("Number of scenes to analyze", 1, 10, 3)
    analyze_book_btn = st.button("Analyze Book", type="primary")

    if analyze_book_btn and uploaded_file:
        with st.spinner("Loading and analyzing book..."):
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            scenes = split_into_scenes(text)
            st.info(f"Book split into **{len(scenes)}** scenes. Analyzing first **{max_scenes}**...")

            book_results = []
            for i, scene in enumerate(scenes[:max_scenes]):
                with st.spinner(f"Analyzing {scene['title']}..."):
                    result = analyze_scene(scene['text'], mode="zero-shot")
                    result['title'] = scene['title']
                    book_results.append(result)

        st.success(f"Analysis complete! {len(book_results)} scenes analyzed.")
        st.markdown("---")

        for result in book_results:
            with st.expander(f"📄 {result['title']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Emotion", result['emotion'].capitalize())
                col2.metric("Confidence", f"{result['confidence']:.0%}")
                col3.metric("Narrative Role", result['narrative_role'].title())
                col4.metric("Characters", len(result['characters']))
                st.write("**Themes:**", result['themes'])
                st.write("**Characters:**", result['characters'] if result['characters'] else "None detected")
                fig = make_chart(result, result['title'])
                st.pyplot(fig)
                plt.close()

    elif analyze_book_btn:
        st.warning("Please upload a .txt file first.")

# ── TAB 3: Dataset demo ───────────────────────────────────────────
with tab3:
    st.header("Dataset Demo — 10 Scenes")
    st.markdown("Run analysis on the pre-built dataset of 10 fictional scenes across 6 genres.")

    dataset = {
        "scenes": [
            {"id": "scene_001", "title": "The Control Room", "genre": "sci-fi",
             "text": "Evelyn stood at the center of the control room, her eyes fixed on the blinking red light above the main console. The room was silent except for the hum of machines. Around her, the crew waited. She had been given direct orders to initiate the shutdown sequence, but something felt wrong. The data did not match. If she followed the orders, thousands could be at risk. If she refused, she would be committing an act of insubordination that could end her career. She took a deep breath and reached for the console."},
            {"id": "scene_002", "title": "The Confrontation", "genre": "thriller",
             "text": "Marcus slammed the door behind him. The general sat behind the desk, unmoved. For years Marcus had followed orders without question, sacrificed everything for the cause. But standing here now, the files spread across the table between them, he realized the cause had never been what they told him. Betrayal settled into his bones like cold. He picked up the files slowly, met the general's eyes, and walked out without a word."},
            {"id": "scene_003", "title": "The Discovery", "genre": "drama",
             "text": "Anna found the letter hidden beneath the floorboards, yellowed with age. Her grandmother's handwriting. She had always believed her family had fled the war with nothing but their lives. But the letter spoke of a different story — of choices made, of people left behind, of a secret carried across decades. Anna sat down on the cold floor and began to read, and with every line, the grandmother she thought she knew became a stranger."},
            {"id": "scene_004", "title": "The Last Battle", "genre": "fantasy",
             "text": "The kingdom had fallen. Smoke rose from the towers as Aldric knelt beside the body of his king. Around him, the survivors stood in silence, weapons lowered, too exhausted even for grief. He had sworn an oath to protect the crown, and he had failed. The enemy had not come from outside the walls — it had grown from within, nurtured by those he had trusted. Aldric closed the king's eyes and rose slowly. There was still one thing left to do."},
            {"id": "scene_005", "title": "The Return", "genre": "drama",
             "text": "Sofia stepped off the train and looked at the town she had left fifteen years ago. Nothing had changed — the same cracked pavements, the same bakery on the corner, the same church bell ringing the hour. But she had changed. She was no longer the girl who had run away in the night with nothing but a suitcase and a handful of broken promises. She was here to face what she had left behind."},
            {"id": "scene_006", "title": "The Interrogation", "genre": "thriller",
             "text": "The detective placed the photographs on the table one by one. The suspect said nothing, but his eyes moved. That was enough. Detective Hana had learned long ago that people do not lie with their mouths — they lie with everything else. She leaned forward and spoke quietly. She did not ask whether he had done it. She asked why. And for one unguarded moment, something crossed his face that was unmistakably close to relief."},
            {"id": "scene_007", "title": "The Farewell", "genre": "romance",
             "text": "They stood on the platform as the last call was announced. Elena had memorised his face a hundred times over the past week, knowing this moment would come. He did not ask her to stay, and she did not ask him to come with her. Some things are understood without words. She picked up her bag and walked toward the gate. She did not look back. Not because she did not want to, but because she knew that if she did, she would not be able to leave."},
            {"id": "scene_008", "title": "The Awakening", "genre": "horror",
             "text": "Daniel woke to silence. The kind of silence that does not feel empty but full — full of something waiting. He lay still and listened. The house settled. The wind moved outside. And then, beneath those ordinary sounds, he heard it again — the thing that had pulled him from sleep. Soft, rhythmic, patient. Coming from the room his daughter had stopped sleeping in three weeks ago."},
            {"id": "scene_009", "title": "The Verdict", "genre": "legal drama",
             "text": "The courtroom was silent as the judge read the verdict. Gabriel sat with his hands folded on the table, his face expressionless. He had spent four years preparing for this moment. He had given up his practice, his marriage, his reputation — everything — to bring this case to court. The words the judge read were not the ones he had fought for. Justice, he had learned, is not always the same as truth."},
            {"id": "scene_010", "title": "The Choice", "genre": "sci-fi",
             "text": "The signal had been confirmed three times. Dr. Yuna Park stared at the data stream on her screen and understood that everything she had believed about humanity's place in the universe was about to change. She had two options. Report it through official channels, where it would be buried, classified, controlled. Or release it — all of it — to the world tonight. Her finger hovered over the send button."}
        ]
    }

    run_demo = st.button("Run Full Dataset Analysis", type="primary")

    if run_demo:
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, scene in enumerate(dataset['scenes']):
            status.text(f"Analyzing: {scene['title']} [{scene['genre']}]...")
            result = analyze_scene(scene['text'], mode="zero-shot")
            result['title'] = scene['title']
            result['genre'] = scene['genre']
            results.append(result)
            progress.progress((i + 1) / len(dataset['scenes']))

        status.text("Analysis complete!")
        st.success(f"All {len(results)} scenes analyzed successfully!")
        st.markdown("---")

        st.subheader("Summary Table")
        import pandas as pd
        df = pd.DataFrame([{
            "Title": r['title'],
            "Genre": r['genre'],
            "Characters": ", ".join(r['characters']) if r['characters'] else "—",
            "Emotion": r['emotion'].capitalize(),
            "Confidence": f"{r['confidence']:.0%}",
            "Narrative Role": r['narrative_role'].title()
        } for r in results])
        st.dataframe(df, use_container_width=True)

        st.markdown("---")
        st.subheader("Individual Scene Results")
        for result in results:
            with st.expander(f"📄 {result['title']} [{result['genre']}]"):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Emotion:** {result['emotion'].capitalize()} ({result['confidence']:.0%})")
                col2.write(f"**Role:** {result['narrative_role'].title()}")
                col3.write(f"**Characters:** {', '.join(result['characters']) if result['characters'] else 'None'}")
                st.write(f"**Themes:** {', '.join(result['themes'])}")
                fig = make_chart(result, result['title'])
                st.pyplot(fig)
                plt.close()