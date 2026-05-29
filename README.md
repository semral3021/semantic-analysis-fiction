# Semantic Analysis of Works of Fiction

**Master Thesis Prototype**  
**Author:** Samral Feyziyev  
**University:** Vilnius Tech  
**Year:** 2026  

---

## Project Description

This prototype implements a structured five-layer semantic analysis pipeline 
specifically designed for fictional narrative text. The system automatically 
extracts semantic meaning from fictional scenes across five analytical layers:

- **Characters** — who is present in the scene
- **Entities** — important objects, locations, and concepts
- **Themes** — dominant semantic keywords using BERTopic
- **Emotional Tone** — emotion label and confidence score using transformer model
- **Narrative Role** — function of the scene in the story

The prototype supports two analysis modes:
- **Zero-shot mode** — analyzes scenes without any examples
- **Guided mode** — uses example interpretations for more precise output

---

## Technologies Used

| Library | Purpose |
|---|---|
| Python 3.12 | Core language |
| SpaCy | Preprocessing and Named Entity Recognition |
| BERTopic | Unsupervised thematic analysis |
| Hugging Face Transformers | Emotion classification |
| PyTorch | Deep learning backend |
| PyMuPDF | PDF book text extraction |
| Scikit-learn | TF-IDF fallback theme extraction |
| Matplotlib | Visualization charts |
| NumPy | Numerical operations |
| Streamlit | Web application interface |
| Pandas | Summary tables |

---

## Models Used

| Model | Task |
|---|---|
| en_core_web_sm (SpaCy) | Tokenization, POS tagging, NER |
| j-hartmann/emotion-english-distilroberta-base | Emotion classification |
| BERTopic | Thematic clustering and keyword extraction |
| TfidfVectorizer (scikit-learn) | Fallback theme extraction |

---

## Repository Structure
