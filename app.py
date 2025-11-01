import re, ssl, random, warnings
import streamlit as st
import nltk, spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Optional MiniLM model
try:
    from sentence_transformers import SentenceTransformer, util
    ST_OK = True
except Exception:
    SentenceTransformer = None
    util = None
    ST_OK = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Setup ----------
def bootstrap_nltk():
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
    for r in ["punkt", "averaged_perceptron_tagger", "wordnet"]:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

@st.cache_resource
def load_spacy():
    import spacy as _spacy
    try:
        return _spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return _spacy.load("en_core_web_sm")

@st.cache_resource
def load_model():
    if not ST_OK:
        return None
    try:
        return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    except Exception:
        return None

bootstrap_nltk()
NLP = load_spacy()
SBERT = load_model()

def closest_synonym(word, candidates):
    if not candidates:
        return None
    if SBERT is None or util is None:
        return candidates[0]
    w_emb = SBERT.encode(word, convert_to_tensor=True)
    c_embs = SBERT.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(w_emb, c_embs)[0]
    idx = scores.argmax().item()
    return candidates[idx]

# ---------- Humanizer ----------
class NewsHumanizer:
    def __init__(self):
        self.p_transition = 0.15
        self.p_synonym = 0.25
        self.transitions = ["Meanwhile,", "In addition,", "Separately,", "Earlier,", "Later,"]
        self.hype = {
            r"\bgroundbreaking\b": "notable",
            r"\bgame[- ]?changer\b": "significant",
            r"\bspectacular\b": "major",
            r"\bexplosive\b": "strong",
            r"\binsane\b": "very",
            r"\bviral\b": "widely shared",
            r"\bunprecedented\b": "unusual",
            r"\bmassive\b": "large",
            r"\bthrilling\b": "energetic",
        }
        self.expand = {"n't": " not", "'re": " are", "'s": " is", "'ll": " will",
                       "'ve": " have", "'d": " would", "'m": " am"}

    def _expand_contractions(self, text):
        tokens = word_tokenize(text)
        out = []
        for tok in tokens:
            lw = tok.lower()
            done = False
            for c, e in self.expand.items():
                if lw.endswith(c):
                    new = lw.replace(c, e)
                    if tok[0].isupper():
                        new = new.capitalize()
                    out.append(new)
                    done = True
                    break
            if not done:
                out.append(tok)
        return " ".join(out)

    def _tone_down(self, s):
        for pat, repl in self.hype.items():
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)
        s = re.sub(r"[🔥✨💥🚀🎉🤯👏]+", "", s)
        s = re.sub(r"(!){2,}", "!", s)
        s = re.sub(r"(\?){2,}", "?", s)
        return re.sub(r"\s+", " ", s).strip()

    def _simplify(self, s):
        toks = word_tokenize(s)
        pos = nltk.pos_tag(toks)
        out = []
        for w, p in pos:
            if p[:1] in ("J","N","V","R") and wordnet.synsets(w):
                if random.random() < self.p_synonym:
                    cands = self._simple_synonyms(w, p)
                    if cands:
                        out.append(closest_synonym(w, cands) or w)
                        continue
            out.append(w)
        return " ".join(out)

    def _simple_synonyms(self, w, pos):
        wn_pos = {"J":wordnet.ADJ, "N":wordnet.NOUN, "R":wordnet.ADV, "V":wordnet.VERB}.get(pos[0])
        syns=set()
        for syn in wordnet.synsets(w, pos=wn_pos):
            for l in syn.lemmas():
                name = l.name().replace("_"," ")
                if len(name)<=12 and name.isalpha() and name.lower()!=w.lower():
                    syns.add(name)
        return list(syns)

    def humanize(self, text):
        doc = NLP(text)
        out = []
        for s in doc.sents:
            t = s.text.strip()
            t = self._tone_down(t)
            t = self._expand_contractions(t)
            if random.random() < self.p_transition:
                t = f"{random.choice(self.transitions)} {t}"
            t = self._simplify(t)
            t = re.sub(r"\s+([.,;:!?])", r"\1", t)
            t = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", t)
            out.append(t)
        return re.sub(r"\s+", " ", " ".join(out)).strip()

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI → Human News Converter", page_icon="📰", layout="wide")
st.title("📰 AI → Human News Article Converter")

st.markdown(
    """
Paste any AI-generated or marketing-style article below.  
This tool rewrites it into a **neutral, professional news format** with natural language flow.
""")

text = st.text_area("✍️ Paste your AI text here", height=380, placeholder="Paste article text...")

if st.button("✨ Convert to Human News Style"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Converting... please wait..."):
            humanizer = NewsHumanizer()
            result = humanizer.humanize(text)
        st.success("✅ Done!")

        st.markdown("### 🧾 Original (AI Text)")
        st.write(text)

        st.markdown("### 📰 Humanized News Article")
        st.write(result)

        st.download_button("⬇️ Download Output", result.encode("utf-8"), "news_output.txt", "text/plain")
