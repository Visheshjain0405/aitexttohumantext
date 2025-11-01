# api.py
import re
import ssl
import random
import warnings
from typing import List, Optional

import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional: sentence-transformers to pick natural synonyms (safe fallback without it)
try:
    from sentence_transformers import SentenceTransformer, util
    ST_OK = True
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    ST_OK = False

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# NLTK bootstrap (quiet + cached)
# -----------------------------
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

bootstrap_nltk()
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# -----------------------------
# Lightweight sentence splitter
# -----------------------------
def simple_split_sentences(text: str) -> List[str]:
    # If spaCy is available with the en model, use it; else a regex fallback
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    except Exception:
        # Regex fallback
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

# -----------------------------
# Humanizer core
# -----------------------------
class NewsHumanizer:
    """
    Convert AI-ish / marketing-y text into neutral, readable news style:
      - Tone down hype/emoji/excess punctuation
      - Expand contractions (it's -> it is)
      - Light synonyms (optionally guided by MiniLM)
      - Optional neutral transitions for flow
    """

    def __init__(
        self,
        use_embeddings: bool = True,
        p_synonym: float = 0.25,
        p_transition: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.use_embeddings = use_embeddings and ST_OK
        self.p_synonym = p_synonym
        self.p_transition = p_transition
        if seed is not None:
            random.seed(seed)

        self.transitions = ["Meanwhile,", "In addition,", "Separately,", "Earlier,", "Later,"]
        self.hype_map = {
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
        self.expand_map = {"n't":" not","'re":" are","'s":" is","'ll":" will","'ve":" have","'d":" would","'m":" am"}

        # Lazy-load model on first use (saves memory on small dynos)
        self._sbert = None

    def _load_sbert(self):
        if not self.use_embeddings or self._sbert is not None:
            return
        try:
            self._sbert = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
        except Exception:
            self._sbert = None
            self.use_embeddings = False  # fallback

    def _closest_synonym(self, word: str, candidates: List[str]) -> Optional[str]:
        if not candidates:
            return None
        if not self.use_embeddings or util is None:
            return candidates[0]
        self._load_sbert()
        if self._sbert is None:
            return candidates[0]
        w = self._sbert.encode(word, convert_to_tensor=True)
        c = self._sbert.encode(candidates, convert_to_tensor=True)
        scores = util.cos_sim(w, c)[0]
        idx = scores.argmax().item()
        return candidates[idx]

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _expand_contractions(self, s: str) -> str:
        tokens = word_tokenize(s)
        out = []
        for tok in tokens:
            lw = tok.lower()
            replaced = False
            for c, e in self.expand_map.items():
                if lw.endswith(c):
                    new_tok = lw.replace(c, e)
                    if tok and tok[0].isupper():
                        new_tok = new_tok.capitalize()
                    out.append(new_tok)
                    replaced = True
                    break
            if not replaced:
                out.append(tok)
        return " ".join(out)

    def _tone_down(self, s: str) -> str:
        # hype words, emojis, extra punctuation
        for pat, repl in self.hype_map.items():
            s = re.sub(pat, repl, s, flags=re.IGNORECASE)
        s = re.sub(r"[🔥✨💥🚀🎉🤯👏]+", "", s)
        s = re.sub(r"(!){2,}", "!", s)
        s = re.sub(r"(\?){2,}", "?", s)
        return self._normalize_spaces(s)

    def _simple_synonyms(self, w: str, pos: str) -> List[str]:
        wn_pos = {"J": wordnet.ADJ, "N": wordnet.NOUN, "R": wordnet.ADV, "V": wordnet.VERB}.get(pos[:1])
        syns = set()
        for syn in wordnet.synsets(w, pos=wn_pos):
            for l in syn.lemmas():
                name = l.name().replace("_", " ")
                if len(name) <= 12 and name.isalpha() and name.lower() != w.lower():
                    syns.add(name)
        return list(syns)

    def _simplify_with_synonyms(self, s: str) -> str:
        tokens = word_tokenize(s)
        pos_tags = nltk.pos_tag(tokens)
        out = []
        for w, p in pos_tags:
            if p[:1] in ("J", "N", "V", "R") and wordnet.synsets(w) and random.random() < self.p_synonym:
                cands = self._simple_synonyms(w, p)
                if cands:
                    out.append(self._closest_synonym(w, cands) or w)
                    continue
            out.append(w)
        return " ".join(out)

    def _clean_punctuation_spacing(self, s: str) -> str:
        s = re.sub(r"\s+([.,;:!?])", r"\1", s)
        s = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", s)
        return s

    def humanize(self, text: str) -> str:
        sentences = simple_split_sentences(text)
        out = []
        for s in sentences:
            t = self._tone_down(s)
            t = self._expand_contractions(t)
            if random.random() < self.p_transition:
                t = f"{random.choice(self.transitions)} {t}"
            t = self._simplify_with_synonyms(t)
            t = self._clean_punctuation_spacing(t)
            out.append(t)
        return self._normalize_spaces(" ".join(out))


# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="News Humanizer API", version="1.0.0")

# Open CORS for dev; tighten origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace with your domain(s) in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HUMANIZER = NewsHumanizer(use_embeddings=True)

# -----------------------------
# Schemas
# -----------------------------
class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    p_synonym: Optional[float] = Field(None, ge=0.0, le=1.0, description="0..1 probability of synonym substitution")
    p_transition: Optional[float] = Field(None, ge=0.0, le=1.0, description="0..1 probability to insert a neutral transition")
    seed: Optional[int] = Field(None, description="Random seed for deterministic output")

class HumanizeResponse(BaseModel):
    humanized_text: str

class HumanizeBatchRequest(BaseModel):
    items: List[HumanizeRequest]

class HumanizeBatchResponse(BaseModel):
    results: List[HumanizeResponse]

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"name": "News Humanizer API", "version": "1.0.0"}

@app.post("/v1/humanize", response_model=HumanizeResponse)
def humanize_endpoint(req: HumanizeRequest):
    # allow per-request tuning
    if req.p_synonym is not None:
        HUMANIZER.p_synonym = req.p_synonym
    if req.p_transition is not None:
        HUMANIZER.p_transition = req.p_transition
    if req.seed is not None:
        random.seed(req.seed)

    output = HUMANIZER.humanize(req.text)
    return {"humanized_text": output}

@app.post("/v1/humanize-batch", response_model=HumanizeBatchResponse)
def humanize_batch_endpoint(req: HumanizeBatchRequest):
    results = []
    for item in req.items:
        if item.p_synonym is not None:
            HUMANIZER.p_synonym = item.p_synonym
        if item.p_transition is not None:
            HUMANIZER.p_transition = item.p_transition
        if item.seed is not None:
            random.seed(item.seed)
        results.append(HumanizeResponse(humanized_text=HUMANIZER.humanize(item.text)))
    return {"results": results}
