"""Text to speech.

Primary: Kokoro-82M through kokoro-onnx (Apache-2.0, runs on a plain CPU, no API
key, no quota). The model files are downloaded once from the project's GitHub
release and cached by the workflow.

A deterministic fake voice is included so the whole pipeline can be tested
offline. It is clearly synthetic and is never used in production runs.
"""
from __future__ import annotations

import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICES_FILE = "voices-v1.0.bin"
SR = 24000


@dataclass
class Narration:
    audio: np.ndarray                      # mono float32 at SR
    sr: int
    sentence_times: list[tuple[float, float]]
    word_times: list[list[tuple[str, float, float]]] = field(default_factory=list)  # per sentence


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def _word_weights(words: list[str]) -> list[float]:
    out = []
    for w in words:
        wt = count_syllables(w) + 0.35 * len(re.sub(r"[^A-Za-z0-9]", "", w)) / 4
        if w.endswith((",", ";", ":")):
            wt += 0.5
        elif w.endswith((".", "!", "?")):
            wt += 0.8
        out.append(wt)
    return out


def split_words(text: str, t0: float, t1: float) -> list[tuple[str, float, float]]:
    words = text.split()
    wts = _word_weights(words)
    total = sum(wts) or 1.0
    out, cur = [], t0
    for w, wt in zip(words, wts):
        d = (t1 - t0) * wt / total
        out.append((w, cur, cur + d))
        cur += d
    return out


def _trim(x: np.ndarray, thresh: float = 0.012) -> np.ndarray:
    peak = float(np.max(np.abs(x))) or 1.0
    idx = np.where(np.abs(x) > thresh * peak * 4)[0]
    if len(idx) == 0:
        return x
    return x[max(0, idx[0] - 240): idx[-1] + 480]


def assemble(chunks: list[np.ndarray], sr: int, texts: list[str], lead: float = 0.10,
             gap: float = 0.24, gap_after_first: float = 0.36) -> Narration:
    parts = [np.zeros(int(lead * sr), np.float32)]
    times, words = [], []
    cur = lead
    for i, (c, text) in enumerate(zip(chunks, texts)):
        c = _trim(c.astype(np.float32))
        dur = len(c) / sr
        times.append((cur, cur + dur))
        words.append(split_words(text, cur, cur + dur))
        parts.append(c)
        cur += dur
        g = gap_after_first if i == 0 else gap
        parts.append(np.zeros(int(g * sr), np.float32))
        cur += g
    parts.append(np.zeros(int(0.5 * sr), np.float32))
    return Narration(np.concatenate(parts), sr, times, words)


def download_models(model_dir: str) -> tuple[str, str]:
    os.makedirs(model_dir, exist_ok=True)
    paths = []
    for name in (MODEL_FILE, VOICES_FILE):
        p = os.path.join(model_dir, name)
        if not os.path.exists(p) or os.path.getsize(p) < 1_000_000:
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(KOKORO_BASE + name, p)
        paths.append(p)
    return paths[0], paths[1]


BUNDLE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
BUNDLE_DIR = "kokoro-multi-lang-v1_0"
SPEAKERS = ("af_alloy,af_aoede,af_bella,af_heart,af_jessica,af_kore,af_nicole,af_nova,af_river,af_sarah,af_sky,"
            "am_adam,am_echo,am_eric,am_fenrir,am_liam,am_michael,am_onyx,am_puck,am_santa,bf_alice,bf_emma,"
            "bf_isabella,bf_lily,bm_daniel,bm_fable,bm_george,bm_lewis").split(",")
LETTER_WORDS = dict(a="a", b="bee", c="see", d="dee", e="e", f="eff", g="gee", h="aitch", i="eye", j="jay",
                    k="kay", l="el", m="em", n="en", o="oh", p="pea", q="cue", r="are", s="ess", t="tea",
                    u="you", v="vee", w="double", x="ex", y="why", z="zed")


class BundleKokoro:
    """Kokoro v1.0 straight through onnxruntime, using the model bundle published by
    k2-fsa. English words are phonemised with the bundle's lexicon, so no espeak is
    needed. Used when kokoro-onnx is unavailable; verified to produce clear speech.
    """

    def __init__(self, model_dir: str):
        import onnxruntime as ort
        base = self._ensure(model_dir)
        self.tok = {}
        for line in open(os.path.join(base, "tokens.txt"), encoding="utf-8"):
            line = line.rstrip("\n")
            if line:
                sym, _, i = line.rpartition(" ")
                self.tok[sym or " "] = int(i)
        self.lex = {}
        for line in open(os.path.join(base, "lexicon-us-en.txt"), encoding="utf-8"):
            p = line.rstrip("\n").split(" ")
            if len(p) > 1:
                self.lex.setdefault(p[0], p[1:])
        self.lex["shoob"] = ["ʃ", "ˈ", "u", "ː", "b", "ʰ", "ʰ"]   # Shubh, Indian style: soft sh, long oo, gentle aspirated bh
        self.voices = np.fromfile(os.path.join(base, "voices.bin"), dtype=np.float32).reshape(54, 510, 256)
        self.sess = ort.InferenceSession(os.path.join(base, "model.onnx"), providers=["CPUExecutionProvider"])

    @staticmethod
    def _ensure(model_dir: str) -> str:
        import tarfile
        base = os.path.join(model_dir, BUNDLE_DIR)
        if os.path.exists(os.path.join(base, "model.onnx")):
            return base
        os.makedirs(model_dir, exist_ok=True)
        arc = os.path.join(model_dir, BUNDLE_DIR + ".tar.bz2")
        print("downloading Kokoro bundle (about 350 MB, cached after the first run) ...", flush=True)
        urllib.request.urlretrieve(BUNDLE_URL, arc)
        with tarfile.open(arc, "r:bz2") as tf:
            tf.extractall(model_dir, filter="data")
        os.remove(arc)
        return base

    def _word_phones(self, w: str) -> list[str]:
        lw = w.lower()
        if lw == "shoob" and "shoob" not in self.lex:      # the name Shubh, spoken as one word
            return self.lex["shoe"] + ["b"]
        if lw in self.lex:
            return self.lex[lw]
        for suf, add in (("'s", "z"), ("s", "z"), ("ed", "d"), ("ing", None)):
            if lw.endswith(suf) and lw[: -len(suf)] in self.lex and len(lw) > len(suf) + 2:
                base = self.lex[lw[: -len(suf)]]
                return base + ([add] if add else ["ɪ", "ŋ"])
        if "-" in lw:
            return [p for part in lw.split("-") for p in self.lex.get(part, [])]
        out: list[str] = []                      # unknown word or acronym: spell it
        for ch in lw:
            out += self.lex.get(LETTER_WORDS.get(ch, ch), [])
            out.append(" ")
        return out

    def phonemize(self, text: str) -> list[int]:
        ids: list[int] = []
        sp = self.tok.get(" ")
        for m in re.finditer(r"[A-Za-z][A-Za-z'\-]*|[,.!?;:]", text):
            w = m.group(0)
            if not w[0].isalpha():
                if w in self.tok:
                    ids.append(self.tok[w])
                continue
            if ids and sp is not None and ids[-1] != sp:
                ids.append(sp)
            for p in self._word_phones(w):
                if p == " ":
                    if sp is not None and ids and ids[-1] != sp:
                        ids.append(sp)
                elif p in self.tok:
                    ids.append(self.tok[p])
        return ids[:500]

    def create(self, text: str, voice: str, speed: float) -> np.ndarray:
        from . import brand
        ids = self.phonemize(brand.say(text))
        tokens = np.array([[0] + ids + [0]], dtype=np.int64)
        style = self.voices[SPEAKERS.index(voice)][min(len(ids), 509)][None, :]
        out = self.sess.run(None, {"tokens": tokens, "style": style.astype(np.float32),
                                   "speed": np.array([speed], np.float32)})[0]
        return out.astype(np.float32)


def synth_bundle(texts: list[str], voice: str, speed: float, model_dir: str) -> Narration:
    k = BundleKokoro(model_dir)
    return assemble([k.create(t, voice, speed) for t in texts], SR, texts)


def synth_voice(texts: list[str], voice: str, speed: float, model_dir: str) -> Narration:
    """Preferred: kokoro-onnx (espeak phonemes). Fallback: the verified bundle path."""
    try:
        return synth_kokoro(texts, voice, speed, model_dir)
    except Exception as e:  # noqa: BLE001 - any failure here falls back, loudly
        print(f"kokoro-onnx path failed ({type(e).__name__}: {str(e)[:120]}); using the bundle path", flush=True)
        return synth_bundle(texts, voice, speed, model_dir)


def synth_kokoro(texts: list[str], voice: str, speed: float, model_dir: str) -> Narration:
    from kokoro_onnx import Kokoro  # imported lazily: heavy and only needed here
    model, voices = download_models(model_dir)
    k = Kokoro(model, voices)
    chunks = []
    for t in texts:
        samples, sr = k.create(__import__("reelbot.brand", fromlist=["say"]).say(t), voice=voice, speed=speed, lang="en-us")
        if sr != SR:
            raise RuntimeError(f"unexpected sample rate {sr}")
        chunks.append(np.asarray(samples, np.float32))
    return assemble(chunks, SR, texts)


# --------------------------------------------------------------------------
# offline stand-in voice (tests only)
# --------------------------------------------------------------------------
def _fake_sentence(text: str, rng: np.random.Generator) -> np.ndarray:
    out = []
    for w in text.split():
        syl = count_syllables(w)
        for s in range(syl):
            d = rng.uniform(0.13, 0.2)
            n = int(d * SR)
            tt = np.arange(n) / SR
            f0 = 105 * (1 + 0.1 * np.sin(2 * np.pi * 3 * tt + rng.uniform(0, 6)))
            ph = 2 * np.pi * np.cumsum(f0) / SR
            sig = sum(np.sin(h * ph) / h ** 0.9 for h in range(1, 14))
            env = np.sin(np.pi * np.arange(n) / n) ** 1.4 * rng.uniform(0.6, 1.0)
            out.append((sig * env * 0.1).astype(np.float32))
        pause = 0.2 if w.endswith((",", ";", ":")) else 0.3 if w.endswith((".", "!", "?")) else 0.04
        out.append(np.zeros(int(pause * SR), np.float32))
    return np.concatenate(out) if out else np.zeros(SR // 4, np.float32)


def synth_fake(texts: list[str], seed: int = 5) -> Narration:
    rng = np.random.default_rng(seed)
    return assemble([_fake_sentence(t, rng) for t in texts], SR, texts)
