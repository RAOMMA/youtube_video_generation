"""Runs inside tts_venv: Whisper listens to a spoken line and checks that it says what the script says.
Used by qwen_voices.py (retake glitchy lines) and voice_qa_asr.py (audit an episode)."""
import difflib
import re

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
CONTRACTIONS = {"here's": "here is", "there's": "there is", "it's": "it is", "that's": "that is",
                "what's": "what is", "let's": "let us", "we've": "we have", "we'll": "we will", "we're": "we are",
                "you're": "you are", "you'll": "you will", "i'll": "i will", "i'm": "i am", "don't": "do not",
                "doesn't": "does not", "didn't": "did not", "can't": "can not", "won't": "will not",
                "isn't": "is not", "aren't": "are not", "they're": "they are", "he's": "he is", "she's": "she is"}
_asr = {}


def _int_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:  # 120 -> "one hundred and twenty"
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + _int_words(n % 100))
    if n < 10000 and n % 1000 == 0:
        return ONES[n // 1000] + " thousand"
    return " ".join(ONES[int(d)] for d in str(n))


def _numbers(m) -> str:
    whole, _, frac = m.group(0).partition(".")
    out = _int_words(int(whole)) if whole else "zero"
    return out + (" point " + " ".join(ONES[int(d)] for d in frac) if frac else "")


HOMOPHONES = {"claude": "cloud", "claudes": "clouds"}  # Whisper writes these as the word that sounds the same


def words(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("-", " ").replace("%", " percent")
    t = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", " dot ", t)  # "claude.ai" / "CLAUDE.md" are read aloud as "dot"
    t = re.sub(r"(?<![a-z])(claude|claudes)(?![a-z])", lambda m: HOMOPHONES[m.group(1)], t)
    t = re.sub(r"ml\s*wal+[iy]\s*sarkar", "mlwalisarkar", t)  # the channel name, however Whisper spells it
    t = re.sub(r"(?<=[a-z])is(e|es|ed|ing)(?![a-z])", r"iz\g<1>", t)  # British / American spelling
    t = re.sub(r"[$]([0-9]+)", r"\g<1> dollar", t)  # "$5" is spoken "five dollar(s)"
    t = re.sub(r"(?<=dollar)s(?![a-z])", "", t)
    t = re.sub(r"[0-9]*[.]?[0-9]+", _numbers, t)
    t = " ".join(CONTRACTIONS.get(w, w) for w in t.split())
    # stretched sounds count the same: "wheee" = "whee", "yaaay" = "yay"
    return [re.sub(r"(.)\1{2,}", r"\1\1", w) for w in re.findall(r"[a-z']+", t)]


def score(script_text: str, heard: str) -> tuple[float, int]:
    """(similarity 0..1, number of script words that were not heard correctly)"""
    a, b = words(script_text), words(heard)
    sm = difflib.SequenceMatcher(None, a, b)
    wrong = len(a) - sum(blk.size for blk in sm.get_matching_blocks())
    return sm.ratio(), wrong


def acceptable(script_text: str, heard: str) -> bool:
    a, b = "".join(words(script_text)), "".join(words(heard))
    if a == b:  # only spacing differs: "icecream" / "ice cream"
        return True
    # letter-level match for lines of 6+ words: spacing/spelling variants ("a i" / "ai", "md" / "m d") aren't glitches,
    # while real slurs ("prices over time" -> "grace is over time") still fall below 0.95
    if len(words(script_text)) >= 6 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.95:
        return True
    ratio, wrong = score(script_text, heard)
    n = len(words(script_text))
    return wrong == 0 or ratio >= 0.9 or (wrong <= 1 and n >= 6) or (wrong <= 2 and n >= 12 and ratio >= 0.85)


def transcribe(audio, sr, model: str = "openai/whisper-small.en") -> str:
    if model not in _asr:
        from transformers import pipeline
        _asr[model] = pipeline("automatic-speech-recognition", model=model, device=-1)  # CPU: the GPU is rendering
    return _asr[model]({"raw": audio, "sampling_rate": sr}, chunk_length_s=30)["text"].strip()


def check(script_text: str, audio, sr) -> tuple[bool, str]:
    """Is the take right? whisper-small listens first; when it objects, the more accurate whisper-medium gets a
    second listen, so Whisper's own mishearings ("court marks" for "quote marks") don't count as glitches."""
    heard = transcribe(audio, sr)
    if acceptable(script_text, heard):
        return True, heard
    heard2 = transcribe(audio, sr, "openai/whisper-medium.en")
    return acceptable(script_text, heard2), heard2
