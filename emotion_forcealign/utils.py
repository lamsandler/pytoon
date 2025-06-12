import random
import re
from typing import List


def get_emotion_idx(transcript) -> List[tuple[int, str]]:
    """Detect where emotion tags are placed in transcript."""
    transcript = transcript.replace("—", "")
    transcript = alpha_with_punct_and_tags(transcript).upper().split()
    idxs = []
    offset = 0
    for i in range(len(transcript) - 1):
        current_word = transcript[i]
        if current_word.startswith("<") and current_word.endswith(">"):
            idxs.append((i - offset, current_word))
            offset += 1
    return idxs


def strip_tag(emotion_tag: str):
    if not emotion_tag.startswith("<") or not emotion_tag.endswith(">"):
        return emotion_tag
    return emotion_tag[1:-1].lower()


def get_breath_idx(transcript):
    """Detect where breaths might occur."""
    transcript = transcript.replace("—", " ")
    transcript = alpha_with_punct(transcript).upper().split()
    idxs = []
    for i in range(len(transcript) - 1):
        if "," in transcript[i]:
            idxs.append(i + 1)
        elif "." in transcript[i] and random.choice([True, False, False]):
            idxs.append(i + 1)
    return idxs

def alpha_with_punct(text):
    return re.sub(r"[^a-zA-Z\s,.]", "", text)

def alpha_with_punct_and_tags(text):
    return re.sub(r"[^a-zA-Z\s,.<>]", "", text)

def alphabetical(text):
    return re.sub(r"[^a-zA-Z\s]", "", text)
