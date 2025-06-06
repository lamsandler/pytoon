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


def alpha_with_punct_and_tags(text):
    return re.sub(r"[^a-zA-Z\s,.<>]", "", text)
