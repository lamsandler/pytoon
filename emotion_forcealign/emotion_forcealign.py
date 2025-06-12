import re
from dataclasses import dataclass
from typing import List

from .forcealign import ForceAlign, Word, phonemizer, Phoneme
from .utils import get_emotion_idx

@dataclass
class EmotionWord(Word):
    emotion: str | None

class EmotionForceAlign(ForceAlign):
    emotion_idx: List[tuple[int, str]]

    def __init__(self, audio_file: str, transcript: str = None):
        self.original_transcript = transcript
        if self.original_transcript is not None:
            transcript = re.sub(r"<[^>]*>", ".", self.original_transcript)
        super().__init__(audio_file, transcript)
        if self.original_transcript is not None:
            self.emotion_idx = get_emotion_idx(self.original_transcript)
        else:
            self.emotion_idx = []

    def inference(self):
        trellis = self.get_trellis()
        path = self.backtrack(trellis)
        segments = self.merge_repeats(path)
        word_segments = self.merge_words(segments)

        words = []
        idx = 0
        for word in word_segments:
            ratio = self.waveform.size(1) / trellis.size(0)
            start = int(ratio * word.start)
            end = int(ratio * word.end)
            time_start = round((start / self.bundle.sample_rate), 3)
            time_end = round((end / self.bundle.sample_rate), 3)

            phonemes = phonemizer(word.label)
            phoneme_duration = (time_end - time_start) / len(phonemes)

            start_phoneme = time_start
            for i, _ in enumerate(phonemes):
                end_phoneme = start_phoneme + phoneme_duration - 0.1
                phoneme = Phoneme(phoneme=phonemes[i], time_start=start_phoneme, time_end=end_phoneme)
                self.phoneme_alignments.append(phoneme)
                start_phoneme += phoneme_duration

            breath = idx in self.breath_idx

            emotion = None
            for emotion_idx in self.emotion_idx:
                if emotion_idx[0] == idx:
                    emotion = emotion_idx[1]
                    break

            words.append(
                EmotionWord(
                    word=word.label,
                    phonemes=phonemes,
                    time_start=time_start,
                    time_end=time_end,
                    breath=breath,
                    emotion=emotion,
                )
            )
            idx += 1

        self.word_alignments = words
        return words