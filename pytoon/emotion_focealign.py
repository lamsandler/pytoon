from dataclasses import dataclass

from forcealign import ForceAlign
import re

from forcealign.forcealign import Word, phonemizer, Phoneme


def get_emotion_idx(transcript) -> list[tuple[str, int, int]]:
    """
    Find and replace emotion tags in transcript and return their positions with emotion types.

    Args:
        transcript (str): Input text with emotion tags like <anger>

    Returns:
        tuple: list of tuples with (emotion_name, emotion_start_idx, emotion_end_idx)
    """
    # Find all emotion tags and their positions
    pattern = r'<([^>]+)>'
    matches = list(re.finditer(pattern, transcript))

    emotions = []

    for match in matches:
        emotion = match.group(1)
        start_idx = match.start()
        end_idx = match.end()

        emotions.append((emotion, start_idx, end_idx))

    emotions.sort(key=lambda x: x[1])

    return emotions

@dataclass
class EmotionWord(Word):
    emotion: str = None

    def __repr__(self):
        return f"{self.word}: {self.time_start} -- {self.time_end}(s)"


class EmotionForceAlign(ForceAlign):
    emotion_idxs: list[tuple] = []
    original_transcript: str = None

    def __init__(self, audio_file: str, transcript: str = None):
        super().__init__(audio_file, transcript)
        self.original_transcript = self.raw_text

        self.emotion_idxs = get_emotion_idx(transcript=self.original_transcript)

    def inference(self) -> list[EmotionWord]:
        trellis = self.get_trellis()
        path = self.backtrack(trellis)
        segments = self.merge_repeats(path)
        word_segments = self.merge_words(segments)

        words = []
        current_emotion = None
        word_count = 0
        emotion_tag_found = False

        sorted_emotions = sorted(self.emotion_idxs, key=lambda x: x[1])
        next_emotion_idx = 0

        for word in word_segments:
            ratio = self.waveform.size(1) / trellis.size(0)
            start = int(ratio * word.start)
            end = int(ratio * word.end)
            time_start = round((start / self.bundle.sample_rate), 3)
            time_end = round((end / self.bundle.sample_rate), 3)

            word_label = word.label
            current_emotion = None

            if next_emotion_idx < len(sorted_emotions):
                emotion, start_idx, end_idx = sorted_emotions[next_emotion_idx]

                if word_count == 0 and start_idx == 0:
                    current_emotion = emotion
                    next_emotion_idx += 1
                elif word_count > 0:
                    words_before_tag = len(self.original_transcript[:start_idx].split())

                    if word_count == words_before_tag:
                        current_emotion = emotion
                        next_emotion_idx += 1

            phonemes = phonemizer(word_label)
            phoneme_duration = (time_end - time_start) / len(phonemes)

            start_phoneme = time_start
            for i, _ in enumerate(phonemes):
                end_phoneme = start_phoneme + phoneme_duration - 0.1
                phoneme = Phoneme(phoneme=phonemes[i], time_start=start_phoneme, time_end=end_phoneme)
                self.phoneme_alignments.append(phoneme)
                start_phoneme += phoneme_duration

            words.append(
                EmotionWord(
                    word=word_label,
                    phonemes=phonemes,
                    time_start=time_start,
                    time_end=time_end,
                    emotion=current_emotion,
                    breath=False,
                )
            )

            word_count += 1

        self.word_alignments = words
        return words
