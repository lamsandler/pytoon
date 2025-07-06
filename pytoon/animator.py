import copy
import os
import random
import shutil
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from moviepy import ImageSequenceClip, CompositeVideoClip, CompositeAudioClip, AudioFileClip, VideoClip
from tqdm import tqdm

from emotion_forcealign.utils import strip_tag
from .dataloader import get_assets, Pose
from .lipsync import viseme_sequencer

# Set up logger
logger = logging.getLogger(__name__)


class FrameSequence:
    emotion_changes: List[str | None]

    def __init__(self):
        self.pose_files = []
        self.mouth_files = []
        self.pose_images = []
        self.mouth_images = []
        self.mouth_coords = []
        self.emotion_changes = []


def set_up_frames_directory(frames_dir: Path):
    logger.debug(f"Setting up frames directory: {frames_dir}")
    if frames_dir.exists():
        logger.debug(f"Removing existing directory: {frames_dir}")
        shutil.rmtree(frames_dir)
    logger.debug(f"Creating directory: {frames_dir}")
    frames_dir.mkdir(exist_ok=True, parents=True)
    logger.debug(f"Frames directory setup complete: {frames_dir}")


def save_frame(frame, frame_index: int, frames_dir: Path):
    output_path = frames_dir / f"frame_{frame_index:06d}.png"
    try:
        frame.save(output_path)
        logger.debug(f"Saved frame {frame_index} to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving frame {frame_index}: {e}")
        return None

class animate:
    """Animates a cartoon that is lip synced to provieded audio voiceover."""

    def __init__(self, audio_file: str, transcript: str = None, fps: int = 48, frames_dir: Path = Path("./temp")):
        logger.info(f"Initializing animation with audio file: {audio_file}")
        self.audio_file = audio_file
        self.sequence = FrameSequence()
        self.assets = get_assets()
        self.fps = fps
        self.frames_dir = frames_dir

        # Initialize blinking rate (blink every 3 seconds)
        self.blink_rate = 3.0
        logger.debug(f"Blink rate set to {self.blink_rate} seconds")

        # Create sequence of mouth images
        logger.info("Creating viseme sequence from audio")
        self.viseme_sequence = viseme_sequencer(self.audio_file, transcript, self.fps)
        self.build_mouth_sequence()

        self.duration = len(self.sequence.mouth_files) / self.fps
        logger.info(f"Created {len(self.sequence.mouth_files)} mouth frames")
        logger.info(f"Animation duration: {self.duration:.2f} seconds")

        logger.info("Building pose sequence")
        self.build_pose_sequence()

        self.frame_size = self.get_frame_size()
        logger.debug(f"Frame size: {self.frame_size}")

        # Create the animation
        logger.info("Compiling animation")
        self.compile_animation()
        logger.info("Animation initialization complete")

    def build_pose_sequence(self):
        """Creates the sequence of pose images for the video"""
        logger.debug("Starting build_pose_sequence")
        emotion = self.get_random_emotion()
        pose = random.choice(emotion)
        logger.debug(f"Initial emotion selected: {emotion}")

        # Add a character pose frame for every frame of a mouth
        for i, _ in tqdm(enumerate(self.sequence.mouth_files), total=len(self.sequence.mouth_files), desc="Building pose sequence"):

            if self.sequence.emotion_changes[i] is not None:
                # Change the emotion of the character
                emotion_name = strip_tag(self.sequence.emotion_changes[i])
                emotion = self.get_emotion(emotion_name)
                pose = random.choice(emotion)
                logger.debug(f"Frame {i}: Emotion changed to {emotion_name}")

            eyes = self.blink_manager(idx=i)
            self.sequence.pose_files.append(pose.image_files[eyes])
            self.sequence.mouth_coords.append(pose.mouth_coordinates)
            logger.debug(f"Frame {i}: Eye state: {eyes}")

        # Prepend absolute path to all pose images
        logger.debug("Adding absolute paths to pose files")
        self.sequence.pose_files = [f"{os.path.dirname(__file__)}{file}" for file in self.sequence.pose_files]

        # Create mouth PIL image for every frame, with image transformations based on pose
        logger.debug("Starting mouth image transformations")
        for i, _ in tqdm(enumerate(self.sequence.mouth_files), total=len(self.sequence.mouth_files), desc="Transforming mouth images"):
            transformed_image = mouth_transformation(
                mouth_file=self.sequence.mouth_files[i],
                mouth_coord=self.sequence.mouth_coords[i],
            )
            self.sequence.mouth_images.append(transformed_image)

        logger.debug(f"Completed build_pose_sequence with {len(self.sequence.pose_files)} pose frames")
        return

    def blink_manager(self, idx):

        BLINK_DURATION = 0.16
        SUB_BLINKS = ["middle", "shut", "middle"]

        frames_between_blinks = int(self.blink_rate * self.fps)
        frames_per_blink = int(BLINK_DURATION * self.fps)
        frames_per_sub_blink = int(frames_per_blink / len(SUB_BLINKS)) + 1

        full_cycle = frames_between_blinks + (frames_per_sub_blink * len(SUB_BLINKS))

        start_1 = frames_between_blinks
        start_2 = start_1 + frames_per_sub_blink
        start_3 = start_2 + frames_per_sub_blink
        end_3 = start_3 + frames_per_sub_blink

        if start_1 <= (idx % full_cycle) < start_2:
            eyes = "middle"

        elif start_2 <= (idx % full_cycle) < start_3:
            eyes = "shut"

        elif start_3 <= (idx % full_cycle) < end_3:
            eyes = "middle"

        else:
            eyes = "open"

        return eyes

    def build_mouth_sequence(self):
        """Generates a sequence of mouth images for video"""
        logger.debug("Starting build_mouth_sequence")

        # Add mouth images to mouth image file sequence
        viseme_count = 0
        emotion_change_count = 0

        for i, _ in tqdm(enumerate(self.viseme_sequence), total=len(self.viseme_sequence), desc="Processing viseme sequence"):
            if self.viseme_sequence[i].visemes:
                viseme_count += len(self.viseme_sequence[i].visemes)
                self.sequence.mouth_files.extend(self.viseme_sequence[i].visemes)
                emotion_changes = [None] * len(self.viseme_sequence[i].visemes)

                if self.viseme_sequence[i].emotion is not None:
                    emotion_changes[0] = self.viseme_sequence[i].emotion
                    emotion_change_count += 1
                    logger.debug(f"Emotion change at viseme {i}: {self.viseme_sequence[i].emotion}")
                    self.sequence.emotion_changes.extend(emotion_changes)
                else:
                    self.sequence.emotion_changes.extend(emotion_changes)

        logger.debug(f"Processed {len(self.viseme_sequence)} viseme sequences with {viseme_count} total visemes")
        logger.debug(f"Found {emotion_change_count} emotion changes")

        # Prepend absolute path to mouth images
        logger.debug("Adding absolute paths to mouth files")
        for i, _ in tqdm(enumerate(self.sequence.mouth_files), total=len(self.sequence.mouth_files), desc="Preparing mouth files"):
            file = self.sequence.mouth_files[i]
            new_file = f"{os.path.dirname(__file__)}/assets/visemes/positive/{file}"
            self.sequence.mouth_files[i] = new_file

        logger.debug(f"Completed build_mouth_sequence with {len(self.sequence.mouth_files)} mouth files")

    def get_random_emotion(self) -> list[Pose] | None:
        """Generates a random emotion to use in sequence"""
        emotions_list = list(self.assets.__dict__.keys())
        emotion = random.choice(emotions_list)
        return getattr(self.assets, emotion)

    def get_emotion(self, emotion: str = None) -> list[Pose] | None:
        """Generates a random emotion to use in sequence

        Returns:
            list[Pose]: List of poses from a random emotion
        """
        for emotion_poses in list(self.assets.__dict__.keys()):
            if emotion_poses == emotion:
                return getattr(self.assets, emotion_poses)
        return None

    def get_frame_size(self):
        pose_image = cv2.imread(self.sequence.pose_files[0])
        height, width, _ = pose_image.shape
        return width, height

    def compile_animation(self):
        logger.info(f"Starting animation compilation to directory: {self.frames_dir}")
        set_up_frames_directory(self.frames_dir)
        logger.debug("Frames directory set up")

        final_frame: Image
        saved_frames = 0

        for i, _ in tqdm(enumerate(self.sequence.pose_files), total=len(self.sequence.pose_files), desc="Saving frames"):
            frame = cv2.imread(self.sequence.pose_files[i], cv2.IMREAD_UNCHANGED)
            if frame is None:
                logger.warning(f"Failed to read pose image at index {i}: {self.sequence.pose_files[i]}")
                continue

            if self.sequence.mouth_files[i] is not None:
                logger.debug(f"Rendering frame {i} with mouth overlay")
                final_frame = render_frame(
                    pose_img=frame,
                    mouth_img=self.sequence.mouth_images[i],
                    mouth_coord=self.sequence.mouth_coords[i],
                )
            else:
                logger.debug(f"Rendering frame {i} without mouth overlay")
                final_frame = frame

            result = save_frame(final_frame, i, self.frames_dir)
            if result:
                saved_frames += 1

        logger.info(f"Animation compilation complete. Saved {saved_frames} frames to {self.frames_dir}")

    def export(self, path: str, background: VideoClip, scale: float = 0.7):
        logger.info(f"Starting export to {path}")
        logger.debug(f"Export parameters: scale={scale}, fps={self.fps}")

        try:
            logger.debug("Creating ImageSequenceClip from frames")
            animation_clip = ImageSequenceClip("./temp", fps=self.fps, with_mask=True)
            logger.debug(f"Created animation clip with dimensions: {animation_clip.size}")

            new_height = int(background.size[1] * scale)
            new_width = int(animation_clip.w * (new_height / animation_clip.h))
            logger.debug(f"Resizing animation to width={new_width}, height={new_height}")
            animation_clip = animation_clip.resized(width=new_width, height=new_height)

            # Overlay the animation on top of the background clip
            logger.debug("Creating composite video clip")
            final_clip = CompositeVideoClip(
                clips=[background, animation_clip.set_position(("right", "bottom"))], use_bgclip=True
            )

            # Add speech audio to clip with 0.2 second delay
            logger.debug(f"Adding audio from {self.audio_file}")
            audio_clip = AudioFileClip(self.audio_file)
            audio_clip = CompositeAudioClip([audio_clip.with_start(0.2)])
            final_clip = final_clip.with_audio(audio_clip)

            # Export video to .mp4
            logger.info(f"Writing video file to {path}")
            final_clip.write_videofile(
                path, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4, fps=self.fps
            )
            logger.info(f"Video export complete: {path}")

            logger.debug(f"Cleaning up temporary directory: {self.frames_dir}")
            shutil.rmtree(self.frames_dir)

            return path
        except Exception as e:
            logger.error(f"Error during video export: {e}")
            raise


def mouth_transformation(mouth_file, mouth_coord) -> Image:
    """Transforms mouth image with scaling, flipping, and rotation.
        This transformation is applied because, the same mouth shape images
        are used for different pose images, but the size, angle, and position
        of a mouth image will depend on which pose image is being used.

    Returns:
        Image: PIL Image object of mouth image with applied transformations
    """
    logger.debug(f"Transforming mouth image: {mouth_file}")
    try:
        mouth = copy.deepcopy(Image.open(mouth_file))

        # Flip mouth horizontally if necessary
        if mouth_coord.flip_x is True:
            logger.debug("Flipping mouth horizontally")
            mouth = mouth.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Scale mouth image if necessary
        if mouth_coord.scale_y != 1:
            og_width, og_height = mouth.size
            new_width = int(abs(og_width * mouth_coord.scale_x))
            new_height = int(og_height * mouth_coord.scale_y)
            logger.debug(f"Scaling mouth from {og_width}x{og_height} to {new_width}x{new_height}")
            try:
                mouth = mouth.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f"Failed to resize mouth image: {e}")

        # Apply image rotation if necessary
        if mouth_coord.rotation != 0:
            logger.debug(f"Rotating mouth by {-mouth_coord.rotation} degrees")
            mouth = mouth.rotate(-mouth_coord.rotation, resample=Image.Resampling.BICUBIC)

        return mouth
    except Exception as e:
        logger.error(f"Error transforming mouth image {mouth_file}: {e}")
        raise


def bgra_to_rgba(image):
    # Swap blue and red channels
    try:
        logger.debug("Converting BGRA image to RGBA format")
        b, g, r, a = np.rollaxis(image, axis=-1)
        return np.dstack([r, g, b, a])
    except Exception as e:
        logger.error(f"Error converting BGRA to RGBA: {e}")
        raise


def render_frame(pose_img: Image, mouth_img: Image, mouth_coord):
    try:
        logger.debug("Rendering frame with mouth overlay")
        pose_img = bgra_to_rgba(pose_img)  # convert to rgba
        pose_img = Image.fromarray(pose_img)
        mouth_width, mouth_height = mouth_img.size

        # Location in pose image where mouth / viseme image will be added
        paste_coordinates = (
            int(mouth_coord.x - (mouth_width / 2)),
            int(mouth_coord.y - (mouth_height / 2)),
        )
        logger.debug(f"Pasting mouth at coordinates: {paste_coordinates}")

        # Paste the mouth image onto the face image at the specified coordinates
        pose_img.paste(im=mouth_img, box=paste_coordinates, mask=mouth_img)
        return pose_img
    except Exception as e:
        logger.error(f"Error rendering frame: {e}")
        raise
