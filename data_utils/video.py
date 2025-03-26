import json
import os, re
import platform
from typing import Optional

import moviepy.editor as mp

from ai_services.asr import OpenAIASRService
from ai_services.llm import OpenAILLMService
from data_utils.file import generate_checksum
from prompt_template import PromptTemplate
from pydub import AudioSegment
from pydub.utils import which
if platform.system() == "Windows":
    AudioSegment.converter = which("C:\\ffmpeg\\bin\\ffmpeg")


def parse_summary(text: str) -> str:
    # get all <summary> ... </summary> contents
    summaries = re.findall(r"<summary>(.*?)</summary>", text, re.DOTALL)
    return summaries[-1].strip() if summaries else text


class MovVideoLoader:
    def __init__(self):
        self.prompt_template = PromptTemplate.from_file("prompts/video_display_text.txt")
        llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
        asr_model = os.environ.get("ASR_MODEL", "whisper-1")
        self.llm = OpenAILLMService(llm_model)
        self.asr = OpenAIASRService(asr_model)
        self.temp_dir = self.create_temp_dir()

    def load(self, path: str) -> dict | list:
        if os.path.isfile(path):
            print(f"Loading video: {os.path.abspath(path)}")
            return self.load_video(path)
        elif os.path.isdir(path):
            results = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(".mov") or file.lower().endswith(".mp4"):
                        res = self.load(os.path.join(root, file))
                        results.append(res)
            return results
        else:
            raise ValueError(f"Invalid path: {path}")

    def data_exists(self, checksum: str) -> Optional[dict]:
        if os.path.exists(os.path.join(self.temp_dir, f"{checksum}.json")):
            data = json.load(open(
                os.path.join(self.temp_dir, f"{checksum}.json"),
                "r",
                encoding="utf-8"
            ))
            return data
        return None

    def load_video(self, path: str) -> dict:
        checksum = generate_checksum(path)
        video_filename = self.get_video_title(path)
        if (data := self.data_exists(checksum)) is not None:
            return data
        audio_path = os.path.join(self.temp_dir, f"{video_filename}.mp3")
        self.extract_audio(path, audio_path)
        transcription = self.transcribe_audio(os.path.abspath(audio_path))
        display_text = self.generate_display_text(transcription)
        display_text = parse_summary(display_text)
        title = self.get_video_title(path)
        current_directory = os.getcwd()
        target_file = os.path.abspath(path)
        relative_path = os.path.relpath(target_file, current_directory)
        data = {
            "text": display_text,
            "metadata": {
                "title": title,
                "transcript": transcription["text"],
                "display_text": display_text,
                "source_url": relative_path.replace(os.path.sep, '/'),
                "checksum": checksum
            }
        }
        self.save_video_metadata(data, checksum)
        # print(json.dumps(data, indent=4, ensure_ascii=False))
        return data

    @staticmethod
    def create_temp_dir():
        working_dir = os.environ.get("WORKING_DIR")
        if not working_dir or not os.path.exists(working_dir):
            raise ValueError(f"Working dir unavailable: {working_dir}")
        temp_dir = os.path.join(working_dir, ".tmp")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def extract_audio(self, video_path, audio_path):
        if video_path.lower().endswith(".mov"):
            video = mp.VideoFileClip(video_path)
            audio = video.audio

            # 将音频提取到临时文件
            filename = self.get_video_title(video_path)
            temp_wav_path = os.path.join(self.temp_dir, f"{filename}_temp.wav")
            audio.write_audiofile(temp_wav_path)
            video.close()
            audio = AudioSegment.from_wav(temp_wav_path)
            audio.export(audio_path, format="mp3")
            # 删除临时 WAV 文件
            os.remove(temp_wav_path)
        elif video_path.lower().endswith(".mp4"):
            audio = AudioSegment.from_file(video_path, format='mp4')
            audio.export(audio_path, format="mp3")
        else:
            raise ValueError(f"视频格式不支持：{video_path}")

    @staticmethod
    def transcribe_audio(audio_path) -> dict:
        trans = OpenAIASRService().transcribe(
            audio_path,
            timestamp_granularities=["segment"]
        )
        return trans

    def generate_display_text(self, transcription) -> str:
        text = transcription["text"]
        formatted_segments = self.format_segments(transcription)
        prompt = self.prompt_template.invoke(text=text, segments=formatted_segments)
        display_text = self.llm.invoke(
            prompt,
            temperature=0,
            seed=int(os.environ.get("SEED", 42))
        )
        return display_text

    @staticmethod
    def format_segments(transcription) -> str:
        def _format_seconds_to_timestamp(seconds):
            # 00:00:00
            hours = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60
            return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

        segments = transcription["segments"]
        formatted_segments = []
        for segment in segments:
            text = segment["text"].strip()
            if not text:
                continue
            start = _format_seconds_to_timestamp(segment["start"])
            end = _format_seconds_to_timestamp(segment["end"])
            formatted_segments.append(
                f"{start} - {end}: {text}"
            )
        return "\n".join(formatted_segments)

    def save_video_metadata(self, data, video_filename):
        with open(
                os.path.join(self.temp_dir, f"{video_filename}.json"),
                "w",
                encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def get_video_title(path) -> str:
        # filename without extension is the title
        return os.path.splitext(os.path.basename(path))[0]
