import sys

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import gradio as gr
import argparse
import os

from main.interface import (
    search_videos_by_keywords,
    get_keywords_from_image,
    MAX_VIDEO_COUNT,
)

# sys.stderr = open(os.devnull, 'w')

VIDEO_DIR = os.environ.get("VIDEO_DIR", "videos")
WORKING_DIR = os.environ.get("WORKING_DIR", "data")

DEFAULT_THRESHOLD = 0.45


def search_videos(keywords, threshold):
    keywords = keywords.split(", ")
    try:
        videos = search_videos_by_keywords(keywords, threshold)
    except Exception as e:
        print(e)
    results = []
    for video in videos:
        video_path = video['metadata']['source_url']
        if video_path.endswith((".mov",".MOV")):
            _video_path = '.'.join(video_path.split('.')[:-1])+".mp4"
            if os.path.exists(_video_path):
                video_path = _video_path
            else:
                print(".mov文件无法在Gradio页面中直接播放，建议使用.mp4格式")
        description = video['metadata']['display_text']
        description += f"\n\n*相关度: {video['score']:.2f}*"
        print(video_path)
        print(description)
        results.append((video_path, description))
    return results


def create_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            image_input = gr.Image(label="上传图片", type="pil", height=600)
        with gr.Row():
            clear_btn = gr.Button("清空")
            search_btn = gr.Button("搜索", interactive=False, variant="secondary")
            threshold_slider = gr.Slider(minimum=0.0, maximum=0.95, value=DEFAULT_THRESHOLD, step=0.01,
                                         label="相关度阈值")
        with gr.Row():
            keyword_frame = gr.Textbox(
                label="关键字",
                visible=False,
                show_label=True,
            )

        video_outputs = []
        for i in range(MAX_VIDEO_COUNT):  # 创建10个视频输出组件（隐藏）
            with gr.Row():
                with gr.Column(scale=1):  # 视频占1/3宽度
                    video = gr.Video(height=400, visible=False)
                with gr.Column(scale=2):  # 描述占2/3宽度
                    description = gr.Markdown(visible=False)
            video_outputs.append((video, description))

        def on_search(keywords, threshold):
            if not keywords:
                return [gr.update(value=None, visible=False) for _ in range(MAX_VIDEO_COUNT * 2)]
            results = search_videos(keywords, threshold)
            outputs = []
            for (video_path, desc), (video_comp, desc_comp) in zip(results, video_outputs):
                outputs.extend([
                    gr.update(value=video_path, visible=True),
                    gr.update(value=desc, visible=True)
                ])
            # 如果结果少于5个，隐藏多余的组件
            for _ in range(len(results), MAX_VIDEO_COUNT):
                outputs.extend([gr.update(value=None, visible=False), gr.update(value="", visible=False)])
            return outputs

        def on_clear():
            return [None, gr.update(value=DEFAULT_THRESHOLD), gr.update(visible=False)] + [
                gr.update(value=None, visible=False) for _ in range(MAX_VIDEO_COUNT * 2)
            ]

        def on_image_upload(image):
            if image is not None:
                keywords = get_keywords_from_image(image)
                text = ", ".join(keywords)
                return [
                    gr.update(interactive=True, variant="primary"),
                    gr.update(value=text, visible=True),
                ] + [gr.update(value=None, visible=False) for _ in range(MAX_VIDEO_COUNT * 2)]
            else:
                return [
                    gr.update(interactive=False, variant="secondary"),
                    gr.update(value=None, visible=False),
                ] + [gr.update(value=None, visible=False) for _ in range(MAX_VIDEO_COUNT * 2)]

        image_input.change(
            on_image_upload,
            inputs=[image_input],
            outputs=[search_btn, keyword_frame] + [comp for pair in video_outputs for comp in pair]
        )

        search_btn.click(
            on_search,
            inputs=[keyword_frame, threshold_slider],
            outputs=[comp for pair in video_outputs for comp in pair]
        )

        threshold_slider.release(
            on_search,
            inputs=[keyword_frame, threshold_slider],
            outputs=[comp for pair in video_outputs for comp in pair]
        )

        clear_btn.click(
            on_clear,
            outputs=[
                        image_input,
                        threshold_slider,
                        keyword_frame
                    ] + [comp for pair in video_outputs for comp in pair]
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="知识点拍照搜索视频")
    parser.add_argument("--port", type=int, default=8888, help="运行应用的端口号")
    args = parser.parse_args()

    demo = create_demo()
    data_dir = os.path.abspath(os.path.join(WORKING_DIR, VIDEO_DIR))
    demo.launch(server_name="0.0.0.0", server_port=args.port, allowed_paths=[data_dir], root_path="/video-search")


if __name__ == "__main__":
    main()
