import base64
from io import BytesIO


def image_file_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        data_str = base64.b64encode(image_file.read()).decode("utf-8")
    image_type = image_path.split(".")[-1].lower()
    if image_type == "jpg":
        prefix = "data:image/jpeg;base64,"
    else:
        prefix = f"data:image/{image_type};base64,"
    return f"{prefix}{data_str}"


def image_data_to_base64(image):
    buffered = BytesIO()
    # 根据原始图片格式保存，不限制格式
    image_format = image.format if image.format else 'PNG'  # 获取图片格式，如果为空则默认为 PNG
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_url = f"data:image/{image_format.lower()};base64,{img_str}"
    return base64_url
