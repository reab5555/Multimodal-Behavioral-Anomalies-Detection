def frame_to_timecode(frame_num, total_frames, duration):
    total_seconds = (frame_num / total_frames) * duration
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def seconds_to_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def timecode_to_seconds(timecode):
    h, m, s = map(int, timecode.split(':'))
    return h * 3600 + m * 60 + s


def add_timecode_to_image(image, timecode):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", 15)
    draw.text((10, 10), timecode, (255, 0, 0), font=font)
    return np.array(img_pil)


def add_timecode_to_image_body(image, timecode):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("arial.ttf", 100)
    draw.text((10, 10), timecode, (255, 0, 0), font=font)
    return np.array(img_pil)

