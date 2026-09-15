# utils/image_sizes.py  (or any .py on your path)
import imagesize

def image_hw(path: str) -> tuple[str, tuple[int, int]] | None:
    try:
        width, height = imagesize.get(path)
        if width < 0 or height < 0:
            return None
        return (path, (height, width))
    except Exception:
        return None