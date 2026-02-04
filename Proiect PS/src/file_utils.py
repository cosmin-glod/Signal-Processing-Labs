from pathlib import Path


def list_images_from_path(input_path):
    """Return a list of image files (png/jpg/jpeg/bmp)"""
    extensions = [".png", ".jpg", ".jpeg", ".bmp"]
    path = Path(input_path)

    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input not found: {str(path)}")

    files = []
    for p in path.iterdir():
        if p.suffix.lower() in extensions:
            files.append(p)

    files.sort()
    if not files:
        raise FileNotFoundError(f"No images found in folder: {str(path)}")

    return files


def ensure_output_dirs(out_root):
    """Create output folders if they don't exist"""
    out_root = Path(out_root)
    dirs = {
        "sobel": out_root / "sobel",
        "canny": out_root / "canny",
        "grids": out_root / "grids",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs
