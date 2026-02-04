from pathlib import Path
import json
import time
import cv2
import numpy as np

import env_utils
import file_utils
import image_io
import sobel
import canny
import thresholding


# -------------------------
# Simple grid helpers (no viz.py)
# -------------------------
def to_uint8_bgr(img):
    """Convert any image to uint8 BGR for OpenCV."""
    if img.ndim == 2:
        img8 = image_io.as_uint8(image_io.normalize_to_01(img))
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    if img.ndim == 3 and img.shape[2] == 3:
        img8 = img if img.dtype == np.uint8 else image_io.as_uint8(image_io.normalize_to_01(img))
        return cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

    raise ValueError("Expected 2D grayscale or 3-channel image")


def add_label(img_bgr, text, font_scale=0.5, thickness=1):
    """Draw a small label in the top-left corner."""
    labeled = img_bgr.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(labeled, (0, 0), (tw + 6, th + 6), (0, 0, 0), -1)
    cv2.putText(labeled, text, (3, th + 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return labeled


def make_grid(images, order, ncols=2, pad=6, bg=0):
    """Create a labeled grid image from a dict of images."""
    tiles = []
    for name in order:
        if name in images:
            img_bgr = to_uint8_bgr(images[name])
            img_bgr = add_label(img_bgr, name)
            tiles.append(img_bgr)

    if not tiles:
        raise ValueError("No images to put in grid")

    max_h = max(t.shape[0] for t in tiles)
    max_w = max(t.shape[1] for t in tiles)

    resized = []
    for t in tiles:
        if t.shape[0] != max_h or t.shape[1] != max_w:
            t = cv2.resize(t, (max_w, max_h), interpolation=cv2.INTER_LINEAR)
        resized.append(t)

    n = len(resized)
    cols = min(max(1, ncols), n)
    rows = (n + cols - 1) // cols

    grid_h = rows * max_h + pad * (rows - 1)
    grid_w = cols * max_w + pad * (cols - 1)
    grid = np.full((grid_h, grid_w, 3), int(bg), dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            y = r * (max_h + pad)
            x = c * (max_w + pad)
            grid[y : y + max_h, x : x + max_w] = resized[idx]
            idx += 1

    return grid


def run_pipeline():
    # Project root = parent of src
    root = Path(__file__).resolve().parent.parent

    # Read env file at project root
    env_file = root / ".env"
    env_values = env_utils.read_env_file(env_file)
    print(f"Loaded .env dictionary: {env_values}\n")

    # Load configuration
    input_path = env_values.get("EDGE_INPUT", "data/images")
    output_path = env_values.get("EDGE_OUT", "results")
    methods_used = env_values.get("EDGE_METHODS", "sobel canny").split()

    padding_style = env_values.get("EDGE_PADDING", "reflect")
    threshold = float(env_values.get("EDGE_THRESHOLD", "0.2"))
    percentile = float(env_values.get("EDGE_PERCENTILE", "90"))
    canny_low = int(env_values.get("EDGE_CANNY_LOW", "50"))
    canny_high = int(env_values.get("EDGE_CANNY_HIGH", "150"))
    blur = int(env_values.get("EDGE_BLUR", "0"))
    sigma = float(env_values.get("EDGE_SIGMA", "1.0"))
    output_grid = env_values.get("EDGE_GRID", "true").lower() in ["1", "true", "yes", "y", "on"]

    # Resolve paths relative to project root
    root = Path(__file__).resolve().parent.parent
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_path.is_absolute():
        output_path = root / output_path

    print("Config loaded:")
    print(f"\tinput_path: {input_path}")
    print(f"\toutput_path: {output_path}")
    print(f"\tmethods used: {methods_used}\n")

    # Get images from input path
    images_list = file_utils.list_images_from_path(input_path)
    print(f"Found {len(images_list)} images\n")

    # Create output folders
    output_dirs = file_utils.ensure_output_dirs(output_path)
    print(f"Output folders created: {output_dirs}\n")

    # Process all images
    summary = {
        "parameters": {
            "input": str(input_path),
            "output": str(output_path),
            "methods": methods_used,
            "padding": padding_style,
            "threshold": threshold,
            "percentile": percentile,
            "canny_low": canny_low,
            "canny_high": canny_high,
            "blur": blur,
            "sigma": sigma,
            "grid": output_grid,
        },
        "files": [str(p) for p in images_list],
        "results": [],
    }

    for image_path in images_list:
        print(f"Processing: {image_path.name}")
        img = image_io.load_image(image_path, grayscale=True)
        item = {
            "file": str(image_path),
            "shape": list(img.shape),
            "methods": {},
        }
        # Images used for grid
        grid_images = {
            "grayscale": img
        }

        if "sobel" in methods_used:
            # Compute Sobel magnitude
            t0 = time.perf_counter()
            _, _, mag, _ = sobel.gradient_sobel(img, padding=padding_style)
            mag_norm = image_io.normalize_to_01(mag)

            # Apply thresholds
            bin_fixed = thresholding.threshold_fixed(mag_norm, threshold)
            bin_p, thr_p = thresholding.threshold_percentile(mag_norm, percentile)
            t1 = time.perf_counter()

            # Save outputs in a folder per image
            stem = image_path.stem
            sobel_dir = output_dirs["sobel"] / stem
            sobel_dir.mkdir(parents=True, exist_ok=True)
            image_io.save_image(sobel_dir / "mag.png", mag_norm)
            image_io.save_image(sobel_dir / "bin_fixed.png", bin_fixed)
            image_io.save_image(sobel_dir / f"bin_p{int(percentile)}.png", bin_p)

            item["methods"]["sobel"] = {
                "time_s": t1 - t0,
                "edge_ratio_fixed": float((bin_fixed > 0).mean()),
                "edge_ratio_p": float((bin_p > 0).mean()),
                "percentile_threshold": thr_p,
                "outputs": {
                    "mag": str(sobel_dir / "mag.png"),
                    "bin_fixed": str(sobel_dir / "bin_fixed.png"),
                    "bin_percentile": str(sobel_dir / f"bin_p{int(percentile)}.png"),
                },
            }

            grid_images["sobel mag"] = mag_norm
            grid_images["sobel bin"] = bin_fixed

        if "canny" in methods_used:
            # Compute Canny edges
            t0 = time.perf_counter()
            edges = canny.canny_edges(
                img,
                low=canny_low,
                high=canny_high,
                blur_k_size=blur,
                blur_sigma=sigma,
            )
            t1 = time.perf_counter()

            # Save outputs
            stem = image_path.stem
            canny_path = output_dirs["canny"] / f"{stem}_canny.png"
            image_io.save_image(canny_path, edges)

            item["methods"]["canny"] = {
                "time_s": t1 - t0,
                "edge_ratio": float((edges > 0).mean()),
                "outputs": {
                    "edges": str(canny_path),
                },
            }

            grid_images["canny"] = edges

        summary["results"].append(item)

        # Save grid with labels
        if output_grid:
            order = ["grayscale", "sobel mag", "sobel bin", "canny"]
            grid_img = make_grid(grid_images, order=order, ncols=2, pad=6, bg=0)
            grid_path = output_dirs["grids"] / f"{image_path.stem}_grid.png"
            image_io.save_image(grid_path, grid_img)
            item["grid"] = str(grid_path)

    summary_path = output_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    run_pipeline()
