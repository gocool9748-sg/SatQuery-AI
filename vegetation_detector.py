import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def detect_vegetation(image_path):
    """
    Local RGB vegetation detector.

    IMPORTANT FIX vs. the previous version:
    The old code set its threshold with
        threshold = np.percentile(exg, 70.0)
    which ALWAYS classifies roughly the greenest 30% of *any* image as
    vegetation, regardless of whether real vegetation is present. That
    is the main source of false positives (deserts, parking lots,
    bare soil, even water glare would get ~30% "vegetation").

    This version instead:
      - uses a fixed, absolute ExG threshold as the primary evidence
        (vegetation-ness is judged against a fixed physical bar, not
        against the rest of the image)
      - requires genuine green dominance in raw RGB, not just a
        positive ExG value
      - adds a hue/saturation check in HSV so gray, cyan-water, or
        low-saturation pixels can't pass as vegetation
      - adds a sane brightness band (rejects near-black and blown-out
        pixels)
      - uses morphological cleanup + connected components for region
        counting and coverage, instead of naive greenest-quadrant
        counting

    Returns:
        vegetation_detected, coverage, regions, result_path,
        mask_path, region_data, method
    """

    image_path = str(image_path)

    # ---------------------------------------------------------
    # LOAD IMAGE
    # ---------------------------------------------------------

    img = np.asarray(Image.open(image_path).convert("RGB"))
    img_f = img.astype(np.float32)

    r = img_f[:, :, 0]
    g = img_f[:, :, 1]
    b = img_f[:, :, 2]

    h_img, w_img = img.shape[:2]
    image_area = h_img * w_img

    # ---------------------------------------------------------
    # EXCESS GREEN INDEX
    # ---------------------------------------------------------

    exg = 2.0 * g - r - b

    # ---------------------------------------------------------
    # HSV FOR HUE / SATURATION EVIDENCE
    # ---------------------------------------------------------

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0].astype(np.float32)   # 0-179 (OpenCV convention)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # ---------------------------------------------------------
    # ABSOLUTE COLOR EVIDENCE (the actual fix)
    # ---------------------------------------------------------

    # Fixed floor, not relative to this image's own distribution.
    # Typical vegetation ExG in 8-bit RGB sits well above ~25-30;
    # bare soil, concrete, and water usually sit at or below ~10-15.
    EXG_ABSOLUTE_FLOOR = 28.0

    strong_exg = exg > EXG_ABSOLUTE_FLOOR

    # Genuine green dominance in raw channels (not just the ExG
    # combination, which can be tricked by generally dark/desaturated
    # pixels).
    green_dominant = (
        (g > r + 12) &
        (g > b + 12)
    )

    # Plant-like hue band with a real saturation floor, so gray
    # asphalt or hazy/washed-out areas can't pass.
    green_hue = (
        (hue >= 28) &
        (hue <= 95) &
        (sat >= 40)
    )

    # Reasonable brightness band: reject near-black (deep shadow) and
    # blown-out highlights.
    reasonable_brightness = (val > 30) & (val < 250)

    # Exclude cyan/blue-green pixels that lean toward water rather
    # than vegetation (green present, but blue also elevated relative
    # to green).
    not_water_like = (b < g - 6)

    mask = (
        strong_exg &
        green_dominant &
        green_hue &
        reasonable_brightness &
        not_water_like
    )

    # ---------------------------------------------------------
    # MORPHOLOGICAL CLEANUP (remove speckle noise)
    # ---------------------------------------------------------

    mask_u8 = (mask.astype(np.uint8)) * 255

    open_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, open_kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_kernel)

    mask = mask_u8 > 0

    # ---------------------------------------------------------
    # VEGETATION COVERAGE
    # ---------------------------------------------------------

    coverage = float(np.mean(mask) * 100.0)

    # ---------------------------------------------------------
    # REAL REGION DETECTION (connected components, not quadrants)
    # ---------------------------------------------------------

    min_area = max(400, int(image_area * 0.00015))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    region_data = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[i]

        region_data.append({
            "area_pixels": area,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "center_x": round(float(cx), 2),
            "center_y": round(float(cy), 2),
        })

    region_data.sort(key=lambda item: item["area_pixels"], reverse=True)
    regions = len(region_data)

    # ---------------------------------------------------------
    # CREATE VEGETATION OVERLAY
    # ---------------------------------------------------------

    overlay = img.copy()
    overlay[mask] = np.array([40, 220, 80], dtype=np.uint8)

    blended = (0.65 * img_f + 0.35 * overlay.astype(np.float32)).astype(np.uint8)

    stem = Path(image_path).stem
    parent = Path(image_path).parent

    result_path = parent / f"{stem}_vegetation_result.png"
    mask_path = parent / f"{stem}_vegetation_mask.png"

    Image.fromarray(blended).save(result_path)
    Image.fromarray(mask_u8).save(mask_path)

    # ---------------------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------------------

    return {
        "vegetation_detected": regions > 0,
        "coverage": round(coverage, 2),
        "regions": regions,
        "result_path": str(result_path),
        "mask_path": str(mask_path),
        "region_data": region_data,
        "method": "LOCAL RGB + ExG (absolute threshold + hue/saturation gating)"
    }


# -------------------------------------------------------------
# DIRECT TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SATQUERY AI — VEGETATION DETECTOR")
    print("=" * 60)

    image_path = input("Enter the full path of the image to test:\n> ").strip()

    if not image_path:
        print("No image path provided.")
        raise SystemExit(1)

    try:
        result = detect_vegetation(image_path)

        print()
        print("=" * 60)
        print("VEGETATION RESULT")
        print("=" * 60)

        print(f"Vegetation detected: {result['vegetation_detected']}")
        print(f"Coverage: {result['coverage']:.2f}%")
        print(f"Regions: {result['regions']}")
        print(f"Method: {result['method']}")
        print(f"Result image: {result['result_path']}")
        print(f"Mask image: {result['mask_path']}")

        print("=" * 60)

    except Exception as e:
        print()
        print("VEGETATION DETECTOR ERROR:")
        print(e)