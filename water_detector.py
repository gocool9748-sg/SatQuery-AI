import cv2
import numpy as np
from pathlib import Path


def detect_water(image_path):
    """
    RGB water detector for satellite/aerial images, tuned to reduce
    false positives (shadows, asphalt, gray rooftops, low-saturation
    surfaces being mistaken for water).

    Key changes vs. the previous version:
      - much higher saturation / brightness floors on the hue checks,
        since low-saturation gray objects were passing as "blue water"
      - explicit shadow rejection (shadows under open sky pick up a
        blue color cast from skylight and were fooling the old blue
        checks)
      - tightened "low_red" / "blue_green_dominant" thresholds, which
        previously matched almost any non-red-dominant pixel
      - rebalanced score weights so several weak/ambiguous cues can no
        longer stack up past the water threshold on their own
      - raised the overall score cutoff

    IMPORTANT:
    This is still an RGB-based heuristic estimate, not NDWI / true
    multispectral water classification. If your imagery has a NIR
    band, NDWI = (Green - NIR) / (Green + NIR) thresholded around 0
    will be far more reliable than any RGB-color heuristic, because
    water absorbs NIR almost completely regardless of its visible
    color, while shadows/asphalt/roofs do not behave that way in NIR.
    Use this RGB version only when you don't have NIR data.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original = image.copy()
    h_img, w_img = image.shape[:2]
    image_area = h_img * w_img

    # =========================================================
    # 1. COLOR FEATURES
    # =========================================================

    b, g, r = cv2.split(image)

    b_f = b.astype(np.float32) / 255.0
    g_f = g.astype(np.float32) / 255.0
    r_f = r.astype(np.float32) / 255.0

    total = r_f + g_f + b_f + 1e-6

    rn = r_f / total
    gn = g_f / total
    bn = b_f / total

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # =========================================================
    # 2. WATER COLOR EVIDENCE
    # =========================================================

    blue_advantage = b.astype(np.int16) - r.astype(np.int16)
    green_advantage = g.astype(np.int16) - r.astype(np.int16)

    # Tightened: was (-2, >2). Loose versions of this matched almost
    # any pixel that wasn't strongly red.
    blue_green_over_red = (
        (blue_advantage > 4) &
        (green_advantage > 6)
    )

    # Tightened blue thresholds (was +8 / -12).
    strong_blue = (
        (b.astype(np.int16) >= r.astype(np.int16) + 18) &
        (b.astype(np.int16) >= g.astype(np.int16) - 6)
    )

    cyan_water = (
        (g.astype(np.int16) >= r.astype(np.int16) + 10) &
        (b.astype(np.int16) >= r.astype(np.int16) + 8) &
        (np.abs(g.astype(np.int16) - b.astype(np.int16)) <= 70) &
        (sat >= 35)
    )

    # Hue families with much higher saturation/value floors than
    # before (old sat>=8-12 let near-gray pixels through).
    blue_hue = (
        (hue >= 85) &
        (hue <= 135) &
        (sat >= 45) &
        (val >= 35)
    )

    cyan_teal_hue = (
        (hue >= 55) &
        (hue < 100) &
        (sat >= 40) &
        (val >= 35)
    )

    # Greenish water is the easiest to confuse with vegetation, so
    # keep it narrow and require the pixel to NOT look vegetation-like.
    exg_precheck = 2.0 * g_f - r_f - b_f
    greenish_water_hue = (
        (hue >= 40) &
        (hue < 70) &
        (sat >= 40) &
        (val >= 45) &
        (exg_precheck < 0.06)
    )

    # Tightened normalized-color checks (old versions matched most
    # non-red pixels in the image).
    low_red = rn < 0.34
    blue_green_dominant = (bn + gn) > rn + 0.22

    # =========================================================
    # 3. TEXTURE
    # =========================================================

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean = cv2.GaussianBlur(gray, (11, 11), 0)
    mean_sq = cv2.GaussianBlur(gray * gray, (11, 11), 0)

    variance = np.maximum(mean_sq - mean * mean, 0)
    local_std = np.sqrt(variance)

    smooth = local_std < 42
    very_smooth = local_std < 26

    # =========================================================
    # 4. SHADOW REJECTION (new)
    # =========================================================

    # Shadows lit only by blue skylight are dim, only moderately
    # saturated, and can show a mild blue/cyan cast — which is exactly
    # what used to trip the old blue/cyan checks. Distinguish them
    # from real water mainly by low absolute brightness plus modest
    # saturation (real water reflecting open sky is usually brighter
    # and/or more saturated than shadow regions).
    shadow_like = (
        (val < 95) &
        (sat < 70) &
        (blue_advantage > -5) &
        (blue_advantage < 30)
    )

    # =========================================================
    # 5. VEGETATION REJECTION
    # =========================================================

    exg = 2.0 * g_f - r_f - b_f

    strong_vegetation = (
        (exg > 0.16) &
        (g.astype(np.int16) > r.astype(np.int16) + 12) &
        (g.astype(np.int16) > b.astype(np.int16) + 8)
    )

    textured_vegetation = (
        (exg > 0.10) &
        (local_std > 20) &
        (g.astype(np.int16) > r.astype(np.int16) + 8)
    )

    vegetation_hard_reject = (
        strong_vegetation &
        ~blue_hue &
        ~cyan_teal_hue &
        ~strong_blue
    )

    # =========================================================
    # 6. WATER SCORE
    # =========================================================

    score = np.zeros_like(r_f, dtype=np.float32)

    # Strong, fairly specific cues keep high weight.
    score += blue_hue.astype(np.float32) * 2.0
    score += cyan_teal_hue.astype(np.float32) * 1.7
    score += greenish_water_hue.astype(np.float32) * 0.7
    score += strong_blue.astype(np.float32) * 1.8
    score += cyan_water.astype(np.float32) * 1.5
    score += blue_green_over_red.astype(np.float32) * 1.0

    # Weak / ambiguous cues get lower weight so they can no longer
    # stack up past the threshold on their own.
    score += blue_green_dominant.astype(np.float32) * 0.5
    score += low_red.astype(np.float32) * 0.4
    score += smooth.astype(np.float32) * 0.4
    score += very_smooth.astype(np.float32) * 0.25

    # Penalize vegetation.
    score -= strong_vegetation.astype(np.float32) * 1.5
    score -= textured_vegetation.astype(np.float32) * 0.8

    # Penalize shadow-like pixels heavily — this is the main new
    # false-positive fix.
    score -= shadow_like.astype(np.float32) * 2.2

    # Raised cutoff (was 3.6): now requires multiple real cues, not
    # just a pile of weak ones.
    water_candidate = score >= 4.6

    # A pixel can also qualify via strong, unambiguous color evidence
    # even if the additive score falls a bit short — but now this path
    # also requires a real saturation/brightness floor and rejects
    # shadow-like pixels explicitly.
    strong_color_water = (
        (blue_hue | cyan_teal_hue | strong_blue | cyan_water) &
        low_red &
        (smooth | very_smooth) &
        (sat >= 40) &
        (val >= 35) &
        ~vegetation_hard_reject &
        ~shadow_like
    )

    water_candidate |= strong_color_water
    water_candidate &= ~shadow_like

    # =========================================================
    # 7. REMOVE COMMON NON-WATER COLORS
    # =========================================================

    bright_neutral = (val > 215) & (sat < 30)
    water_candidate &= ~bright_neutral

    # Widened dark-neutral / shadow floor (was val<20, sat<30).
    dark_neutral = (val < 40) & (sat < 45)
    water_candidate &= ~dark_neutral

    # =========================================================
    # 8. MORPHOLOGICAL CLEANUP
    # =========================================================

    mask = water_candidate.astype(np.uint8) * 255

    close_kernel = np.ones((9, 9), np.uint8)
    open_kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = np.where(blurred >= 110, 255, 0).astype(np.uint8)

    # =========================================================
    # 9. CONNECTED COMPONENTS
    # =========================================================

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    clean_mask = np.zeros_like(mask)

    # Slightly higher minimum area than before to drop small
    # speckle false positives (was 0.00012 of image area).
    min_area = max(700, int(image_area * 0.00020))

    raw_regions = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        aspect_ratio = max(w, h) / max(1, min(w, h))

        # Reject thin isolated artifacts (common shape for shadow
        # slivers along building edges).
        if aspect_ratio > 50 and area < image_area * 0.02:
            continue

        clean_mask[labels == i] = 255

        cx, cy = centroids[i]

        raw_regions.append({
            "area_pixels": area,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "center_x": round(float(cx), 2),
            "center_y": round(float(cy), 2),
        })

    mask = clean_mask

    # =========================================================
    # 10. MERGE NEARBY WATER SECTIONS FOR REGION COUNTING
    # =========================================================

    grouping_kernel = np.ones((45, 45), np.uint8)
    grouped = cv2.dilate(mask, grouping_kernel, iterations=1)

    grouped_num, grouped_labels, _, _ = cv2.connectedComponentsWithStats(
        grouped, connectivity=8
    )

    regions = []

    for i in range(1, grouped_num):
        group_pixels = (grouped_labels == i)
        actual_water = group_pixels & (mask > 0)
        area = int(np.count_nonzero(actual_water))

        if area < min_area:
            continue

        ys, xs = np.where(actual_water)
        if len(xs) == 0:
            continue

        x = int(xs.min())
        y = int(ys.min())
        right = int(xs.max())
        bottom = int(ys.max())

        regions.append({
            "area_pixels": area,
            "x": x,
            "y": y,
            "width": right - x + 1,
            "height": bottom - y + 1,
            "center_x": round(float(xs.mean()), 2),
            "center_y": round(float(ys.mean()), 2),
        })

    regions.sort(key=lambda item: item["area_pixels"], reverse=True)

    # =========================================================
    # 11. COVERAGE
    # =========================================================

    water_pixels = int(np.count_nonzero(mask))
    coverage = (water_pixels / max(1, image_area)) * 100.0

    # =========================================================
    # 12. LARGEST WATER LOCATION
    # =========================================================

    if regions:
        largest = regions[0]
        cx = largest["center_x"]
        cy = largest["center_y"]

        if cx < w_img * 0.33:
            horizontal = "left"
        elif cx < w_img * 0.66:
            horizontal = "center"
        else:
            horizontal = "right"

        if cy < h_img * 0.33:
            vertical = "top"
        elif cy < h_img * 0.66:
            vertical = "middle"
        else:
            vertical = "bottom"

        largest_location = f"{vertical}-{horizontal}"
    else:
        largest_location = "none"

    # =========================================================
    # 13. CREATE OUTPUT OVERLAY
    # =========================================================

    overlay = original.copy()
    overlay[mask > 0] = (255, 120, 0)

    result = cv2.addWeighted(original, 0.68, overlay, 0.32, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 0, 0), 2)

    for index, region in enumerate(regions, start=1):
        x = region["x"]
        y = region["y"]

        cv2.putText(
            result,
            f"Water {index}",
            (x, max(22, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    # =========================================================
    # 14. SAVE OUTPUTS
    # =========================================================

    output_dir = Path("runs")
    output_dir.mkdir(exist_ok=True)

    mask_path = output_dir / "water_mask_rgb.png"
    result_path = output_dir / "water_detection.png"

    score_norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    score_path = output_dir / "water_probability.png"

    # Also save the shadow mask so you can visually check what's
    # being rejected — very useful for tuning on your own imagery.
    shadow_path = output_dir / "shadow_rejected.png"

    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(result_path), result)
    cv2.imwrite(str(score_path), score_norm)
    cv2.imwrite(str(shadow_path), (shadow_like.astype(np.uint8) * 255))

    # =========================================================
    # 15. RETURN SAME API AS BEFORE
    # =========================================================

    return {
        "water_detected": len(regions) > 0,
        "coverage": round(float(coverage), 2),
        "regions": len(regions),
        "largest_region_area": regions[0]["area_pixels"] if regions else 0,
        "largest_location": largest_location,
        "mask_path": str(mask_path),
        "result_path": str(result_path),
        "probability_path": str(score_path),
        "shadow_rejected_path": str(shadow_path),
        "region_data": regions,
    }


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WATER DETECTOR TEST (false-positive-reduced)")
    print("=" * 60)

    image_input = input("\nEnter the full path of the image to test:\n> ").strip().strip('"')

    if not image_input:
        print("\nERROR: No image path was provided.")
    else:
        test_image = Path(image_input)

        if not test_image.exists():
            print("\nERROR: Image not found:")
            print(test_image)
        else:
            try:
                test_result = detect_water(str(test_image))

                print("\n" + "=" * 60)
                print("DETECTION RESULT")
                print("=" * 60)

                print(f"Image           : {test_image}")
                print(f"Water detected  : {test_result['water_detected']}")
                print(f"Water coverage  : {test_result['coverage']}%")
                print(f"Water regions   : {test_result['regions']}")
                print(f"Largest area    : {test_result['largest_region_area']} pixels")
                print(f"Largest location: {test_result['largest_location']}")

                print("\nGenerated files:")
                print(f"Mask             : {test_result['mask_path']}")
                print(f"Detection        : {test_result['result_path']}")
                print(f"Probability      : {test_result['probability_path']}")
                print(f"Shadow-rejected  : {test_result['shadow_rejected_path']}")

                print("\nSUCCESS: water detector test completed.")

            except Exception as e:
                print(f"\nERROR while testing water detector: {e}")