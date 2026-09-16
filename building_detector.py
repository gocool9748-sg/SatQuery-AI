from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "building_model.onnx"

# Your current test image
IMAGE_PATH = Path(r"G:\SatQuery\images\building_test1.jpg")

OUTPUT_DIR = BASE_DIR / "runs"

OVERLAY_PATH = OUTPUT_DIR / "building_detection.png"
MASK_PATH = OUTPUT_DIR / "building_mask.png"
PROB_PATH = OUTPUT_DIR / "building_probability.png"

# HOTOSM sliding-window settings
TILE_SIZE = 256
STRIDE = 192

# Official HOTOSM threshold
THRESHOLD = 0.4371

# Remove extremely small regions
MIN_BUILDING_AREA = 80


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Building model not found:\n{MODEL_PATH}"
        )

    print("🧠 Loading HOTOSM building model...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    inp = session.get_inputs()[0]

    print("\nModel input:")
    print("  Name :", inp.name)
    print("  Shape:", inp.shape)
    print("  Type :", inp.type)

    for output in session.get_outputs():

        print("\nModel output:")
        print("  Name :", output.name)
        print("  Shape:", output.shape)
        print("  Type :", output.type)

    return session


# =========================================================
# PREPROCESS
# =========================================================

def preprocess_tile(tile_bgr):
    """
    Convert OpenCV BGR image to the normalization expected
    by the HOTOSM building model.

    Input:
        BGR uint8 image

    Output:
        NCHW float32 tensor
    """

    # BGR -> RGB
    tile_rgb = cv2.cvtColor(
        tile_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float [0,1]
    rgb = tile_rgb.astype(
        np.float32
    ) / 255.0

    # HOTOSM model normalization
    mean = np.array(
        [0.4297, 0.4002, 0.3433],
        dtype=np.float32
    ).reshape(3, 1, 1)

    std = np.array(
        [0.2056, 0.1674, 0.1599],
        dtype=np.float32
    ).reshape(3, 1, 1)

    # HWC -> CHW
    rgb = np.transpose(
        rgb,
        (2, 0, 1)
    )

    # Normalize
    rgb = (rgb - mean) / std

    # CHW -> NCHW
    tensor = np.expand_dims(
        rgb,
        axis=0
    )

    return tensor


# =========================================================
# PREDICT ONE TILE
# =========================================================

def predict_tile(session, tile_bgr):

    input_name = session.get_inputs()[0].name

    tensor = preprocess_tile(tile_bgr)

    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )

    # First model output
    raw = np.asarray(
        outputs[0][0],
        dtype=np.float32
    )

    print(
        f"\nRaw output shape: {raw.shape}"
    )

    print(
        f"Raw output range: "
        f"{raw.min():.4f} → {raw.max():.4f}"
    )

    if raw.ndim != 3:
        raise RuntimeError(
            f"Unexpected model output shape: {raw.shape}"
        )

    channels, height, width = raw.shape

    print(
        f"Output channels: {channels}"
    )

    # =====================================================
    # HOTOSM MODEL OUTPUT
    #
    # Channel 0 = building mask logits
    # Channel 1 = boundary
    # Channel 2 = signed distance
    #
    # We ONLY use channel 0.
    # =====================================================

    if channels < 1:
        raise RuntimeError(
            "Model returned no output channels."
        )

    building_logit = raw[0]

    # Logit -> probability
    probability = 1.0 / (
        1.0 + np.exp(
            -np.clip(
                building_logit,
                -50,
                50
            )
        )
    )

    probability = np.asarray(
        probability,
        dtype=np.float32
    )

    probability = np.clip(
        probability,
        0.0,
        1.0
    )

    return probability


# =========================================================
# TILE POSITIONS
# =========================================================

def make_positions(length):

    if length <= TILE_SIZE:
        return [0]

    positions = list(
        range(
            0,
            length - TILE_SIZE + 1,
            STRIDE
        )
    )

    last = length - TILE_SIZE

    if positions[-1] != last:
        positions.append(last)

    return positions


# =========================================================
# FULL IMAGE DETECTION
# =========================================================

def detect_buildings(session, image):

    height, width = image.shape[:2]

    xs = make_positions(width)
    ys = make_positions(height)

    print("\n🏠 Running building segmentation...")

    print(
        f"Image : {width} × {height}"
    )

    print(
        f"Tiles : {len(xs)} × {len(ys)} "
        f"= {len(xs) * len(ys)}"
    )

    print(
        f"Window: {TILE_SIZE}"
    )

    print(
        f"Stride: {STRIDE}"
    )

    # Accumulate overlapping predictions
    probability_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    probability_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    total_tiles = len(xs) * len(ys)

    completed = 0

    for y in ys:

        for x in xs:

            original_tile = image[
                y:y + TILE_SIZE,
                x:x + TILE_SIZE
            ]

            tile_h, tile_w = original_tile.shape[:2]

            # =================================================
            # PAD EDGE TILES
            # =================================================

            if (
                tile_h != TILE_SIZE
                or tile_w != TILE_SIZE
            ):

                tile = np.zeros(
                    (
                        TILE_SIZE,
                        TILE_SIZE,
                        3
                    ),
                    dtype=np.uint8
                )

                tile[
                    :tile_h,
                    :tile_w
                ] = original_tile

            else:

                tile = original_tile

            # =================================================
            # MODEL
            # =================================================

            prediction = predict_tile(
                session,
                tile
            )

            # Remove padding
            prediction = prediction[
                :tile_h,
                :tile_w
            ]

            # =================================================
            # OVERLAP AVERAGING
            # =================================================

            probability_sum[
                y:y + tile_h,
                x:x + tile_w
            ] += prediction

            probability_count[
                y:y + tile_h,
                x:x + tile_w
            ] += 1.0

            completed += 1

            print(
                f"\rProcessed "
                f"{completed}/{total_tiles} tiles",
                end=""
            )

    print()

    # =====================================================
    # FINAL PROBABILITY MAP
    # =====================================================

    probability_map = (
        probability_sum /
        np.maximum(
            probability_count,
            1.0
        )
    )

    return probability_map


# =========================================================
# CREATE BUILDING MASK
# =========================================================

def create_building_mask(
    probability_map
):

    # Threshold probability
    mask = (
        probability_map >= THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    # Very light cleanup
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# =========================================================
# COUNT BUILDINGS
# =========================================================

def analyze_buildings(mask):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    buildings = []

    for label in range(
        1,
        num_labels
    ):

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA
            ]
        )

        # Ignore tiny regions
        if area < MIN_BUILDING_AREA:
            continue

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP
            ]
        )

        width = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH
            ]
        )

        height = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT
            ]
        )

        buildings.append(
            {
                "area_px": area,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )

    return buildings


# =========================================================
# DRAW BUILDING OUTLINES + NUMBERS
# =========================================================

def create_overlay(
    image,
    mask,
    buildings
):

    output = image.copy()

    # Find connected building regions
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    building_number = 0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_BUILDING_AREA:
            continue

        building_number += 1

        # Draw outline
        cv2.polylines(
            output,
            [contour],
            True,
            (0, 255, 255),
            2
        )

        # Bounding box
        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Number
        cv2.putText(
            output,
            str(building_number),
            (
                x,
                max(
                    20,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("🏠 SATQUERY BUILDING DETECTOR")
    print("=" * 60)

    # =====================================================
    # CHECK IMAGE
    # =====================================================

    if not IMAGE_PATH.exists():

        print(
            "\n❌ Test image not found:"
        )

        print(
            IMAGE_PATH
        )

        print(
            "\nChange IMAGE_PATH "
            "near the top of this file."
        )

        return

    # =====================================================
    # LOAD MODEL
    # =====================================================

    session = load_model()

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_COLOR
    )

    if image is None:

        print(
            "\n❌ OpenCV could not read "
            "the image."
        )

        return

    height, width = image.shape[:2]

    print("\n📷 Input image")

    print(
        f"  Width : {width}"
    )

    print(
        f"  Height: {height}"
    )

    # =====================================================
    # DETECTION
    # =====================================================

    probability_map = detect_buildings(
        session,
        image
    )

    print(
        "\n✅ Segmentation complete."
    )

    print(
        "Probability range:",
        f"{probability_map.min():.4f}",
        "→",
        f"{probability_map.max():.4f}"
    )

    # =====================================================
    # MASK
    # =====================================================

    mask = create_building_mask(
        probability_map
    )

    # =====================================================
    # COUNT
    # =====================================================

    buildings = analyze_buildings(
        mask
    )

    building_pixels = int(
        np.count_nonzero(mask)
    )

    total_pixels = int(
        mask.size
    )

    coverage = (
        building_pixels /
        total_pixels *
        100.0
    )

    # =====================================================
    # OVERLAY
    # =====================================================

    overlay = create_overlay(
        image,
        mask,
        buildings
    )

    # =====================================================
    # OUTPUT DIRECTORY
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # =====================================================
    # SAVE MASK
    # =====================================================

    cv2.imwrite(
        str(MASK_PATH),
        mask
    )

    # =====================================================
    # SAVE OVERLAY
    # =====================================================

    cv2.imwrite(
        str(OVERLAY_PATH),
        overlay
    )

    # =====================================================
    # SAVE PROBABILITY MAP
    # =====================================================

    probability_uint8 = (
        np.clip(
            probability_map,
            0.0,
            1.0
        ) * 255
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        str(PROB_PATH),
        probability_uint8
    )

    # =====================================================
    # RESULTS
    # =====================================================

    print("\n" + "=" * 60)
    print("🏠 BUILDING RESULTS")
    print("=" * 60)

    print(
        f"Buildings detected : "
        f"{len(buildings)}"
    )

    print(
        f"Building coverage  : "
        f"{coverage:.2f}%"
    )

    print(
        f"Threshold           : "
        f"{THRESHOLD}"
    )

    print("=" * 60)

    print("\n📁 Saved:")

    print(
        f"  Overlay     : "
        f"{OVERLAY_PATH}"
    )

    print(
        f"  Mask        : "
        f"{MASK_PATH}"
    )

    print(
        f"  Probability : "
        f"{PROB_PATH}"
    )

    print("\n✅ Done!")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()