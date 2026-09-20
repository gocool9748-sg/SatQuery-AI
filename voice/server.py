from pathlib import Path
import sys, json, os
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
INDEX_FILE = BASE_DIR / "index.html"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from water_detector import detect_water
    WATER_DETECTOR_AVAILABLE = True
    WATER_IMPORT_ERROR = None
except Exception as e:
    detect_water = None
    WATER_DETECTOR_AVAILABLE = False
    WATER_IMPORT_ERROR = str(e)

try:
    from vegetation_detector import detect_vegetation
    VEGETATION_DETECTOR_AVAILABLE = True
    VEGETATION_IMPORT_ERROR = None
except Exception as e:
    detect_vegetation = None
    VEGETATION_DETECTOR_AVAILABLE = False
    VEGETATION_IMPORT_ERROR = str(e)

try:
    from building_detector import (
        load_model as load_building_model,
        detect_buildings,
        create_building_mask,
        analyze_buildings,
    )
    BUILDING_DETECTOR_AVAILABLE = True
    BUILDING_IMPORT_ERROR = None
except Exception as e:
    load_building_model = None
    detect_buildings = None
    create_building_mask = None
    analyze_buildings = None
    BUILDING_DETECTOR_AVAILABLE = False
    BUILDING_IMPORT_ERROR = str(e)

# DOTA / YOLO satellite object detector.
# satellite_detector.py expects its model path relative to the project root,
# so temporarily import it from the project directory.
try:
    _old_cwd = os.getcwd()
    os.chdir(PROJECT_DIR)
    from satellite_detector import detect_satellite_objects
    SATELLITE_DETECTOR_AVAILABLE = True
    SATELLITE_IMPORT_ERROR = None
except Exception as e:
    detect_satellite_objects = None
    SATELLITE_DETECTOR_AVAILABLE = False
    SATELLITE_IMPORT_ERROR = str(e)
finally:
    try:
        os.chdir(_old_cwd)
    except Exception:
        pass

BUILDING_SESSION = None

app = FastAPI(title="SatQuery AI Voice Server")

SHARED_VOICE_IMAGE = PROJECT_DIR / "runs" / "voice_current_image.png"
DEFAULT_IMAGE = PROJECT_DIR / "urbanwater.jpg"

def get_current_image():
    env_path = os.getenv("SATQUERY_IMAGE_PATH")
    if env_path:
        return Path(env_path)
    if SHARED_VOICE_IMAGE.exists():
        return SHARED_VOICE_IMAGE
    return DEFAULT_IMAGE

@app.get("/")
def home():
    return FileResponse(INDEX_FILE)

@app.get("/health")
def health():
    image_path = get_current_image()
    return {
        "status": "ok",
        "service": "SatQuery AI Voice Server",
        "water_detector": WATER_DETECTOR_AVAILABLE,
        "vegetation_detector": VEGETATION_DETECTOR_AVAILABLE,
        "building_detector": BUILDING_DETECTOR_AVAILABLE,
        "satellite_object_detector": SATELLITE_DETECTOR_AVAILABLE,
        "current_image": str(image_path),
        "current_image_exists": image_path.exists(),
        "shared_voice_image_exists": SHARED_VOICE_IMAGE.exists(),
        "using_shared_image": image_path == SHARED_VOICE_IMAGE,
    }

@app.get("/api/current-image")
def current_image_status():
    image_path = get_current_image()
    return {
        "success": image_path.exists(),
        "image_path": str(image_path),
        "shared_voice_image": str(SHARED_VOICE_IMAGE),
        "shared_voice_image_exists": SHARED_VOICE_IMAGE.exists(),
        "using_shared_image": image_path == SHARED_VOICE_IMAGE,
    }

@app.get("/api/analyze-water")
def analyze_water_direct():
    if not WATER_DETECTOR_AVAILABLE:
        return JSONResponse(status_code=500, content={"success": False, "error": "water_detector.py could not be imported", "details": WATER_IMPORT_ERROR})
    image_path = get_current_image()
    if not image_path.exists():
        return {"success": False, "error": "Image not found", "image_path": str(image_path)}
    try:
        return {"success": True, "image_path": str(image_path), "analysis": detect_water(str(image_path))}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/analyze-building")
def analyze_building_direct():
    if not BUILDING_DETECTOR_AVAILABLE:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "building_detector.py could not be imported",
            "details": BUILDING_IMPORT_ERROR,
        })
    image_path = get_current_image()
    if not image_path.exists():
        return {"success": False, "error": "Image not found", "image_path": str(image_path)}
    try:
        result = building_voice_analysis(image_path)
        return {"success": True, "image_path": str(image_path), "analysis": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/analyze-vegetation")
def analyze_vegetation_direct():
    if not VEGETATION_DETECTOR_AVAILABLE:
        return JSONResponse(status_code=500, content={"success": False, "error": "vegetation_detector.py could not be imported", "details": VEGETATION_IMPORT_ERROR})
    image_path = get_current_image()
    if not image_path.exists():
        return {"success": False, "error": "Image not found", "image_path": str(image_path)}
    try:
        return {"success": True, "image_path": str(image_path), "analysis": detect_vegetation(str(image_path))}
    except Exception as e:
        return {"success": False, "error": str(e)}

def parse_tool_calls(message):
    calls = message.get("toolCallList") or message.get("toolCalls") or []
    parsed = []
    for tool_call in calls:
        function_data = tool_call.get("function", {})
        if not isinstance(function_data, dict):
            function_data = {}
        tool_call_id = tool_call.get("id") or function_data.get("id")
        tool_name = tool_call.get("name") or function_data.get("name")
        raw_arguments = tool_call.get("arguments") if tool_call.get("arguments") is not None else function_data.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}
        parsed.append((tool_call_id, tool_name, arguments))
    return parsed

def water_voice_result(image_path):
    if not WATER_DETECTOR_AVAILABLE:
        return "The water detection system is currently unavailable because the water detector could not be loaded."
    if not image_path.exists():
        return "The satellite image is not available right now."
    try:
        d = detect_water(str(image_path))
        if d.get("water_detected", False):
            return (f"Water was detected in the satellite image. Water coverage is {float(d.get('coverage', 0)):.2f} percent. "
                    f"There are {int(d.get('regions', 0))} detected water regions. The largest water region is located at {d.get('largest_location', 'unknown')}.")
        return "No significant water was detected in the satellite image."
    except Exception as e:
        print("WATER DETECTOR ERROR:", e)
        return f"The water analysis failed. The detector reported: {e}"

def building_voice_result(image_path):
    global BUILDING_SESSION

    if not BUILDING_DETECTOR_AVAILABLE:
        return "The building detection system is currently unavailable because the building detector could not be loaded."
    if not image_path.exists():
        return "The satellite image is not available right now."

    try:
        if BUILDING_SESSION is None:
            print("LOADING BUILDING MODEL FOR VOICE...")
            BUILDING_SESSION = load_building_model()

        image = __import__("cv2").imdecode(
            __import__("numpy").fromfile(str(image_path), dtype=__import__("numpy").uint8),
            __import__("cv2").IMREAD_COLOR,
        )
        if image is None:
            return "I could not read the current satellite image for building analysis."

        probability_map = detect_buildings(BUILDING_SESSION, image)
        mask = create_building_mask(probability_map)
        buildings = analyze_buildings(mask)

        import numpy as np
        building_pixels = int(np.count_nonzero(mask))
        total_pixels = int(mask.size)
        coverage = (building_pixels / total_pixels * 100.0) if total_pixels else 0.0

        count = len(buildings)
        if count == 0:
            return f"No significant buildings were detected in the satellite image. Building coverage is {coverage:.2f} percent."

        return (
            f"I detected {count} buildings in the satellite image. "
            f"Building coverage is {coverage:.2f} percent."
        )
    except Exception as e:
        print("BUILDING DETECTOR ERROR:", e)
        return f"The building analysis failed. The detector reported: {e}"


def building_voice_analysis(image_path):
    global BUILDING_SESSION
    if BUILDING_SESSION is None:
        BUILDING_SESSION = load_building_model()
    import cv2
    import numpy as np
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV could not read the current satellite image.")
    probability_map = detect_buildings(BUILDING_SESSION, image)
    mask = create_building_mask(probability_map)
    buildings = analyze_buildings(mask)
    building_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    coverage = (building_pixels / total_pixels * 100.0) if total_pixels else 0.0
    return {
        "building_count": len(buildings),
        "coverage": coverage,
        "threshold": 0.4371,
    }

def object_voice_result(image_path):
    """Run the existing DOTA/YOLO OBB detector on the current image."""
    if not SATELLITE_DETECTOR_AVAILABLE:
        return (
            "The satellite object detection system is currently unavailable "
            "because the object detector could not be loaded."
        )
    if not image_path.exists():
        return "The satellite image is not available right now."

    try:
        results = detect_satellite_objects(str(image_path))
        total_objects = 0
        class_counts = {}

        for result in results:
            if result.obb is None or result.obb.cls is None:
                continue

            for i in range(len(result.obb.cls)):
                class_id = int(result.obb.cls[i])
                class_name = result.names[class_id]
                total_objects += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if total_objects == 0:
            return "No satellite objects were detected in the current image."

        ordered = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
        breakdown = ", ".join(
            f"{count} {name}" for name, count in ordered[:8]
        )

        return (
            f"I detected {total_objects} satellite objects in the image. "
            f"The detected classes are: {breakdown}."
        )
    except Exception as e:
        print("SATELLITE OBJECT DETECTOR ERROR:", e)
        return f"The satellite object analysis failed. The detector reported: {e}"


@app.get("/api/analyze-objects")
def analyze_objects_direct():
    if not SATELLITE_DETECTOR_AVAILABLE:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "satellite_detector.py could not be imported",
            "details": SATELLITE_IMPORT_ERROR,
        })
    image_path = get_current_image()
    if not image_path.exists():
        return {"success": False, "error": "Image not found", "image_path": str(image_path)}
    try:
        return {
            "success": True,
            "image_path": str(image_path),
            "analysis": object_voice_result(image_path),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def vegetation_voice_result(image_path):
    if not VEGETATION_DETECTOR_AVAILABLE:
        return "The vegetation detection system is currently unavailable because the vegetation detector could not be loaded."
    if not image_path.exists():
        return "The satellite image is not available right now."
    try:
        d = detect_vegetation(str(image_path))
        if d.get("vegetation_detected", False):
            return (f"Vegetation was detected in the satellite image. Vegetation coverage is {float(d.get('coverage', 0)):.2f} percent. "
                    f"There are approximately {int(d.get('regions', 0))} vegetation regions.")
        return "No significant vegetation was detected in the satellite image."
    except Exception as e:
        print("VEGETATION DETECTOR ERROR:", e)
        return f"The vegetation analysis failed. The detector reported: {e}"

@app.post("/vapi/tools")
async def vapi_tools(request: Request):
    try:
        body = await request.json()
        print("\n" + "=" * 70 + "\nVAPI TOOL CALL RECEIVED\n" + "=" * 70)
        print(json.dumps(body, indent=2, default=str))
        message = body.get("message", {})
        results = []
        for tool_call_id, tool_name, arguments in parse_tool_calls(message):
            print("\n" + "-" * 70)
            print("TOOL NAME:", tool_name)
            print("TOOL CALL ID:", tool_call_id)
            print("ARGUMENTS:", json.dumps(arguments, indent=2, default=str))
            image_path = get_current_image()
            print("IMAGE BEING ANALYZED:", image_path)
            if tool_name == "analyze_water":
                result_text = water_voice_result(image_path)
            elif tool_name == "analyze_vegetation":
                result_text = vegetation_voice_result(image_path)
            elif tool_name in ("analyze_buildings", "analyze_building"):
                result_text = building_voice_result(image_path)
            elif tool_name in ("analyze_objects", "analyze_satellite_objects"):
                result_text = object_voice_result(image_path)
            else:
                result_text = f"The SatQuery tool named {tool_name} is not available."
            result_text = " ".join(str(result_text).split())
            results.append({"toolCallId": tool_call_id, "result": result_text})
            print("RESULT SENT TO VAPI:", result_text)
        response = {"results": results}
        print("\nFINAL RESPONSE TO VAPI\n", json.dumps(response, indent=2, default=str))
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        print("VAPI SERVER ERROR:", e)
        return JSONResponse(status_code=200, content={"results": [{"toolCallId": "unknown", "error": str(e)}]})

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
