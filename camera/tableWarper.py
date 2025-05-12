import cv2 as cv
import numpy as np
from cv2 import aruco
import os, json, atexit

# 1) CONFIGURE YOUR TABLE “CANVAS” SIZE IN PIXELS
# These dimensions define the size of the output warped image (top-down view of the table).
TABLE_W, TABLE_H = 1000, 500  # Width and height of the table in pixels

# 2) MAP each marker ID → its desired (x, y) in that top-down view
# This dictionary maps ArUco marker IDs to their corresponding positions in the warped (top-down) view.
# Marker IDs 0–3 are used as the corners of the table.
DEST_PT = {
    0: [0,         0],          # Top-left corner of the table
    1: [TABLE_W,   0],          # Top-right corner of the table
    2: [TABLE_W,   TABLE_H],    # Bottom-right corner of the table
    3: [0,         TABLE_H],    # Bottom-left corner of the table
    4: [TABLE_W//2, 0],         # Top-middle (optional, not used for homography)
    5: [TABLE_W//2, TABLE_H],   # Bottom-middle (optional, not used for homography)
}

# ArUco dictionary defines the type of markers being used. DICT_6X6_250 means:
# - 6x6 grid markers
# - 250 unique marker IDs
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

_corner_cache = {} # maps corner ID → last seen (x,y) for each corner
# Local memory storage for previous corner_cache information (Pulls data initially and saves when done)
_CACHE_FILE = os.path.join(os.path.dirname(__file__), "corner_cache.json")
_CACHE_TMPFILE = _CACHE_FILE + ".tmp"

def _loadCache():
    """Load existing cache if it exists, and print out what was loaded."""
    if not os.path.exists(_CACHE_FILE):
        print(f"[warp] no cache file found at {_CACHE_FILE}")
        return
    try:
        with open(_CACHE_FILE, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            _corner_cache[int(k)] = np.array(v, dtype=np.float32)
        print(f"[warp] loaded corner_cache:", _corner_cache)
    except Exception as e:
        print(f"[warp] failed to load cache: {e}")

def saveCache():
    """Atomically write the full cache to disk, flushing and fsyncing."""
    try:
        # turn numpy arrays into pure-Python lists of floats
        data = { str(k): [ float(x) for x in v.tolist() ] 
                 for k, v in _corner_cache.items() }

        # write JSON out to a temp file first
        with open(_CACHE_TMPFILE, "w") as f:
            f.write(json.dumps(data))
            f.flush()
            os.fsync(f.fileno())

        # atomically overwrite the real cache file
        os.replace(_CACHE_TMPFILE, _CACHE_FILE)
        print(f"[warp] saved corner_cache ({len(data)} entries) to {_CACHE_FILE}")
    except Exception as e:
        print(f"[warp] failed to save cache: {e}")

_loadCache() # load existing cache on module import

def getWarpedFrame(frame, debug_mode = True):
    """
    Warps the input frame to a top-down view of the table using ArUco markers.

    Parameters:
    - frame: BGR image (numpy array) from your capture device.

    Returns:
    - warped: Top-down warped BGR frame of size (TABLE_W, TABLE_H).
    - None: If fewer than 4 corner markers (IDs 0, 1, 2, 3) are visible or homography fails.
    """
    corners, ids = applyGrayFiltersToFrameAndDetectMarkers(frame)

    # 1. Update corner cahche with the latest detected corners.
    if ids is not None:
        ids = ids.flatten()
        for c, mid in zip(corners, ids):
            if mid in (0,1,2,3):
                # use the marker’s *inner* corner, not its centroid:
                # c[0][0]=top-left, [1]=top-right, [2]=bot-right, [3]=bot-left
                # we’ll pick the iᵗʰ corner for marker ID i:
                pt = c[0][mid]
                _corner_cache[mid] = pt

    # 2. Build the src and dst points from the cache (fall back to the last seen point if not found)
    src_pts, dst_pts = [], []
    for mid in (0,1,2,3):
        if mid not in _corner_cache:
            # if we don’t have a corner for this marker, we can’t warp
            return None
        else:
            src_pts.append(_corner_cache[mid])
            dst_pts.append(DEST_PT[mid])

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    # Optional, draw rectangle on input for debugging
    if debug_mode:
        debugTableLines(frame, src, corners, ids)
        

    # 3. Complete and apply homography
    H, status = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    if H is None:
            print("Homography estimation failed")
            return None

    # Warp the input frame to the top-down view using the homography matrix.
    # - H: Homography matrix.
    # - (TABLE_W, TABLE_H): Size of the output warped image.
    warped = cv.warpPerspective(frame, H, (TABLE_W, TABLE_H))

    return warped

def applyGrayFiltersToFrameAndDetectMarkers(frame, ):
    params = populateArucoParamSettings() # Params for Aruco Detection Settings

    # 1) Color Background Gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 2) Gaussian blur to smooth out sensor noise
    gray = cv.GaussianBlur(gray, (5,5), 0)
    # 3) Histogram equalization to boost contrast
    gray = cv.equalizeHist(gray)
 
    # detect ArukoMarkers
    corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=params)
    return corners, ids

def debugTableLines(frame, src, corners, ids):
    debug = frame.copy()
    cv.polylines(debug,
                [src.reshape(-1,1,2).astype(int)],
                isClosed=True, color=(0,255,0), thickness=2)
    # Identify the Aruco markers in the video
    aruco.drawDetectedMarkers(debug, corners, ids)
    cv.imshow("Corner Cache Rectangle Table View", debug)

def populateArucoParamSettings():
    aruco_params = aruco.DetectorParameters()
    # Adaptive‐Thresh window sizes: smaller windows = more local adaptivity  
    aruco_params.adaptiveThreshWinSizeMin    = 3  
    aruco_params.adaptiveThreshWinSizeMax    = 23  
    aruco_params.adaptiveThreshWinSizeStep   = 10  

    # Constant subtracted from mean to fine-tune binarization  
    aruco_params.adaptiveThreshConstant      = 7  

    # Ignore tiny/huge blobs  
    aruco_params.minMarkerPerimeterRate      = 0.03  
    aruco_params.maxMarkerPerimeterRate      = 4.0  

    # Approximation accuracy (lower = tighter polygon fit)  
    aruco_params.polygonalApproxAccuracyRate = 0.05  

    # Enable sub-pixel corner refinement for sharper corner localization  
    aruco_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX  
    aruco_params.cornerRefinementWinSize     = 5  
    aruco_params.cornerRefinementMaxIterations = 30  
    aruco_params.cornerRefinementMinAccuracy = 0.1  

    return aruco_params

