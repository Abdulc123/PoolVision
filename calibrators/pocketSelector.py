# pocket_selector.py

import cv2 as cv
import numpy as np
import json
import os
import config.settings as Settings
from PoolVision import PoolVision
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame

POCKET_RADIUS = 40   # how big around each click you want to ignore
OUTPUT_FILE   = os.path.join(os.path.dirname(__file__), "JsonData/pockets.json")

# list to hold pocket centers and their respective radii
pocket_centers = []

def on_mouse(event, x, y, flags, param):
    global pocket_centers, POCKET_RADIUS
    if event == cv.EVENT_LBUTTONDOWN:
        if len(pocket_centers) < 6:
            pocket_centers.append((x, y, POCKET_RADIUS))
            print(f"  → Pocket #{len(pocket_centers)} at {(x, y)} with radius {POCKET_RADIUS}")
        else:
            print("  → Already selected 6 pockets. Press 'z' to undo if needed.")

def update_radius(val):
    global POCKET_RADIUS
    POCKET_RADIUS = val

def main():
    # 1) set up window & mouse callback
    cv.namedWindow("Select Pockets")
    cv.setMouseCallback("Select Pockets", on_mouse)

    # Add a slider to adjust the pocket radius
    cv.createTrackbar("Radius", "Select Pockets", POCKET_RADIUS, 100, update_radius)

    poolVision = PoolVision(show_simulated_table=True)

    print("Click each pocket (6 total: 0=TL, 1=TM, 2=TR, 3=BR, 4=BM, 5=BL), then press 's' to save and quit, 'z' to undo the last pocket, or adjust the circle size using the slider.\n")

    while True:
        poolVision.convertFrameToWarpedFrame()
        warped = poolVision.warped_frame
        if warped is None:
            continue

        vis = warped.copy()
        # Draw circles for already clicked pockets with their respective radii
        for idx, (px, py, radius) in enumerate(pocket_centers):
            cv.circle(vis, (px, py), radius, (0, 0, 255), 2)
            cv.putText(vis, str(idx), (px+10, py-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv.imshow("Select Pockets", vis)
        key = cv.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save pockets to JSON and exit
            if len(pocket_centers) != 6:
                print("  → Please select exactly 6 pockets before saving.")
                continue
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            pockets_dict = {str(i): [int(px), int(py), int(radius)] for i, (px, py, radius) in enumerate(pocket_centers)}
            with open(OUTPUT_FILE, "w") as f:
                json.dump(pockets_dict, f, indent=2)
            print(f"\nSaved {len(pocket_centers)} pockets to '{OUTPUT_FILE}'")
            break
        elif key == ord('q'):
            print("Aborted without saving.")
            break
        elif key == ord('z'):
            # Undo the last pocket
            if pocket_centers:
                removed_pocket = pocket_centers.pop()
                print(f"  → Removed last pocket at {removed_pocket[:2]} with radius {removed_pocket[2]}")
            else:
                print("  → No pockets to undo.")

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
