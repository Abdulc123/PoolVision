# pocket_selector.py

import cv2 as cv
import numpy as np
import config.settings as Settings
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame

POCKET_RADIUS = 40   # how big around each click you want to ignore
OUTPUT_FILE   = 'pockets.npy'

# list to hold pocket centers and their respective radii
pocket_centers = []

def on_mouse(event, x, y, flags, param):
    global pocket_centers, POCKET_RADIUS
    if event == cv.EVENT_LBUTTONDOWN:
        # Append the pocket position and the current radius
        pocket_centers.append((x, y, POCKET_RADIUS))
        print(f"  → Pocket #{len(pocket_centers)} at {(x, y)} with radius {POCKET_RADIUS}")

def update_radius(val):
    global POCKET_RADIUS
    POCKET_RADIUS = val

def main():
    # 1) set up window & mouse callback
    cv.namedWindow("Select Pockets")
    cv.setMouseCallback("Select Pockets", on_mouse)

    # Add a slider to adjust the pocket radius
    cv.createTrackbar("Radius", "Select Pockets", POCKET_RADIUS, 100, update_radius)

    cam = CameraCapture(
        camera_index=0,
        width=Settings.WARPED_TABLE_W,
        height=Settings.WARPED_TABLE_H
    )

    print("Click each pocket (6 total), then press 's' to save and quit, 'z' to undo the last pocket, or adjust the circle size using the slider.\n")

    while True:
        frame = cam.get_frame()
        warped = getWarpedFrame(frame, debug_mode=False)
        if warped is None:
            continue

        vis = warped.copy()
        # Draw circles for already clicked pockets with their respective radii
        for (px, py, radius) in pocket_centers:
            cv.circle(vis, (px, py), radius, (0, 0, 255), 2)

        cv.imshow("Select Pockets", vis)
        key = cv.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save pockets to disk and exit
            np.save(OUTPUT_FILE, np.array(pocket_centers))
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
