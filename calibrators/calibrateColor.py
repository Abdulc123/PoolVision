import cv2 as cv
import numpy as np
import config.settings as Settings
from camera.capture import CameraCapture
from camera.TableWarper import getWarpedFrame, saveCache

# -----------------------------------------------------------------------------
# Globals to accumulate clicked‐pixel samples and store running HSV bounds
# -----------------------------------------------------------------------------
samples: list[np.ndarray] = []
lower = np.array([0,   0,   0], dtype=np.int32)
upper = np.array([179, 255, 255], dtype=np.int32)

def on_mouse(event, x, y, flags, param):
    """
    Mouse callback to sample HSV values around the click and update lower/upper bounds.
    Only samples if the click is on a ball (i.e. ball_mask[y,x] != 0).
    """
    global lower, upper, samples
    if event == cv.EVENT_LBUTTONDOWN:
        hsv = param['hsv']
        ball_mask = param['ball_mask']
        # ensure click is on a ball region
        if ball_mask[y, x] > 0:
            # grab a 10×10 patch around the click (clamp to image borders)
            y0, y1 = max(0, y-5), min(hsv.shape[0], y+5)
            x0, x1 = max(0, x-5), min(hsv.shape[1], x+5)
            patch = hsv[y0:y1, x0:x1]
            pixels = patch.reshape(-1, 3)
            samples.append(pixels)

            # stack and compute new min/max with padding
            all_pix = np.vstack(samples)
            min_vals = np.min(all_pix, axis=0) - np.array([5,20,20])
            max_vals = np.max(all_pix, axis=0) + np.array([5,20,20])

            # clamp to valid HSV ranges
            lower = np.maximum(min_vals, [0,0,0]).astype(np.int32)
            upper = np.minimum(max_vals, [179,255,255]).astype(np.int32)

            print(f"Updated HSV range -> lower={lower.tolist()}, upper={upper.tolist()}")

def main():
    cap = cv.VideoCapture(0)
    cv.namedWindow("Ball Calibration")
    # param dict to share hsv & ball_mask with the mouse callback
    param = {'hsv': None, 'ball_mask': None}
    cv.setMouseCallback("Ball Calibration", on_mouse, param)
    camera = CameraCapture(camera_index=0, width=Settings.WARPED_TABLE_W, height=Settings.WARPED_TABLE_H)

    while True:
        frame = camera.get_frame()

        # — your warp step here if you have one —
        warped = getWarpedFrame(frame, debug_mode=False)  

        hsv       = cv.cvtColor(warped, cv.COLOR_BGR2HSV)
        # mask out the green table
        table_mask = cv.inRange(hsv,
                                Settings.LOWER_GREEN_HSV_RANGE,
                                Settings.UPPER_GREEN_HSV_RANGE)
        # invert so balls are white (255) and everything else is black (0)
        ball_mask = cv.bitwise_not(table_mask)

        # share with callback
        param['hsv'] = hsv
        param['ball_mask'] = ball_mask

        # if we've sampled at least one patch, highlight the calibrated color
        if samples:
            color_mask = cv.inRange(hsv, lower, upper)
            highlight  = cv.bitwise_and(ball_mask, color_mask)
        else:
            highlight = ball_mask.copy()

        # visualize: overlay highlight on the warped image
        vis = warped.copy()
        # tint the detected pixels yellow for clarity
        vis[highlight > 0] = (0, 255, 255)

        text = f"LOW: {lower[0]},{lower[1]},{lower[2]}   UP: {upper[0]},{upper[1]},{upper[2]}"
        cv.putText(vis,
                    text,
                    (10, 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA)

        cv.imshow("Ball Calibration", vis)
        cv.imshow("Ball Area Mask (inverted)", ball_mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()