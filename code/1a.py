import os
import cv2
import numpy as np

# 2-DoF SSD
def simple_tracker(roi, im0, im1, max_iterations, threshold):
    """
    Func definitions:
    roi: [x, y, w, h] ROI in im0 to track in im1
    im0: Previous frame
    im1: Current frame
    max_iterations: Self-explanatory
    threshold: Convergence threshold

    return: [x_new, y_new, w, h] Updated ROI in im1
    """
    if len(im0.shape) == 3:
        im0_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    else:
        im0_gray = im0.copy()
    if len(im1.shape) == 3:
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        im1_gray = im1.copy()

    x, y, w, h = roi
    template = im0_gray[y:y+h, x:x+w]
    if template.size == 0:
        return roi

    u, v = 0.0, 0.0
    gradx = cv2.Sobel(im1_gray, cv2.CV_32F, 1, 0)
    grady = cv2.Sobel(im1_gray, cv2.CV_32F, 0, 1)

    for k in range(max_iterations):
        x1 = int(round(x + u))
        y1 = int(round(y + v))

        # Check ROI boundaries to stay within image
        x1 = max(0, min(im1_gray.shape[1] - w, x1))
        y1 = max(0, min(im1_gray.shape[0] - h, y1))

        patch = im1_gray[y1:y1+h, x1:x1+w]
        if patch.shape != template.shape:
            break

        error = template.astype(np.float32) - patch.astype(np.float32)
        patch_gradx = gradx[y1:y1+h, x1:x1+w]
        patch_grady = grady[y1:y1+h, x1:x1+w]

        Ix = patch_gradx.reshape(-1)
        Iy = patch_grady.reshape(-1)
        E  = error.reshape(-1)

        A = np.vstack((Ix, Iy)).T  # [Ix, Iy]

        # Solve A * [du, dv] = E
        try:
            Q, R = np.linalg.qr(A)
            delta = np.linalg.solve(R, Q.T @ E)
            du, dv = delta[0], delta[1]
        except np.linalg.LinAlgError:
            break

        u_new = u + du
        v_new = v + dv

        if np.sqrt(du**2 + dv**2) < threshold:
            u, v = u_new, v_new
            break

        u, v = u_new, v_new

    # Ensure final ROI stays within the image
    x_new = max(0, min(im1_gray.shape[1] - w, int(round(x + u))))
    y_new = max(0, min(im1_gray.shape[0] - h, int(round(y + v))))

    return [x_new, y_new, w, h]

# Read images and function call
folder = "bus"
all_images = []
count = 0

# Filter image files (on macOS, to prevent read errors regarding to ".DS_Store")
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", "pgm")

for filename in sorted(os.listdir(folder)):
    if not filename.lower().endswith(valid_extensions):
        continue  # Skip non-image
    full_path = os.path.join(folder, filename)
    # Read image
    frame = cv2.imread(full_path)
    all_images.append(frame)
    count += 1

if count == 0:
    print("No images found")
    exit()

base_image = all_images[0].copy()
base_image_bgr = cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR)
(x, y, w, h) = cv2.selectROI("ROI", base_image_bgr, fromCenter=False)
cv2.destroyWindow("ROI")

if w == 0 or h == 0:
    print("Invalid ROI")
    exit()

print(f"Initial ROI x:{x} y:{y} w:{w} h:{h}")

current_roi = [x, y, w, h]
for i in range(1, count):
    last_frame = all_images[i-1]
    next_frame = all_images[i]

    current_roi = simple_tracker(current_roi, last_frame, next_frame,
                                 max_iterations=100, threshold=1e-3)

    (nx, ny, nw, nh) = current_roi
    print(f"Current ROI x:{nx} y:{ny} w:{nw} h:{nh}")

    next_frame_bgr = cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR)

    cv2.rectangle(next_frame_bgr, (nx, ny), (nx+nw, ny+nh), (0, 255, 0), 2)
    cv2.putText(next_frame_bgr, f"Frame: {i}", (10,30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                color=(0,255,0), thickness=2)

    cv2.imshow("Tracking", next_frame_bgr)
    # 60fps, frame interval = 16.67 ms
    key = cv2.waitKey(17)
    # esc to quit
    if key == 27:
        break

cv2.destroyAllWindows()