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

# Apply Tracker to file
video_path = "example_videos/bird.mp4"
output_path = "media/2DoF_tracked_video.mp4"
output_fps = 60

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open the video file: {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

# Read the first frame and display for ROI selection
ret, frame = cap.read()
if not ret:
    print(f"Error: Unable to read the video file: {video_path}")
    cap.release()
    exit()

frame_bgr = frame.copy()
(x, y, w, h) = cv2.selectROI("Select ROI", frame_bgr, fromCenter=False)
cv2.destroyWindow("Select ROI")

if w == 0 or h == 0:
    print("Invalid ROI, please reselect.")
    cap.release()
    out.release()
    exit()

print(f"Initial ROI x:{x} y:{y} w:{w} h:{h}")

use_static_template = False  # Set to False to update template every frame

current_roi = [x, y, w, h]
static_template_frame = frame.copy()  # Store the initial template

while True:
    ret, next_frame = cap.read()
    if not ret:
        print("End of video.")
        break

    if use_static_template:
        template_frame = static_template_frame  # Use the first frame as template
    else:
        template_frame = frame  # Update template with the last frame

    # Apply tracker
    current_roi = simple_tracker(current_roi, template_frame, next_frame,
                                 max_iterations=50, threshold=1e-3)

    (nx, ny, nw, nh) = current_roi
    cv2.rectangle(next_frame, (nx, ny), (nx+nw, ny+nh), (0, 255, 0), 5)
    cv2.putText(next_frame, "Tracking", (10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                color=(0, 255, 0), thickness=2)

    cv2.imshow("Tracking", next_frame)

    # Write frame to output video
    out.write(next_frame)

    # Update the previous frame
    frame = next_frame.copy()

    # Exit on 'ESC' key press
    key = cv2.waitKey(17) & 0xFF
    if key == 27:  # ESC to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracked video saved to {output_path}")