import cv2
import numpy as np
import glob
import os

# Click 4 points to define ROI polygon
points = []
max_points = 4

def mouse_callback(event, x, y, flags, param):
    """
    Sequentially click 4 points on the image to define the vertices of the polygon
    """
    global points
    img = param["img"]
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < max_points:
        points.append((x, y))
        # Draw clicked points
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        # Draw a line
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
        # Close the polygon
        if len(points) == max_points:
            cv2.line(img, points[-1], points[0], (0, 255, 0), 2)
        cv2.imshow("Select ROI", img)

# Ultility functions
def homography_transform(p, x, y):
    """
    Given 8 parameters p=[h1,h2,h3,h4,h5,h6,h7,h8], and the original pixel(x,y)
    calculate the transformed perspective coordinates (u,v), where:
        u = (h1*x + h2*y + h3) / (1 + h7*x + h8*y)
        v = (h4*x + h5*y + h6) / (1 + h7*x + h8*y)
    Return (u,v)
    """
    h1, h2, h3, h4, h5, h6, h7, h8 = p
    denom = 1.0 + h7*x + h8*y
    u = (h1*x + h2*y + h3) / denom
    v = (h4*x + h5*y + h6) / denom
    return (u, v)

def compute_jacobian(p, x, y):
    """
    Compute the Jacobian d(u,v)/d(h1..h8) for p=[h1..h8] using formula:
        u = (h1*x + h2*y + h3) / (1 + h7*x + h8*y)
        v = (h4*x + h5*y + h6) / (1 + h7*x + h8*y)
    where c1 = 1 + h7*x + h8*y (norm factor)
    """
    h1, h2, h3, h4, h5, h6, h7, h8 = p
    c1 = 1.0 + h7*x + h8*y

    # First compute current (u,v)
    u,v = homography_transform(p, x, y)

    J = np.zeros((2,8), dtype=np.float32)
    # row for du/dh
    J[0,0] = x / c1           # d u / d h1
    J[0,1] = y / c1           # d u / d h2
    J[0,2] = 1.0 / c1         # d u / d h3
    J[0,3] = 0.0              # d u / d h4
    J[0,4] = 0.0              # d u / d h5
    J[0,5] = 0.0              # d u / d h6
    J[0,6] = -(u * x) / c1    # d u / d h7
    J[0,7] = -(u * y) / c1    # d u / d h8

    # row for dv/dh
    J[1,0] = 0.0
    J[1,1] = 0.0
    J[1,2] = 0.0
    J[1,3] = x / c1
    J[1,4] = y / c1
    J[1,5] = 1.0 / c1
    J[1,6] = -(v * x) / c1
    J[1,7] = -(v * y) / c1

    return J

def warp_corners(p, corners):
    """
    Use the current p (8-parameter homography) to warp the corners (4 points) to the new coordinate system
    and return the result as an int32 point array
    """
    warped = []
    for (x, y) in corners:
        (u, v) = homography_transform(p, x, y)
        warped.append([u, v])
    return np.array(warped, dtype=np.int32)

# Core function
def highdof_tracker(img0, img1, roi, max_iterations, threshold,
                    mask, p_init):
    """
    Homography LK iteration:
        delta p = (J^T J)^-1 (J^T [T(x) - I(W(x;p))]),
    where J=[nabla I(W(x;p))*dW/dp]. (Gauss-Newton approximation)
    Parameters:
      img0         : Template frame (grayscale)
      img1         : Current frame (grayscale)
      roi          : 4-point coordinates of the polygon in the template frame
      max_iterations : Maximum number of iterations
      threshold      : Threshold for convergence of delta p
      mask         : Binary mask with the same size as img0, polygon region equals 1
      p_init       : Initial 8 parameters [h1..h8]

    Returns:
      p_updated    : 8-parameter homography after iteration
    """
    p = p_init.copy()

    # Find the minimum bounding rectangle for ROI
    (minx, miny, w, h) = cv2.boundingRect(roi.astype(np.int32))

    # Extract template patch and mask patch
    template_patch = img0[miny:miny+h, minx:minx+w]
    mask_patch = mask[miny:miny+h, minx:minx+w]

    # Gradient of the current frame
    gradx = cv2.Sobel(img1, cv2.CV_32F, 1, 0)
    grady = cv2.Sobel(img1, cv2.CV_32F, 0, 1)

    for it in range(max_iterations):
        # Hessian, b
        H_acc = np.zeros((8,8), dtype=np.float32)
        b_acc = np.zeros((8,), dtype=np.float32)

        for py in range(h):
            for px in range(w):
                if mask_patch[py, px] == 0:
                    continue

                T_val = template_patch[py, px]
                X = px + minx
                Y = py + miny

                (u, v) = homography_transform(p, X, Y)
                if (u<0 or v<0 or
                    u>=(img1.shape[1]-1) or v>=(img1.shape[0]-1)):
                    continue

                uf, vf = int(u), int(v)
                du, dv = u - uf, v - vf

                # Bilinear interpolation to obtain I_val
                I00 = img1[vf,   uf]
                I01 = img1[vf,   uf+1]
                I10 = img1[vf+1, uf]
                I11 = img1[vf+1, uf+1]
                I_val = (1-du)*(1-dv)*I00 + du*(1-dv)*I01 + \
                        (1-du)*dv*I10 + du*dv*I11

                error = T_val - I_val

                # Interpolate gradients
                gx00 = gradx[vf,   uf]
                gx01 = gradx[vf,   uf+1]
                gx10 = gradx[vf+1, uf]
                gx11 = gradx[vf+1, uf+1]
                gx_val = (1-du)*(1-dv)*gx00 + du*(1-dv)*gx01 + \
                         (1-du)*dv*gx10 + du*dv*gx11

                gy00 = grady[vf,   uf]
                gy01 = grady[vf,   uf+1]
                gy10 = grady[vf+1, uf]
                gy11 = grady[vf+1, uf+1]
                gy_val = (1-du)*(1-dv)*gy00 + du*(1-dv)*gy01 + \
                         (1-du)*dv*gy10 + du*dv*gy11

                grad_vec = np.array([gx_val, gy_val], dtype=np.float32)
                J_2x8 = compute_jacobian(p, X, Y)
                SD = J_2x8.T @ grad_vec

                H_acc += np.outer(SD, SD)
                b_acc += (SD * error)

        # Solve using QR decomposition to compute delta p
        Q, R = np.linalg.qr(H_acc)
        dp = np.linalg.solve(R, Q.T @ b_acc)

        # Update p
        p += dp

        dp_norm = np.linalg.norm(dp)
        print(f"Iter={it}, dp_norm={dp_norm}")
        if dp_norm < threshold:
            break

    return p


# Version_2.0: read image sequences from folder
def main():
    # Read from folder
    image_folder = "nl_cereal_s3"
    # Assume named after frame00001.jpg, frame00002.jpg...
    image_paths = sorted(glob.glob(os.path.join(image_folder, "frame*.jpg")))
    if not image_paths:
        print("Error: No images found in folder:", image_folder)
        return

    # Read the initial frame as template
    first_frame_path = image_paths[0]
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print("Error: Unable to read first frame image:", first_frame_path)
        return

    # Parameters for video output
    frame_height, frame_width = frame.shape[:2]
    output_path = "media/homography_tracking_demo.mp4"
    output_fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    # Allow to click 4 points on the first frame
    clone_for_roi = frame.copy()
    param_dict = {"img": clone_for_roi}
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select ROI", mouse_callback, param=param_dict)
    cv2.imshow("Select ROI", clone_for_roi)

    print("Please click 4 corner points in the pop-up window to define the ROI polygon (ESC to cancel).")
    while True:
        if len(points) == max_points:
            print("ROI selection complete.")
            break
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # ESC
            print("Canceled.")
            out.release()
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Select ROI")

    if len(points) != max_points:
        print("Invalid ROI (need exactly 4 points).")
        out.release()
        return

    # Convert to array for further processing
    roi_polygon = np.array(points, dtype=np.float32)  # shape=(4,2)

    # Convert to grayscale (as the template image)
    template_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a mask of the same size as template_gray, and mark the polygon region as 1
    mask_all = np.zeros_like(template_gray, dtype=np.uint8)
    cv2.fillPoly(mask_all, [roi_polygon.astype(np.int32)], color=1)

    # Initial p=[1,0,0, 0,1,0, 0,0] as identity transform
    p_init = np.array([1,0,0, 0,1,0, 0,0], dtype=np.float32)
    current_p = p_init.copy()

    # Open result txt file and output tracking results
    txt_filename = "tracking_results_sbu1.txt"
    f_out = open(txt_filename, "w")

    # Start processing all frames
    for i, img_path in enumerate(image_paths):
        # Read current frame
        frame_new = cv2.imread(img_path)
        if frame_new is None:
            print(f"Warning: failed to read image {img_path}, skip.")
            continue

        gray_new = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

        if i == 0:
            # For the first frame, do not perform LK iteration (as it is the template itself), directly visualize
            # current_p is already initial value [1,0,0, 0,1,0, 0,0]
            pass
        else:
            # Track from the template to the current frame
            current_p = highdof_tracker(
                img0=template_gray,
                img1=gray_new,
                roi=roi_polygon,
                max_iterations=200,
                threshold=1e-4,
                mask=mask_all,
                p_init=current_p
            )

        # 4 points of the polygon after warping
        warped_poly = warp_corners(current_p, roi_polygon)

        # Draw locator on the current frame
        cv2.polylines(frame_new, [warped_poly], isClosed=True,
                      color=(0,255,0), thickness=2)
        cv2.putText(frame_new, "Homography LK Tracking", (10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                    color=(0, 255, 0), thickness=2)

        # Write into video
        out.write(frame_new)
        cv2.imshow("Tracking", frame_new)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            print("User interrupted with ESC.")
            break

        # Write 4 corners into text file
        # Assume click sequence: UL, UR, LR, LL
        # warped_poly stored with same order
        ulx, uly = warped_poly[0]
        urx, ury = warped_poly[1]
        lrx, lry = warped_poly[2]
        llx, lly = warped_poly[3]

        # In specific format
        # frame ulx uly urx ury lrx lry llx lly
        # frame00001.jpg 32.00 313.00 202.00 308.00 316.00 ...
        basename = os.path.basename(img_path)
        line_out = f"{basename} {ulx:.2f} {uly:.2f} {urx:.2f} {ury:.2f} {lrx:.2f} {lry:.2f} {llx:.2f} {lly:.2f}\n"
        f_out.write(line_out)

    # Finalization
    f_out.close()
    out.release()
    cv2.destroyAllWindows()
    print("Tracking finished.")
    print("Video saved to:", output_path)
    print("Tracking results saved to:", txt_filename)

if __name__ == "__main__":
    main()
