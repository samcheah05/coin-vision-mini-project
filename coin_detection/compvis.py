import os
import sys

# Silence Python 3.13 loky / wmic core-count subprocess warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

OUTPUT_DIR = "output_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# TASK 1: IMAGE DATA ACQUISITION & FALLBACK GENERATOR
# =====================================================================
def load_or_create_coin_scene(image_path="coin_scene.jpg"):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            print(f"[INFO] Successfully loaded '{image_path}' from disk.")
            return cv2.resize(img, (640, 480))
            
    print(f"[INFO] '{image_path}' not found in current folder.")
    print(f"[INFO] Generating high-fidelity multi-coin test scene with boundary references.")
    img = np.ones((480, 640, 3), dtype=np.uint8) * 230
    
    # Reference boundary edges (sheet border for Hough line transform)
    cv2.rectangle(img, (40, 40), (600, 440), (160, 160, 160), 2)
    
    # Multi-coin distribution: (cx, cy), radius, color_bgr
    coins = [
        ((150, 150), 38, (45, 125, 195)),   # Class 0: Small Copper
        ((220, 150), 46, (35, 175, 215)),   # Class 1: Medium Brass (touching neighbor)
        ((350, 240), 55, (185, 185, 185)), # Class 2: Large Silver / Bimetallic
        ((480, 180), 36, (45, 125, 195)),   # Class 0: Small Copper
        ((230, 330), 45, (35, 175, 215)),   # Class 1: Medium Brass
        ((440, 340), 54, (185, 185, 185))  # Class 2: Large Silver
    ]
    for (cx, cy), r, col in coins:
        cv2.circle(img, (cx, cy), r, col, -1)
        cv2.circle(img, (cx, cy), r, (70, 70, 70), 2)
        cv2.circle(img, (cx, cy), int(r * 0.65), (110, 110, 110), 1)
        
    cv2.imwrite(os.path.join(OUTPUT_DIR, "task1_input_scene.png"), img)
    return img


# =====================================================================
# TASK 2: IMAGE RESTORATION (15 MARKS)
# =====================================================================
def add_gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

def add_salt_and_pepper(img, prob=0.03):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape[:2])
    noisy[rnd < (prob / 2)] = 0
    noisy[rnd > 1 - (prob / 2)] = 255
    return noisy

def add_periodic_noise(img, freq=25, amp=35):
    rows, cols = img.shape[:2]
    x = np.arange(cols)
    y = np.arange(rows)
    X, _ = np.meshgrid(x, y)
    sine = amp * np.sin(2 * np.pi * freq * X / cols)
    return np.clip(img.astype(np.float64) + sine, 0, 255).astype(np.uint8)

def calc_mse_psnr(orig, restored):
    mse = np.mean((orig.astype(np.float64) - restored.astype(np.float64)) ** 2)
    if mse == 0:
        return 0, float('inf')
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return mse, psnr

def notch_filter_periodic_noise(gray_noisy, freq_offset=25):
    dft = np.fft.fft2(gray_noisy)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

    rows, cols = gray_noisy.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol + freq_offset, crow), 5, 0, -1)
    cv2.circle(mask, (ccol - freq_offset, crow), 5, 0, -1)

    filtered_shift = dft_shift * mask
    f_ishift = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(f_ishift)
    restored = np.clip(np.abs(img_back), 0, 255).astype(np.uint8)
    return restored, magnitude_spectrum


# =====================================================================
# TASK 3: FILTERING & EDGE DETECTION (15 MARKS)
# =====================================================================
def run_task3_filtering_and_edges(gray_img):
    asym_kernel = np.array([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  3]], dtype=np.float32)
    corr = signal.correlate2d(gray_img, asym_kernel, mode='same', boundary='symm')
    conv = signal.convolve2d(gray_img, asym_kernel, mode='same', boundary='symm')

    smooth_3x3 = cv2.GaussianBlur(gray_img, (3, 3), 0)
    smooth_11x11 = cv2.GaussianBlur(gray_img, (11, 11), 0)

    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(gray_img, -1, sharp_kernel)

    sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.clip(cv2.magnitude(sx, sy), 0, 255).astype(np.uint8)

    canny = cv2.Canny(gray_img, 60, 160)

    hough_display = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            coords = line.flatten()
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                cv2.line(hough_display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    return corr, conv, smooth_3x3, smooth_11x11, sharpened, sobel, canny, hough_display


# =====================================================================
# TASK 4: SEGMENTATION & WATERSHED (15 MARKS)
# =====================================================================
def run_task4_segmentation(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, global_th = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    _, otsu_th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt_th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 3)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(otsu_th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.45 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    watershed_img = bgr_img.copy()
    markers = cv2.watershed(watershed_img, markers)
    watershed_img[markers == -1] = [0, 0, 255]

    num_coins = len(np.unique(markers)) - 2
    return global_th, otsu_th, adapt_th, watershed_img, markers, dist_transform, max(0, num_coins)


# =====================================================================
# TASK 5: MODEL-BASED VISION (15 MARKS)
# =====================================================================
def run_task5_classification(bgr_img, markers):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    features, labels = [], []
    unique_ids = np.unique(markers)

    for uid in unique_ids:
        if uid <= 1:
            continue
        mask = np.uint8(markers == uid)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 400:
            continue

        perimeter = cv2.arcLength(c, True)
        circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        mean_hsv = cv2.mean(hsv, mask=mask)
        mean_hue = mean_hsv[0]
        mean_sat = mean_hsv[1]

        features.append([area, perimeter, circ, mean_hue, mean_sat])

        if area < 5000:
            labels.append(0)  # Class 0: Small Copper
        elif area < 7500:
            labels.append(1)  # Class 1: Medium Brass
        else:
            labels.append(2)  # Class 2: Large Silver

    X = np.array(features)
    y = np.array(labels)

    knn = KNeighborsClassifier(n_neighbors=min(3, len(y)))
    knn.fit(X, y)
    preds = knn.predict(X)

    print("\n--- Task 5: KNN Classification Results ---")
    print(classification_report(y, preds, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
    return knn, X, y, preds


# =====================================================================
# TASK 6: STEREO VISION (10 MARKS)
# =====================================================================
# =====================================================================
# TASK 6: STEREO VISION (10 MARKS)
# =====================================================================
def run_task6_stereo_disparity(gray_img):
    # 1. Add subtle paper surface texture so stereo block matching has gradient features
    noise_texture = np.random.normal(0, 4, gray_img.shape).astype(np.float64)
    textured_base = np.clip(gray_img.astype(np.float64) + noise_texture, 0, 255).astype(np.uint8)
    
    rows, cols = gray_img.shape
    left_img = textured_base.copy()
    
    # 2. In stereo triangulation (Z = f*B/d), background has low disparity, coins have high disparity
    bg_shift = 4     # Table background disparity = 4 px
    coin_shift = 18  # Raised coins disparity = 18 px (closer to camera)
    
    # Shift background
    M_bg = np.float32([[1, 0, -bg_shift], [0, 1, 0]])
    right_img = cv2.warpAffine(textured_base, M_bg, (cols, rows), borderValue=230)
    
    # Extract coin regions from left image and shift them by 18 px
    _, coin_mask = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)
    M_coin = np.float32([[1, 0, -coin_shift], [0, 1, 0]])
    shifted_coins = cv2.warpAffine(textured_base, M_coin, (cols, rows), borderValue=230)
    shifted_mask = cv2.warpAffine(coin_mask, M_coin, (cols, rows), borderValue=0)
    
    # Composite shifted coins into right image
    right_img[shifted_mask > 0] = shifted_coins[shifted_mask > 0]

    # 3. Compute disparity map using StereoSGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=32,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=16
    )
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    disparity[disparity < 0] = 0
    return disparity


# =====================================================================
# MAIN PIPELINE EXECUTION & VISUALIZATION BUILDER
# =====================================================================
if __name__ == "__main__":
    print("Executing Computer Vision Assignment Pipeline...")
    
    # 1. Image Data (Task 1)
    img = load_or_create_coin_scene("coin_scene.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Image Restoration (Task 2)
    print("\n[Running Task 2: Image Restoration]")
    g_noisy = add_gaussian_noise(gray)
    sp_noisy = add_salt_and_pepper(gray)
    p_noisy = add_periodic_noise(gray)
    
    g_restored = cv2.GaussianBlur(g_noisy, (5, 5), 0)
    sp_restored = cv2.medianBlur(sp_noisy, 5)
    p_restored, mag_spec = notch_filter_periodic_noise(p_noisy)
    
    mse_g, psnr_g = calc_mse_psnr(gray, g_restored)
    mse_sp, psnr_sp = calc_mse_psnr(gray, sp_restored)
    mse_p, psnr_p = calc_mse_psnr(gray, p_restored)
    
    print(f"Gaussian Filter: MSE={mse_g:.2f}, PSNR={psnr_g:.2f} dB")
    print(f"Median Filter (S&P): MSE={mse_sp:.2f}, PSNR={psnr_sp:.2f} dB")
    print(f"Notch Filter (Periodic): MSE={mse_p:.2f}, PSNR={psnr_p:.2f} dB")

    # Plot & Save Task 2
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    axes2[0, 0].imshow(g_noisy, cmap='gray'); axes2[0, 0].set_title("Gaussian Noise")
    axes2[0, 1].imshow(sp_noisy, cmap='gray'); axes2[0, 1].set_title("Salt & Pepper Noise")
    axes2[0, 2].imshow(p_noisy, cmap='gray'); axes2[0, 2].set_title("Periodic Noise")
    axes2[1, 0].imshow(g_restored, cmap='gray'); axes2[1, 0].set_title(f"Gaussian Restored ({psnr_g:.1f} dB)")
    axes2[1, 1].imshow(sp_restored, cmap='gray'); axes2[1, 1].set_title(f"Median Restored ({psnr_sp:.1f} dB)")
    axes2[1, 2].imshow(p_restored, cmap='gray'); axes2[1, 2].set_title(f"Notch Restored ({psnr_p:.1f} dB)")
    for ax in axes2.ravel(): ax.axis('off')
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTPUT_DIR, "task2_restoration_comparison.png"))
    print(f"Saved: {OUTPUT_DIR}/task2_restoration_comparison.png")

    # 3. Filtering & Edge Detection (Task 3)
    print("\n[Running Task 3: Filtering & Edge Detection]")
    corr, conv, s3, s11, sharp, sobel, canny, hough = run_task3_filtering_and_edges(gray)
    
    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
    axes3[0, 0].imshow(s3, cmap='gray'); axes3[0, 0].set_title("Smoothing (3x3)")
    axes3[0, 1].imshow(s11, cmap='gray'); axes3[0, 1].set_title("Smoothing (11x11)")
    axes3[0, 2].imshow(sharp, cmap='gray'); axes3[0, 2].set_title("Sharpened (Laplacian)")
    axes3[0, 3].imshow(corr, cmap='gray'); axes3[0, 3].set_title("Correlation Result")
    axes3[1, 0].imshow(conv, cmap='gray'); axes3[1, 0].set_title("Convolution Result")
    axes3[1, 1].imshow(sobel, cmap='gray'); axes3[1, 1].set_title("Sobel Gradient")
    axes3[1, 2].imshow(canny, cmap='gray'); axes3[1, 2].set_title("Canny Edges")
    axes3[1, 3].imshow(cv2.cvtColor(hough, cv2.COLOR_BGR2RGB)); axes3[1, 3].set_title("Hough Lines Detected")
    for ax in axes3.ravel(): ax.axis('off')
    fig3.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "task3_filtering_and_edges.png"))
    print(f"Saved: {OUTPUT_DIR}/task3_filtering_and_edges.png")

    # 4. Image Segmentation (Task 4)
    print("\n[Running Task 4: Segmentation & Counting]")
    glob_th, otsu_th, adapt_th, w_img, markers, dist_map, coin_count = run_task4_segmentation(img)
    print(f"Total Segmented Coins Detected (Watershed): {coin_count}")
    
    fig4, axes4 = plt.subplots(2, 3, figsize=(14, 8))
    axes4[0, 0].imshow(glob_th, cmap='gray'); axes4[0, 0].set_title("Global Threshold")
    axes4[0, 1].imshow(otsu_th, cmap='gray'); axes4[0, 1].set_title("Otsu Threshold")
    axes4[0, 2].imshow(adapt_th, cmap='gray'); axes4[0, 2].set_title("Adaptive Gaussian")
    axes4[1, 0].imshow(dist_map, cmap='jet'); axes4[1, 0].set_title("Distance Transform")
    axes4[1, 1].imshow(cv2.cvtColor(w_img, cv2.COLOR_BGR2RGB)); axes4[1, 1].set_title("Watershed Result (Red Boundaries)")
    axes4[1, 2].imshow(markers, cmap='tab20'); axes4[1, 2].set_title(f"Segmented Labels (Count={coin_count})")
    for ax in axes4.ravel(): ax.axis('off')
    fig4.tight_layout()
    fig4.savefig(os.path.join(OUTPUT_DIR, "task4_segmentation_comparison.png"))
    print(f"Saved: {OUTPUT_DIR}/task4_segmentation_comparison.png")

    # 5. Model-Based Vision (Task 5)
    print("\n[Running Task 5: Model-Based Vision]")
    knn_model, X, y, preds = run_task5_classification(img, markers)
    
    # 6. Stereo Vision (Task 6)
    print("\n[Running Task 6: Stereo Vision]")
    disp = run_task6_stereo_disparity(gray)
    
    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 5))
    axes6[0].imshow(gray, cmap='gray'); axes6[0].set_title("Left Reference Image")
    axes6[1].imshow(disp, cmap='plasma'); axes6[1].set_title("Stereo Disparity Map (StereoSGBM)")
    for ax in axes6: ax.axis('off')
    fig6.tight_layout()
    fig6.savefig(os.path.join(OUTPUT_DIR, "task6_stereo_disparity.png"))
    print(f"Saved: {OUTPUT_DIR}/task6_stereo_disparity.png")

    print("\n" + "="*60)
    print(f"[SUCCESS] All tasks completed! Figures saved inside: '{OUTPUT_DIR}/'")
    print("="*60)
    
    # Display all figures on screen
    plt.show()