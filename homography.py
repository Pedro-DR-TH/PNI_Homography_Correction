import cv2
import numpy as np

cap = cv2.VideoCapture('/Users/pedro_rodrigues1/Documents/PNI_Lab/ref.MP4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
ret, frame = cap.read()
cv2.imwrite('reference_frame.png', frame)
cap.release()

points = []
img = cv2.imread('reference_frame.png')
display_img = img.copy()

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(display_img, str(len(points)), (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('frame', display_img)
        print(f"Point {len(points)}: ({x}, {y})")

cv2.imshow('frame', display_img)
cv2.setMouseCallback('frame', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(points)

src_pts = np.array(points, dtype=np.float32)

size = 800
margin = 50
center = size / 2
radius = center - margin

n_points = len(src_pts)
angles = np.linspace(0, 2*np.pi, n_points, endpoint=False) - np.pi/2

dst_pts = np.array([
    [center + radius*np.cos(a), center + radius*np.sin(a)]
    for a in angles
], dtype=np.float32)

H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

warped = cv2.warpPerspective(img, H, (size, size))
cv2.imwrite('warped_check.png', warped)

cap = cv2.VideoCapture('/Users/pedro_rodrigues1/Documents/PNI_Lab/ref.MP4')
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('maze_topdown.mp4', fourcc, fps, (size, size))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    warped = cv2.warpPerspective(frame, H, (size, size))
    out.write(warped)

cap.release()
out.release()
