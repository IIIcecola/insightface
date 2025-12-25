import cv2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread("./sample/halloween-9900545_960_720_1.jpg")
faces = app.get(img)  # 检测并分析人脸
print(f"faces: {faces}")
# rimg = app.draw_on(img, faces)  # 绘制结果
# cv2.imwrite("waves-7458726_960_720_t0.jpg", rimg)
