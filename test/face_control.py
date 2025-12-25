import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# ===================== 核心配置（可根据需求调整） =====================
# 需求0：人脸检测框占比阈值（人脸宽/高占图片宽/高的最小比例）
FACE_SIZE_THRESHOLD = 0  # 30%，特写阈值
# 需求1：头部姿态欧拉角阈值（单位：度）
YAW_THRESHOLD = 45    # 偏航角（左右转头）：±45°以内
PITCH_THRESHOLD = 30  # 俯仰角（上下抬头）：±30°以内
ROLL_THRESHOLD = 15   # 滚转角（歪头）：±15°以内
# 需求2：关键点完整性阈值（核心区域：左眼/右眼/鼻尖/左嘴/右嘴）
KEYPOINT_CONF_THRESHOLD = 0.8  # 关键点置信度≥0.8视为未遮挡
# 新增：人脸数量限制（仅允许单人脸）
ALLOWED_FACE_COUNT = 1

# ===================== 初始化InsightFace =====================
# 加载默认模型（buffalo_l），自动下载/使用已下载的模型
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # 先跑CPU，GPU需改CUDA
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1=CPU，det_size=检测分辨率

# ===================== 核心筛选函数 =====================
def is_high_quality_face(face, img_w, img_h):
    """
    判断单个人脸是否为高质量（满足所有筛选条件）
    :param face: InsightFace检测返回的单个人脸对象
    :param img_w: 图片宽度
    :param img_h: 图片高度
    :return: bool（是否高质量）, 筛选失败原因
    """
    # ------------ 需求0：检测框占比筛选 ------------
    x1, y1, x2, y2 = face.bbox.astype(int)
    face_w = x2 - x1
    face_h = y2 - y1
    face_w_ratio = face_w / img_w
    face_h_ratio = face_h / img_h
    if face_w_ratio < FACE_SIZE_THRESHOLD or face_h_ratio < FACE_SIZE_THRESHOLD:
        return False, f"人脸占比不足（宽：{face_w_ratio:.2f}, 高：{face_h_ratio:.2f}）"

    # ------------ 需求1：头部姿态欧拉角筛选 ------------
    # face.pose = (pitch, yaw, roll)，单位：度
    pitch, yaw, roll = face.pose
    if abs(yaw) > YAW_THRESHOLD:
        return False, f"偏航角超标（{yaw:.1f}° > ±{YAW_THRESHOLD}°）"
    if abs(pitch) > PITCH_THRESHOLD:
        return False, f"俯仰角超标（{pitch:.1f}° > ±{PITCH_THRESHOLD}°）"
    if abs(roll) > ROLL_THRESHOLD:
        return False, f"滚转角超标（{roll:.1f}° > ±{ROLL_THRESHOLD}°）"

    # ------------ 需求2：关键点完整性筛选（遮挡判断） ------------
    # InsightFace默认返回5个核心关键点：右眼/左眼/鼻尖/右嘴/左嘴
    # face.kps = 关键点坐标数组 (5,2)，face.kps_conf = 关键点置信度数组 (5,)
    '''core_key_points = ["右眼", "左眼", "鼻尖", "右嘴", "左嘴"]
    for idx, (kp_name, conf) in enumerate(zip(core_key_points, face.kps_conf)):
        if conf < KEYPOINT_CONF_THRESHOLD:
            return False, f"{kp_name}遮挡（置信度：{conf:.2f} < {KEYPOINT_CONF_THRESHOLD}）"
    '''
    # 所有条件满足
    return True, "高质量人脸"

# ===================== 测试单张图片 =====================
if __name__ == "__main__":
    # 替换为你的测试图片路径
    test_img_path = "./sample/halloween-9900545_960_720_1.jpg"
    img = cv2.imread(test_img_path)
    if img is None:
        print(f"错误：无法读取图片 {test_img_path}")
        exit(1)
    img_h, img_w = img.shape[:2]

    # 1. 检测人脸（InsightFace核心调用）
    faces = app.get(img)
    print(f"检测到 {len(faces)} 张人脸")

    # 新增：第一步先校验人脸数量
    if len(faces) != ALLOWED_FACE_COUNT:
        print(f"❌ 不合格 - 人脸数量为{len(faces)}，仅允许{ALLOWED_FACE_COUNT}张人脸")
        exit(0)  # 直接退出，无需后续筛选

    # 2. 筛选高质量人脸（此时faces只有1张）
    high_quality_faces = []
    for i, face in enumerate(faces):
        is_quality, reason = is_high_quality_face(face, img_w, img_h)
        print(f"人脸{i+1}：{'✅ 合格' if is_quality else '❌ 不合格'} - {reason}")
        if is_quality:
            high_quality_faces.append(face)

    # 3. 绘制结果（标注高质量人脸框+关键点）
    '''
    if high_quality_faces:
        # 只绘制高质量人脸
        result_img = app.draw_on(img.copy(), high_quality_faces)
        cv2.imwrite("halloween-9900545_960_720_1.jpg", result_img)
        print(f"\n✅ 筛选出 {len(high_quality_faces)} 张高质量人脸，结果保存为 high_quality_result.jpg")
    else:
        print("\n❌ 未筛选出任何高质量人脸")'''
