import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
from pathlib import Path
import time
import subprocess

# ===================== 核心配置（可根据需求调整） =====================
# 人脸筛选配置
FACE_SIZE_THRESHOLD = 0    # 人脸占比阈值
YAW_THRESHOLD = 45           # 偏航角阈值（±°）
PITCH_THRESHOLD = 30         # 俯仰角阈值（±°）
ROLL_THRESHOLD = 15          # 滚转角阈值（±°）
ALLOWED_FACE_COUNT = 1       # 仅允许单人脸
DET_SCORE_THRESHOLD = 0.8    # 检测置信度阈值
# 视频处理配置
FRAME_SKIP = 9               # 帧跳过数（0=逐帧检测）
MIN_VALID_DURATION = 2       # 最小合格片段时长（秒，替代原MIN_VALID_FRAMES）
VIDEO_FPS = 0                # 0=使用原视频FPS
SPEED_PRINT_INTERVAL = 50    # 速度打印间隔（帧）
TOLERANCE_FRAMES = 3         # 最大连续不合格帧数（容错阈值）

# ===================== 初始化InsightFace =====================
app = FaceAnalysis(providers=['CUDAExecutionProvider'])  # 'CUDAExecutionProvider', 'CPUExecutionProvider'
app.prepare(ctx_id=-1, det_size=(640, 640))

# ===================== 核心筛选函数 =====================
def is_high_quality_face(face, img_w, img_h):
    """判断单个人脸是否为高质量（满足占比+姿态条件）"""
    # ------------ 需求0：检测框占比筛选 ------------
    x1, y1, x2, y2 = face.bbox.astype(int)
    face_w = x2 - x1
    face_h = y2 - y1
    face_w_ratio = face_w / img_w
    face_h_ratio = face_h / img_h
    if face_w_ratio < FACE_SIZE_THRESHOLD or face_h_ratio < FACE_SIZE_THRESHOLD:
        return False, f"人脸占比不足（宽：{face_w_ratio:.2f}, 高：{face_h_ratio:.2f}）"

    # ------------ 需求1：头部姿态欧拉角筛选 ------------
    pitch, yaw, roll = face.pose
    if abs(yaw) > YAW_THRESHOLD:
        return False, f"偏航角超标（{yaw:.1f}° > ±{YAW_THRESHOLD}°）"
    if abs(pitch) > PITCH_THRESHOLD:
        return False, f"俯仰角超标（{pitch:.1f}° > ±{PITCH_THRESHOLD}°）"
    if abs(roll) > ROLL_THRESHOLD:
        return False, f"滚转角超标（{roll:.1f}° > ±{ROLL_THRESHOLD}°）"

    # 所有条件满足
    return True, "高质量人脸"

def get_frame_timestamp(frame_idx, fps):
    """将帧索引转换为时间戳（秒）"""
    return frame_idx / fps

def cut_video_by_timestamp(input_path, output_path, start_ts, end_ts):
    """
    使用ffmpeg裁剪视频（保留音频）
    :param input_path: 原视频路径
    :param output_path: 输出路径
    :param start_ts: 起始时间戳（秒）
    :param end_ts: 结束时间戳（秒）
    """
    duration = end_ts - start_ts
    if duration < MIN_VALID_DURATION:
        print(f"⚠️  片段时长{duration:.2f}秒 < 最小阈值{MIN_VALID_DURATION}秒，跳过保存")
        return False
    
    # FFmpeg命令（静音模式，覆盖输出）
    cmd = [
        "ffmpeg",
        "-ss", str(start_ts),       # 起始时间
        "-i", input_path,           # 输入文件
        "-to", str(end_ts),         # 结束时间
        "-c:v", "copy",             # 视频流直接复制（无重新编码）
        "-c:a", "copy",             # 音频流直接复制
        "-y",                       # 覆盖输出文件
        "-loglevel", "error",       # 仅输出错误信息
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ 保存片段：{output_path}（时长：{duration:.2f}秒）")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 裁剪失败：{output_path}，错误：{e}")
        return False

def process_video(video_path, output_dir="."):
    """处理视频（保留音频+容错帧）"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem  # 视频basename
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频基础信息
    fps = cap.get(cv2.CAP_PROP_FPS) if VIDEO_FPS == 0 else VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    if total_frames == 0:
        print(f"错误：无法读取视频 {video_path}")
        return
    
    # 初始化变量
    clip_num = 0                # 片段编号
    frame_idx = 0               # 当前帧索引
    processed_frames = 0        # 已处理帧数
    start_time = time.time()    # 处理开始时间
    valid_clip_start_ts = None  # 合格片段起始时间戳
    consecutive_invalid = 0     # 连续不合格帧数
    
    print(f"📽️  开始处理：{video_path}")
    print(f"📊 视频信息：FPS={fps:.2f}, 分辨率={width}x{height}, 总帧数={total_frames}, 总时长={total_duration:.2f}秒")
    print(f"⚙️  配置：置信度={DET_SCORE_THRESHOLD}, 容错帧数={TOLERANCE_FRAMES}, 最小片段时长={MIN_VALID_DURATION}秒")


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 帧跳过处理（提升速度）
        if frame_idx % (FRAME_SKIP + 1) != 0:
            frame_idx += 1
            continue
        
        # 1. 检测人脸
        faces = app.get(frame)
        faces = [f for f in faces if f.det_score >= DET_SCORE_THRESHOLD] # 过滤低置信度人脸
        is_frame_valid = False
        reason = ""
        
        if len(faces) != ALLOWED_FACE_COUNT:
            reason = f"人脸数量={len(faces)}（仅允许{ALLOWED_FACE_COUNT}张）"
        else:
            is_quality, reason = is_high_quality_face(faces[0], width, height)
            if is_quality:
                is_frame_valid = True
        
        if is_frame_valid:
            # 有效帧：重置连续不合格计数
            consecutive_invalid = 0
            # 启动新片段（若未开始）
            if valid_clip_start_ts is None:
                valid_clip_start_ts = get_frame_timestamp(frame_idx, fps)
                print(f"🔄 开始合格片段：帧{frame_idx}（时间戳={valid_clip_start_ts:.2f}秒）")
        else:
            # 无效帧：累计连续不合格计数（所有场景都累计）
            consecutive_invalid += 1
            # 超过容错阈值且有正在进行的片段 → 结束并裁剪
            if valid_clip_start_ts is not None and consecutive_invalid > TOLERANCE_FRAMES:
                # 计算片段结束时间戳（容错帧的前一帧，避免包含不合格帧）
                end_ts = get_frame_timestamp(frame_idx - consecutive_invalid, fps)
                output_path = os.path.join(output_dir, f"{video_name}_croped{clip_num}.mp4")
                # 调用裁剪函数
                if cut_video_by_timestamp(video_path, output_path, valid_clip_start_ts, end_ts):
                    clip_num += 1
                # 重置片段状态
                valid_clip_start_ts = None
                consecutive_invalid = 0
                print(f"🔚 结束合格片段：帧{frame_idx}（时间戳={get_frame_timestamp(frame_idx, fps):.2f}秒），原因：{reason}")
        
        # 3. 打印帧信息
        status = "✅" if is_frame_valid else "❌"
        print(f"帧{frame_idx} {status} - {reason}")
        
        # 4. 速度统计
        processed_frames += 1
        if processed_frames % SPEED_PRINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            speed = processed_frames / elapsed
            print(f"📈 已处理{processed_frames}帧，速度：{speed:.2f}帧/秒")
        
        frame_idx += 1

    # 处理最后一段合格片段
    if valid_clip_start_ts is not None:
        output_path = os.path.join(output_dir, f"{video_name}_croped{clip_num}.mp4")
        cut_video_by_timestamp(video_path, output_path, valid_clip_start_ts, total_duration)
        clip_num += 1

    # 收尾统计
    total_elapsed = time.time() - start_time
    avg_speed = processed_frames / total_elapsed if total_elapsed > 0 else 0
    print(f"\n🏁 处理完成！")
    print(f"⏱️  总耗时：{total_elapsed:.2f}秒，平均速度：{avg_speed:.2f}帧/秒")
    print(f"📦 生成合格片段数：{clip_num}（保存路径：{os.path.abspath(output_dir)}）")

    cap.release()

# ===================== 主函数 =====================
if __name__ == "__main__":
    test_video_path = "/home/byd/PythonProjects/ky/MyAudio2Face/insightface/examples/test/sample/24494339-1-192.mp4"
    output_directory = "./output/24494339-1-192_valid_video_clips_1"
    # 检查ffmpeg是否可用
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"❌ 未找到FFmpeg，请确保已安装并加入环境变量，或修改FFMPEG_PATH配置")
        exit(1)
    
    process_video(test_video_path, output_directory)
