import cv2
import numpy as np
import os
from tqdm import tqdm

def locate_watermark(video_path):
    """
    交互式定位水印位置
    
    使用方法：
    1. 鼠标左键点击并拖动来选择水印区域
    2. 按 Enter 键确认选择
    3. 按 ESC 键取消选择
    
    返回：
    tuple: (x, y, width, height) 或 None（如果取消选择）
    """
    # 检查视频是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    
    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        raise Exception("无法读取视频帧")
    
    # 获取屏幕分辨率
    screen_width = 1280  # 默认屏幕宽度
    screen_height = 720  # 默认屏幕高度
    
    # 计算缩放比例
    scale = min((screen_width * 0.8) / frame.shape[1], 
                (screen_height * 0.8) / frame.shape[0])
    
    # 如果图片小于屏幕，则不需要缩放
    if scale >= 1:
        scale = 1
    
    # 计算缩放后的尺寸
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    
    # 缩放原始帧
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    # 存储选择的矩形区域
    selection = None
    drawing = False
    start_x, start_y = -1, -1
    
    # 创建窗口
    window_name = "选择水印区域"
    cv2.namedWindow(window_name)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selection, drawing, start_x, start_y, frame
        
        # 保存原始帧的副本
        img = frame.copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # 绘制临时矩形
                cv2.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow(window_name, img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # 确保宽度和高度为正值
            width = abs(x - start_x)
            height = abs(y - start_y)
            x_pos = min(start_x, x)
            y_pos = min(start_y, y)
            
            # 将坐标转换回原始尺寸
            x_pos = int(x_pos / scale)
            y_pos = int(y_pos / scale)
            width = int(width / scale)
            height = int(height / scale)
            
            selection = (x_pos, y_pos, width, height)
            # 绘制最终矩形
            cv2.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, img)
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # 显示第一帧
    cv2.imshow(window_name, frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter 键
            break
        elif key == 27:  # ESC 键
            selection = None
            break
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()
    
    if selection:
        print(f"已选择水印区域: x={selection[0]}, y={selection[1]}, width={selection[2]}, height={selection[3]}")
    
    return selection

def remove_watermark(input_video_path, output_video_path, x, y, width, height):
    """
    去除视频中指定区域的水印，优化处理速度
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"找不到输入视频文件: {input_video_path}")
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    cap = cv2.VideoCapture(input_video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("水印区域超出视频范围！")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 减小padding以加快处理速度
    padding = 15
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(frame_width, x + width + padding)
    y_end = min(frame_height, y + height + padding)
    
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # 获取ROI区域
                roi = frame[y_start:y_end, x_start:x_end].copy()
                
                # 创建核心掩码
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, 
                            (padding, padding),
                            (roi.shape[1]-padding, roi.shape[0]-padding),
                            255, -1)
                
                # 创建平滑过渡区域
                mask = cv2.GaussianBlur(mask, (padding*2+1, padding*2+1), padding/3)
                
                # 使用两次inpaint处理，平衡效果和速度
                filled_roi = cv2.inpaint(roi, mask, 7, cv2.INPAINT_NS)
                filled_roi = cv2.inpaint(filled_roi, mask, 3, cv2.INPAINT_TELEA)
                
                # 对填充区域进行快速模糊
                blurred = cv2.GaussianBlur(filled_roi, (5, 5), 0)
                
                # 创建三通道掩码
                mask_3d = np.stack([mask/255.0]*3, axis=2)
                
                # 混合结果
                result_roi = (blurred * mask_3d + roi * (1 - mask_3d)).astype(np.uint8)
                
                # 将处理后的区域放回原帧
                frame[y_start:y_end, x_start:x_end] = result_roi
                
                out.write(frame)
                
            except Exception as e:
                print(f"处理帧时出错: {str(e)}")
                out.write(frame)
                
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print(f"视频处理完成，已保存至: {output_video_path}")

def show_video_info(video_path):
    """
    显示视频的基本信息
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频基本属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # 获取文件大小
    file_size = os.path.getsize(video_path)
    file_size_mb = file_size / (1024 * 1024)  # 转换为MB
    
    print("\n视频信息:")
    print(f"文件路径: {video_path}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps} fps")
    print(f"总帧数: {frame_count}")
    print(f"时长: {int(duration//60)}分{int(duration%60)}秒")
    print(f"文件大小: {file_size_mb:.2f} MB")
    
    cap.release()
    return width, height, fps, frame_count

def main():
    try:
        # 使用示例
        video_name = "20241118-dahai"
        input_video = f"video/{video_name}.mp4"
        output_video = f"output/{video_name}_no_watermark.mp4"
        
        # 显示视频信息
        show_video_info(input_video)
        
        # 定位水印
        print("\n请在视频第一帧中选择水印区域...")
        watermark_coords = locate_watermark(input_video)
        
        if watermark_coords:
            x, y, width, height = watermark_coords
            remove_watermark(input_video, output_video, x, y, width, height)
        else:
            print("已取消水印去除操作")
        
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    main() 