import math
import numpy as np

class DataFormatter:
    """
    负责将交互数据格式化为标准化的JSON格式用于RL/VLM训练。
    包含坐标归一化、衍生参数计算等逻辑。
    """
    
    def __init__(self, screen_width, screen_height):
        self.w = screen_width
        self.h = screen_height
        self.NORM_SIZE = 1000.0

    def _normalize(self, x, y):
        """将屏幕坐标转换为[0, 1000]的归一化坐标"""
        nx = int(round(x / self.w * self.NORM_SIZE))
        ny = int(round(y / self.h * self.NORM_SIZE))
        # 确保不越界
        return [max(0, min(int(self.NORM_SIZE), nx)), max(0, min(int(self.NORM_SIZE), ny))]

    def _normalize_bbox(self, bbox):
        """将 bbox [x1, y1, x2, y2] 归一化"""
        # bbox 此时应该是实际像素坐标
        p1 = self._normalize(bbox[0], bbox[1])
        p2 = self._normalize(bbox[2], bbox[3])
        return p1 + p2 # [x1, y1, x2, y2]

    def _calc_distance(self, start_norm, end_norm):
        """计算归一化后的欧氏距离"""
        dx = end_norm[0] - start_norm[0]
        dy = end_norm[1] - start_norm[1]
        return int(math.sqrt(dx**2 + dy**2))

    def _calc_direction_vector(self, start_norm, end_norm, action_type="swipe"):
        """
        计算方向。
        - 如果是 swipe: 返回 "up"/"down"/"left"/"right"
        - 如果是 drag: 返回 弧度 (0 - 2π)
        """
        dx = end_norm[0] - start_norm[0]
        dy = end_norm[1] - start_norm[1]
        
        if action_type == "drag":
            # 计算弧度，范围 [0, 2π]
            angle = math.atan2(dy, dx) # 返回值 -π 到 +π
            if angle < 0:
                angle += 2 * math.pi
            return round(angle, 4)
        
        else: # swipe
            if abs(dx) > abs(dy):
                return "right" if dx > 0 else "left"
            else:
                return "down" if dy > 0 else "up"

    def _calc_speed(self, distance, duration_ms):
        """简单的速度分类"""
        # 速度 = 归一化距离 / 时间(秒)
        # 这里的阈值需要根据实际经验调整
        velocity = distance / (duration_ms / 1000.0)
        return "fast" if velocity > 1000 else "slow" 
        # 例如：1秒滑过全屏(1000)算界限

    def format_tap(self, x, y, bbox=None, description=""):
        """生成点击操作的JSON"""
        return {
            "action": "tap",
            "position": self._normalize(x, y),
            "bbox": self._normalize_bbox(bbox) if bbox else None,
            "intent": f"点击{description}" if description else "点击指定位置",
            "timestamp": None  # 占位，由调用者填充
        }

    def format_swipe(self, start_x, start_y, end_x, end_y, duration_ms, bbox=None, description="", action_type="swipe"):
        """生成滑动/拖拽操作的JSON"""
        norm_start = self._normalize(start_x, start_y)
        norm_end = self._normalize(end_x, end_y)
        distance = self._calc_distance(norm_start, norm_end)
        
        data = {
            "action": action_type, # swipe 或 drag
            "start": norm_start,
            "end": norm_end,
            "duration": duration_ms,
            
            # 衍生参数
            "direction": self._calc_direction_vector(norm_start, norm_end, action_type),
            "distance": distance,
            "speed": self._calc_speed(distance, duration_ms),
            
            # 可选参数
            "bbox": self._normalize_bbox(bbox) if bbox else None,
            "intent": description, # 初始意图，后续可由VLM修正
            "success": False # 默认为False，由后续逻辑更新
        }
        return data
    
def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return str(obj)
