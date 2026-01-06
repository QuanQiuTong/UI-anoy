import uiautomator2 as u2
import time
from pathlib import Path
from PIL import Image
import numpy as np
import math


class UIAutomatorController:
    """UI自动化控制器，封装uiautomator2操作和屏幕处理逻辑"""
    
    def __init__(self, device_serial=None, screenshot_dir="screenshots"):
        """初始化控制器，连接设备"""
        try:
            if device_serial:
                self.d = u2.connect(device_serial)
            else:
                self.d = u2.connect()  # 自动连接第一个设备
            print(f"已连接设备: {self.d.info}")
        except Exception as e:
            print(f"设备连接失败: {e}")
            print("请确保设备已通过USB连接并启用调试模式")
            raise
        
        # 截图目录管理
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(exist_ok=True)
    
    def get_window_size(self):
        """获取设备窗口大小"""
        return self.d.window_size()
    
    def take_screenshot(self, prefix="screen"):
        """截取屏幕并保存到文件"""
        try:
            timestamp = int(time.time() * 1000)
            filename = self.screenshot_dir / f"{prefix}_{timestamp}.png"
            
            image = self.d.screenshot()
            image.save(filename)
            print(f"截图已保存: {filename}")
            
            return {
                'image': image,
                'filename': str(filename),
                'timestamp': timestamp
            }
        except Exception as e:
            print(f"截图失败: {e}")
            return None
    
    def take_screenshot_with_path(self, filepath):
        """截取屏幕并保存到指定路径"""
        try:
            image = self.d.screenshot()
            image.save(filepath)
            print(f"截图已保存: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"截图失败: {e}")
            return None
    
    def click(self, x, y):
        """点击指定坐标"""
        print(f"点击坐标: ({x}, {y})")
        self.d.click(x, y)
    
    def swipe(self, start_x, start_y, end_x, end_y, duration=0.3):
        """滑动操作"""
        print(f"滑动从 ({start_x}, {start_y}) 到 ({end_x}, {end_y})，持续时间: {duration}s")
        self.d.swipe(start_x, start_y, end_x, end_y, duration=duration)
    
    def press_back(self):
        """按返回键"""
        self.d.press("back")
    
    def app_stop(self, package_name):
        """停止应用"""
        self.d.app_stop(package_name)
    
    def app_start(self, package_name):
        """启动应用"""
        self.d.app_start(package_name)
    
    def back(self, package_name):
        if self.get_current_package() != package_name:
            self.app_start(package_name)

    def reset_app_state(self, package_name, stop_wait=1, start_wait=3):
        """重置应用到初始状态"""
        print("重置应用到初始状态...")
        
        # 停止应用
        try:
            self.app_stop(package_name)
            time.sleep(stop_wait)
        except Exception as e:
            print(f"停止应用失败: {e}")
        
        # 启动应用
        try:
            self.app_start(package_name)
            time.sleep(start_wait)  # 等待应用启动
            print(f"应用 {package_name} 已启动")
            return True
        except Exception as e:
            print(f"启动应用失败: {e}")
            return False
    
    def calculate_image_diff(self, img1_path, img2_path, bbox=None, threshold=0.02):
        """计算两张图片的差异，支持指定区域检测"""
        try:
            img1 = Image.open(img1_path).convert('L')
            img2 = Image.open(img2_path).convert('L')
            
            # 如果指定了bbox，裁剪图片
            if bbox:
                # bbox格式: (x1, y1, x2, y2)
                img1 = img1.crop(bbox)
                img2 = img2.crop(bbox)
            
            # 确保两张图片尺寸一致
            if img1.size != img2.size:
                img2 = img2.resize(img1.size)
            
            # 计算差异
            diff = np.abs(np.array(img1).astype(float) - np.array(img2).astype(float))
            diff_ratio = np.mean(diff > 10)
            
            # 调试信息，可选
            if diff_ratio > threshold:
                print(f"检测到变化: 差异比例={diff_ratio:.4f}, 阈值={threshold}")
            else:
                print(f"未检测到变化: 差异比例={diff_ratio:.4f}, 阈值={threshold}")
            
            return diff_ratio > threshold
        except Exception as e:
            print(f"图片比较失败: {e}")
            return False
    
    def convert_coordinates(self, bbox, target_width=None, target_height=None):
        """转换坐标到实际屏幕位置"""
        if target_width is None or target_height is None:
            target_width, target_height = self.get_window_size()
        
        # 假设bbox是归一化坐标 [x1, y1, x2, y2]
        x1_norm, y1_norm, x2_norm, y2_norm = bbox
        
        # 转换到实际屏幕坐标
        x1 = x1_norm * target_width
        y1 = y1_norm * target_height
        x2 = x2_norm * target_width
        y2 = y2_norm * target_height
        
        return (x1, y1, x2, y2)
    
    def convert_normalized_coordinates(self, normalized_coords):
        """转换归一化坐标到实际屏幕坐标"""
        screen_width, screen_height = self.get_window_size()
        
        # 假设normalized_coords是归一化坐标列表
        real_coords = []
        for coord in normalized_coords:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                x_norm, y_norm = coord[0], coord[1]
                x_real = x_norm * screen_width
                y_real = y_norm * screen_height
                real_coords.append((x_real, y_real))
        
        return real_coords
    
    def get_device_info(self):
        """获取设备信息"""
        return self.d.info
    
    def screen_center(self):
        """获取屏幕中心坐标"""
        width, height = self.get_window_size()
        return width // 2, height // 2
    
    def long_click(self, x, y, duration=1.0):
        """长按指定坐标"""
        self.d.long_click(x, y, duration=duration)
    
    def drag(self, start_x, start_y, end_x, end_y, duration=0.5):
        """拖拽操作"""
        self.d.drag(start_x, start_y, end_x, end_y, duration=duration)
    
    def get_current_package(self):
        """获取当前活动应用的包名"""
        return self.d.app_current()['package']
    
    def get_ui_hierarchy(self):
        """获取当前UI层次结构"""
        return self.d.dump_hierarchy()
    
    def normalize_coordinates(self, x, y):
        """将屏幕坐标归一化到[0, 1000]"""
        screen_w, screen_h = self.get_window_size()
        norm_x = x / screen_w * 1000
        norm_y = y / screen_h * 1000
        return norm_x, norm_y

    def calculate_swipe_params(self, start_x, start_y, end_x, end_y):
        """计算滑动的衍生参数"""
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        return dx, dy, distance, angle