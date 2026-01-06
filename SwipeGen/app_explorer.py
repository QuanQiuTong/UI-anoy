import time
import os
import json
from pathlib import Path
from detect import ExplorationDetector
from device_controller import UIAutomatorController
from data_utils import DataFormatter, json_safe

class InteractionTester:
    def __init__(self, controller: UIAutomatorController, app_package: str):
        self.controller = controller
        self.app_package = app_package
        w, h = self.controller.get_window_size()
        self.formatter = DataFormatter(w, h)

    def _get_pixel_bbox(self, bbox_norm_1000):
        w, h = self.controller.get_window_size()
        return [
            bbox_norm_1000[0] / 1000 * w,
            bbox_norm_1000[1] / 1000 * h,
            bbox_norm_1000[2] / 1000 * w,
            bbox_norm_1000[3] / 1000 * h
        ]

    def run_click_test(self, region, name_prefix, auto_back=True):
        """
        执行点击测试
        :param auto_back: L1探索设为False(为了进L2)，L2探索设为True(点完即回)。
        """
        print(f"Testing Click: {name_prefix} | {region.get('description', '')}")
        
        before_res = self.controller.take_screenshot(f"{name_prefix}_before")
        if not before_res: return None
        
        # 计算坐标
        bbox_pixel = self._get_pixel_bbox(region['bbox'])
        x_center = (bbox_pixel[0] + bbox_pixel[2]) / 2
        y_center = (bbox_pixel[1] + bbox_pixel[3]) / 2
        
        # 准备数据
        action_data = self.formatter.format_tap(
            x_center, y_center, 
            bbox=bbox_pixel, 
            description=region.get('description', '')
        )
        
        # 执行点击
        try:
            self.controller.click(x_center, y_center)
            time.sleep(3) # 等待页面跳转
        except Exception as e:
            print(f"Click failed: {e}")
            return None

        after_res = self.controller.take_screenshot(f"{name_prefix}_after")
        
        # --- 全屏变动检测 ---
        # 传入 bbox=None 或不传，controller 会对比整张图片
        # 阈值可以适当调高一点点，防止时间跳变等微小干扰，这里保持默认或设为 0.02
        has_changed = False
        if after_res:
            # 这里的 bbox 传 None，表示检测全屏
            has_changed = self.controller.calculate_image_diff(
                before_res['filename'], 
                after_res['filename'], 
                bbox=None, 
                threshold=0.01  # 1% 的像素变动即认为发生变化
            )
        
        action_data['success'] = has_changed
        action_data['timestamp'] = time.time()

        # 只有在 auto_back=True 且确实发生了变化时，才执行返回
        if auto_back and has_changed:
            print("Auto-back triggered.")
            time.sleep(1)
            self.controller.back(self.app_package)

        return {
            'type': 'tap',
            'has_changed': has_changed,
            'action_data': action_data,
            'screenshot_before': before_res['filename'],
            'screenshot_after': after_res['filename'] if after_res else None,
            'region_info': region
        }

    def run_slide_test(self, region, name_prefix, auto_back=True):
        """
        执行滑动测试
        :param auto_back: 新增参数，逻辑同点击。L1探索设为False，L2探索设为True。
        """
        print(f"Testing Slide: {name_prefix} | {region.get('description', '')}")
        
        before_res = self.controller.take_screenshot(f"{name_prefix}_before")
        if not before_res: return None
        
        bbox_pixel = self._get_pixel_bbox(region['bbox'])
        w = bbox_pixel[2] - bbox_pixel[0]
        h = bbox_pixel[3] - bbox_pixel[1]
        direction = region.get('direction', '').lower()
        
        # 简单的滑动逻辑
        start_x, end_x = bbox_pixel[0] + w * 0.5, bbox_pixel[0] + w * 0.5
        start_y, end_y = bbox_pixel[1] + h * 0.9, bbox_pixel[1] + h * 0.1
        duration = 300

        if 'horiz' in direction:
            start_x, end_x = bbox_pixel[0] + w * 0.9, bbox_pixel[0] + w * 0.1
            start_y, end_y = bbox_pixel[1] + h * 0.5, bbox_pixel[1] + h * 0.5

        action_data = self.formatter.format_swipe(
            start_x, start_y, end_x, end_y, duration,
            bbox=bbox_pixel,
            description=region.get('description', ''),
            action_type="swipe"
        )

        try:
            self.controller.swipe(start_x, start_y, end_x, end_y, duration=duration/1000)
            time.sleep(1.5) # 滑动后等待时间稍长，可能有动画或加载
        except Exception as e:
            print(f"Swipe failed: {e}")
            return None

        after_res = self.controller.take_screenshot(f"{name_prefix}_after")
        
        # --- 全屏变动检测 ---
        has_changed = False
        if after_res:
            has_changed = self.controller.calculate_image_diff(
                before_res['filename'], 
                after_res['filename'], 
                bbox=None,
                threshold=5e-3  # 0.5% 的像素变动即认为发生变化
            )
            
        action_data['success'] = has_changed
        
        # 如果滑动导致了页面切换（例如全屏翻页），在 auto_back=True 时尝试返回
        # 注意：滑动的 Back 行为不一定能复原（比如 Feed 流下滑），但尝试 Back 是通用的回退策略
        if auto_back and has_changed:
            print("Auto-back triggered (Swipe).")
            time.sleep(1)
            self.controller.back(self.app_package)

        return {
            'type': 'swipe',
            'has_changed': has_changed,
            'action_data': action_data,
            'screenshot_before': before_res['filename'],
            'screenshot_after': after_res['filename'] if after_res else None,
            'region_info': region
        }


class AppExplorer:
    def __init__(self, device_serial=None, model_path="", app_package="", 
                 screenshot_dir="screenshots", logs_dir="logs"):
        self.controller = UIAutomatorController(device_serial, screenshot_dir)
        self.detector = ExplorationDetector(model_path)
        self.app_package = app_package
        self.tester = InteractionTester(self.controller, app_package)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)

    def _process_l2_exploration(self, parent_res, l1_index, prefix_type, max_interactions):
        """
        处理二级页面探索的通用逻辑
        
        :param parent_res: L1 操作的结果字典
        :param l1_index: L1 操作的序号
        :param prefix_type: 'Click' 或 'Slide'，用于日志前缀
        :param max_interactions: L2 最大操作数
        """
        # 初始化 L2 结果
        parent_res['l2_exploration'] = None

        if not parent_res['has_changed']:
            # print(f"  (页面未变化，跳过 L2)")
            return

        print(f"  >>> 进入 Level 2 ({prefix_type} #{l1_index} 触发)")
        
        # C. L2 分析
        l2_image_path = parent_res['screenshot_after']
        if not l2_image_path: return

        print("  [Level 2] 分析二级页面...")
        l2_regions = self.detector.analyze_image(l2_image_path)
        
        # 选取 L2 的动作 (混合点击和滑动)
        l2_actions = []
        # 优先加几个滑动
        for r in l2_regions['slidable_regions']:
            r['_act_type'] = 'slide'
            l2_actions.append(r)
        # 再加几个点击
        for r in l2_regions['clickable_regions']:
            r['_act_type'] = 'click'
            l2_actions.append(r)
        
        # 截断数量
        l2_actions = l2_actions[:max_interactions]
        
        l2_results = []
        if l2_actions:
            print(f"  [Level 2] 执行 {len(l2_actions)} 个子操作...")
            
            for j, sub_region in enumerate(l2_actions):
                sub_prefix = f"L1_{prefix_type}{l1_index}_L2_{j}"
                
                # L2 的操作我们设置 auto_back=True，因为只做2层，不再深入
                if sub_region['_act_type'] == 'slide':
                    sub_res = self.tester.run_slide_test(sub_region, sub_prefix, auto_back=True)
                else:
                    sub_res = self.tester.run_click_test(sub_region, sub_prefix, auto_back=True)
                
                if sub_res:
                    l2_results.append(sub_res)
            
            parent_res['l2_exploration'] = l2_results
        else:
            print("  [Level 2] 未发现可交互区域。")

        # D. 完成 L2 探索后，必须回退到 L1
        print("  <<< Level 2 结束，回退到首页")
        self.controller.back(self.app_package)

    def explore_app(self, max_l1_clicks=5, max_l2_interactions=3):
        """
        深度为2的树状探索
        """
        print(f"开始Depth-2应用探索: {self.app_package}")
        print("=" * 60)
        
        # 1. 初始化 App
        if not self.controller.reset_app_state(self.app_package): return
        time.sleep(1.5) 
        
        # 2. L1 (首页) 分析
        l1_screenshot = self.controller.take_screenshot("L1_Home")
        if not l1_screenshot: return

        # 首页一般都可以上下左右滑动
        region_home_v = {
            'bbox': [0, 0, 1000, 1000],
            'direction': 'vertical',
            'description': '在首页上划发现更多内容'
        }
        region_home_h = {
            'bbox': [0, 0, 1000, 1000],
            'direction': 'horizontal',
            'description': '在首页向左滑动发现更多内容'
        }
        
        print("\n[Level 1] 分析首页交互区域...")
        l1_regions = self.detector.analyze_image(l1_screenshot['filename'])
        l1_clicks = l1_regions['clickable_regions'][:max_l1_clicks]
        l1_slides = [region_home_v, region_home_h] + l1_regions['slidable_regions']
        
        results_tree = {
            'l1_slides': [],
            'l1_clicks': []
        }

        # --- 3. L1 滑动测试 (现在支持触发 L2) ---
        print(f"\n[Level 1] 执行滑动测试 ({len(l1_slides)}个)...")
        for i, region in enumerate(l1_slides):
            print(f"\n--- 处理 L1 滑动 #{i} ---")
            # auto_back=False，允许我们观察滑动后的状态并进入L2
            res = self.tester.run_slide_test(region, f"L1_Slide_{i}", auto_back=False)
            
            if res:
                # 尝试进入 L2
                self._process_l2_exploration(res, i, "Slide", max_l2_interactions)
                results_tree['l1_slides'].append(res)

        # --- 4. L1 点击测试 ---
        print(f"\n[Level 1] 执行点击测试 ({len(l1_clicks)}个)...")
        for i, region in enumerate(l1_clicks):
            print(f"\n--- 处理 L1 点击 #{i} ---")
            # auto_back=False，允许进入L2
            res = self.tester.run_click_test(region, f"L1_Click_{i}", auto_back=False)
            
            if res:
                # 尝试进入 L2
                self._process_l2_exploration(res, i, "Click", max_l2_interactions)
                results_tree['l1_clicks'].append(res)

        # 5. 保存报告
        self._save_tree_report(results_tree)

    def _save_tree_report(self, results):
        """保存树状结构的报告"""
        # 简单统计
        l1_clicks = len(results['l1_clicks'])
        l1_slides = len(results['l1_slides'])
        
        count_l2 = 0
        for r in results['l1_clicks'] + results['l1_slides']:
            if r.get('l2_exploration'):
                count_l2 += len(r['l2_exploration'])
        
        print("\n" + "=" * 60)
        print("探索完成！")
        print(f"L1 点击: {l1_clicks}, L1 滑动: {l1_slides}")
        print(f"L2 子操作总数: {count_l2}")
        
        report = {
            'app_package': self.app_package,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'structure': 'depth_2_tree',
            'device': self.controller.get_device_info(),
            'results': results
        }

        report = json_safe(report)
        file_path = self.logs_dir / f"report_tree_{int(time.time())}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {file_path}")
