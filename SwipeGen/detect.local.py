import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import json
import re
from typing import List, Dict
import sys
import os

class ExplorationDetector:
    def __init__(self, model_path: str):
        """初始化交互区域检测器"""
        print(f"正在加载模型: {model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        print("模型加载完成!")
    
    def analyze_image(self, image_path: str) -> Dict[str, List]:
        """分析图片中的可交互区域（可滑动和可点击）"""
        try:
            image = Image.open(image_path)
            print(f"成功加载图片: {image_path}, 尺寸: {image.size}")
        except Exception as e:
            print(f"无法打开图片: {e}")
            return {"clickable_regions": [], "slidable_regions": []}
        
        # 新的提示词，要求同时识别可点击和可滑动区域
        prompt = """请仔细分析这张移动应用UI界面截图，找出所有可滑动的区域，即可以上下或左右滚动/滑动的区域，如列表、轮播图、页面整体等。
页面中找不到可互动区域时，随机找出1个可点击区域，即按钮、图标、链接、卡片、列表项等可以点击的元素。

对于每个区域，请提供：
1. category: "clickable" 或 "slidable"
2. type: 具体类型描述（如：按钮、列表、轮播图等）
3. direction: 如果是slidable，滑动方向（horizontal/vertical/both）
4. bbox: 边界框坐标 [x1,y1,x2,y2]，x,y范围0-1000
5. description: 功能描述
6. interaction: 交互方式（对clickable是"click"/"long_press"，对slidable是"swipe"）

请以JSON数组格式输出，每个元素是一个对象。
只输出JSON格式，不要有其他文字说明。"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        print("正在分析图片中的交互区域...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print("模型输出:")
        print(response[:500])
        print("\n" + "="*50 + "\n")
        
        # 解析并分类结果
        all_regions = self._parse_response(response)
        
        # 按类别分类
        clickable_regions = []
        slidable_regions = []
        
        for region in all_regions:
            category = region.get('category', '').lower()
            if 'click' in category:
                clickable_regions.append(region)
            elif 'slide' in category:
                slidable_regions.append(region)
            else:
                # 根据type推断
                if any(keyword in region.get('type', '').lower() for keyword in ['button', 'icon', 'tab', 'card', 'item']):
                    clickable_regions.append(region)
                elif any(keyword in region.get('type', '').lower() for keyword in ['list', 'scroll', 'carousel', 'swipe']):
                    slidable_regions.append(region)
        
        return {
            "clickable_regions": clickable_regions,
            "slidable_regions": slidable_regions
        }
    
    def _parse_response(self, response: str) -> List[Dict]:
        """解析模型的JSON响应"""
        # 尝试提取JSON部分
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        match = re.search(json_pattern, response, re.DOTALL)
        
        if not match:
            print("未找到JSON格式的输出，尝试直接解析...")
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
            else:
                print("无法解析输出格式")
                return []
        else:
            json_str = match.group(0)
        
        try:
            json_str = json_str.replace('\n', ' ').replace('\t', ' ')
            json_str = re.sub(r'//.*?\n', '', json_str)
            
            regions = json.loads(json_str)
            
            validated_regions = []
            for region in regions:
                if self._validate_region(region):
                    # 确保坐标是浮点数
                    if 'bbox' in region:
                        region['bbox'] = [float(coord) for coord in region['bbox']]
                    validated_regions.append(region)
            
            return validated_regions
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始字符串: {json_str[:200]}...")
            return []
    
    def _validate_region(self, region: Dict) -> bool:
        """验证区域数据格式"""
        if not isinstance(region, dict):
            return False
        
        # 必须有bbox
        if 'bbox' not in region:
            return False
        
        bbox = region['bbox']
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        
        # 检查坐标是否为数字
        try:
            [float(coord) for coord in bbox]
        except (ValueError, TypeError):
            return False
        
        return True
    
    def visualize_results(self, image_path: str, results: Dict[str, List], save_path: str = None):
        """可视化检测结果"""
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            print("请安装Pillow库以可视化结果: pip install pillow")
            return
        
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # 颜色设置
        colors = {
            'clickable': 'green',
            'slidable_horizontal': 'blue',
            'slidable_vertical': 'red',
            'slidable_both': 'purple'
        }
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        print(f"\n检测结果统计:")
        print(f"可点击区域: {len(results['clickable_regions'])} 个")
        print(f"可滑动区域: {len(results['slidable_regions'])} 个")
        print("-" * 50)
        
        all_regions = results['clickable_regions'] + results['slidable_regions']
        
        for i, region in enumerate(all_regions):
            bbox = region['bbox']
            # 转换坐标到实际像素
            bbox = [
                bbox[0] / 1000 * width,
                bbox[1] / 1000 * height,
                bbox[2] / 1000 * width,
                bbox[3] / 1000 * height
            ]
            
            # 确定颜色
            if region in results['clickable_regions']:
                color = colors['clickable']
                label = f"C{i+1}"
            else:
                direction = region.get('direction', '').lower()
                if 'horiz' in direction:
                    color = colors['slidable_horizontal']
                elif 'vert' in direction:
                    color = colors['slidable_vertical']
                else:
                    color = colors['slidable_both']
                label = f"S{i+1-len(results['clickable_regions'])}"
            
            # 绘制边界框
            draw.rectangle(bbox, outline=color, width=3)
            
            # 添加标签
            draw.text((bbox[0], bbox[1] - 25), label, fill=color, font=font)
            
            # 打印信息
            category = "可点击" if region in results['clickable_regions'] else "可滑动"
            print(f"{label} [{category}]: {region.get('type', '未知')}")
            print(f"  位置: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print(f"  描述: {region.get('description', '无')}")
            print("-" * 50)
        
        if save_path:
            image.save(save_path)
            print(f"\n可视化结果已保存到: {save_path}")
        else:
            image.show()
        
        return image


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python detect.py <图片路径> [模型路径]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    MODEL_PATH = sys.argv[2] if len(sys.argv) > 2 else "/home/xiyuan/data/model/Qwen3-VL-8B-Instruct"
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型路径不存在: {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在: {image_path}")
        sys.exit(1)
    
    try:
        detector = ExplorationDetector(MODEL_PATH)
    except Exception as e:
        print(f"初始化检测器失败: {e}")
        sys.exit(1)
    
    results = detector.analyze_image(image_path)
    
    if results['clickable_regions'] or results['slidable_regions']:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_detected.png"
        
        detector.visualize_results(image_path, results, output_path)
        
        json_path = f"{base_name}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"详细结果已保存到: {json_path}")
    else:
        print("未检测到交互区域")

if __name__ == "__main__":
    main()