# detect.py
import base64
import io
import json
import re
import requests
from PIL import Image
from typing import List, Dict


class ExplorationDetector:
    def __init__(self, server_url: str):
        """
        server_url 示例:
        - http://127.0.0.1:8000
        """
        self.server_url = server_url.rstrip("/")

    def analyze_image(self, image_path: str) -> Dict[str, List]:
        try:
            image = Image.open(image_path).convert("RGB")
            # ===== 缩放到 0.5 =====
            scale = 0.5
            new_size = (
                int(image.width * scale),
                int(image.height * scale)
            )
            image = image.resize(new_size, Image.BILINEAR)
            print(f"Loaded image: {image_path}, size={image.size}")
        except Exception as e:
            print(f"Failed to load image: {e}")
            return {"clickable_regions": [], "slidable_regions": []}

        prompt = """请仔细分析这张移动应用UI界面截图，找出所有可滑动的区域，即可以上下或左右滚动/滑动的区域，如列表、轮播图、页面整体等。
只输出最多前6个可滑动区域的详细信息。

对于每个区域，请提供：
1. category: "clickable" 或 "slidable"
2. type: 具体类型描述（如：按钮、列表、轮播图等）
3. direction: 如果是slidable，滑动方向（horizontal/vertical/both）
4. bbox: 边界框坐标 [x1,y1,x2,y2]，x,y范围0-1000
5. description: 对该区域上的操作意图的描述
6. interaction: 交互方式（对clickable是"click"/"long_press"，对slidable是"swipe"）

请以JSON数组格式输出，每个元素是一个对象。
只输出JSON格式，不要有其他文字说明。
"""

        payload = {
            "prompt": prompt,
            "image_base64": self._encode_image(image)
        }

        print("Sending inference request...")
        resp = requests.post(
            f"{self.server_url}/infer",
            json=payload,
            timeout=180
        )
        resp.raise_for_status()

        response_text = resp.json()["text"]

        print("Model response:")
        print(response_text)
        print("=" * 50)

        all_regions = self._parse_response(response_text)

        clickable, slidable = [], []
        for region in all_regions:
            cat = region.get("category", "").lower()
            if "click" in cat:
                clickable.append(region)
            elif "slid" in cat:
                slidable.append(region)
            else:
                if any(k in region.get("type", "").lower()
                       for k in ["button", "icon", "tab", "card", "item"]):
                    clickable.append(region)
                elif any(k in region.get("type", "").lower()
                         for k in ["list", "scroll", "carousel", "swipe"]):
                    slidable.append(region)

        return {
            "clickable_regions": clickable,
            "slidable_regions": slidable
        }

    # ==========================
    # Utilities
    # ==========================

    def _encode_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _parse_response(self, response: str) -> List[Dict]:
        json_pattern = r'\[\s*\{.*?\}\s*\]'
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            json_str = match.group(0)
        else:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start < 0 or end <= start:
                print("No JSON found in response.")
                return []
            json_str = response[start:end]

        try:
            json_str = json_str.replace("\n", " ").replace("\t", " ")
            regions = json.loads(json_str)
        except Exception as e:
            print(f"JSON parse error: {e}")
            return []

        valid = []
        for r in regions:
            if self._validate_region(r):
                r["bbox"] = [float(x) for x in r["bbox"]]
                valid.append(r)
        return valid

    def _validate_region(self, region: Dict) -> bool:
        if not isinstance(region, dict):
            return False
        if "bbox" not in region or not isinstance(region["bbox"], list):
            return False
        if len(region["bbox"]) != 4:
            return False
        try:
            [float(x) for x in region["bbox"]]
        except Exception:
            return False
        return True
