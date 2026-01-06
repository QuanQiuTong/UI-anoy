import json
import os
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

class InstructionGenerator:
    """
    指令生成器：基于视觉变化和动作元数据，生成自然语言指令。
    """
    def __init__(self, model_path):
        print(f"加载指令生成模型: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def generate_instruction(self, before_path, after_path, action_data):
        """
        为单个动作生成指令
        """
        try:
            img_before = Image.open(before_path).convert("RGB")
            img_after = Image.open(after_path).convert("RGB")
        except Exception as e:
            print(f"读取图片失败: {e}")
            return action_data['intent'] # 返回原有的粗略意图

        # 构建提示词
        # 我们将动作参数转化为文字描述辅助模型
        act_type = action_data['action']
        direction = action_data.get('direction', 'unknown')
        bbox = action_data.get('bbox', [0,0,0,0])
        
        # 构造 Prompt: 我们给模型看两张图，告诉它发生了什么动作，让它推测用户的意图指令
        prompt = f"""
I will provide two screenshots (Before and After) and the action executed.
Action: {act_type}
Direction: {direction}
Bounding Box of Element: {bbox}

Task: Generate a specific, natural language command that a user would give to a UI Agent to perform this action.
The command should be concise.

Examples:
- "Scroll down to see more songs."
- "Swipe right to delete this item."
- "Tap the play button."

Please output ONLY the command text.
"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_before},
                    {"type": "image", "image": img_after},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img_before, img_after], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        return response.strip()

    def process_report(self, report_path, update=True):
        """
        处理 run_qwen.py 生成的完整报告文件，为每个条目补充指令
        """
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        print(f"正在处理报告，生成高阶指令...")
        
        # 处理点击结果
        for item in report['details'].get('click_results', []):
            if item['action_data']['success']: # 只为成功的操作生成指令
                print(f"Generating for Click...")
                new_intent = self.generate_instruction(
                    item['screenshot_before'], 
                    item['screenshot_after'], 
                    item['action_data']
                )
                print(f" -> {new_intent}")
                item['action_data']['intent'] = new_intent

        # 处理滑动结果
        for item in report['details'].get('slide_results', []):
            if item['action_data']['success']:
                print(f"Generating for Slide...")
                new_intent = self.generate_instruction(
                    item['screenshot_before'], 
                    item['screenshot_after'], 
                    item['action_data']
                )
                print(f" -> {new_intent}")
                item['action_data']['intent'] = new_intent

        if update:
            base, ext = os.path.splitext(report_path)
            new_path = f"{base}_with_instructions{ext}"
            with open(new_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"已保存包含指令的新报告: {new_path}")

# 使用示例
if __name__ == "__main__":
    MODEL_PATH = "D:/Games/Qwen3-VL-4B-Instruct"
    REPORT_FILE = "logs/report_1735xxxx.json" # 替换为实际路径
    
    if os.path.exists(REPORT_FILE):
        generator = InstructionGenerator(MODEL_PATH)
        generator.process_report(REPORT_FILE)
    else:
        print("请提供有效的报告文件路径")