import comfy
import comfy.utils
from comfy.model_management import get_torch_device
import torch
from openai import OpenAI
import json
import math
from PIL import Image
import io
import base64
import requests
import time
import numpy as np
from io import BytesIO
import uuid
from datetime import datetime
import re
import os
import glob
import folder_paths

# ==================== 多轮对话节点 ====================
class ModelScopeMultiTurnChat_Sevr:
    @classmethod
    def INPUT_TYPES(cls):
        model_list = [
            "deepseek-ai/DeepSeek-R1-0528",
            "deepseek-ai/DeepSeek-V3.1",
            "deepseek-ai/DeepSeek-V3.2-Exp",  # 新增模型
            "iic/Tongyi-DeepResearch-30B-A3B",  # 新增模型
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "ZhipuAI/GLM-4.5",
            "ZhipuAI/GLM-4.6",  # 新增模型
            "其他模型"
        ]
        
        return {
            "required": {
                "user_input": ("STRING", {"multiline": True, "default": "你好"}),
                "system_prompt": ("STRING", {"multiline": True, "default": "你是一个有帮助的助手。"}),
                "model_name": (model_list, {"default": "Qwen/Qwen3-Next-80B-A3B-Instruct"}),
                "reset_conversation": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "api_key": ("STRING", {"default": "ms-73ac35eb-38eb-4571-80ad-42e40956e1e5"}),
                "custom_model": ("STRING", {"default": ""}),
                "conversation_history": ("CONVERSATION_HISTORY",),
            }
        }

    RETURN_TYPES = ("STRING", "CONVERSATION_HISTORY")
    RETURN_NAMES = ("回复", "对话历史")
    FUNCTION = "call_api"
    CATEGORY = "AI接口_Sevr"

    def __init__(self):
        self.history = []

    def call_api(self, user_input, system_prompt, model_name, reset_conversation, 
                api_key="ms-73ac35eb-38eb-4571-80ad-42e40956e1e5", 
                custom_model="", conversation_history=None):
        if reset_conversation or conversation_history is None:
            self.history = [
                {
                    'role': 'system',
                    'content': system_prompt
                }
            ]
        else:
            self.history = conversation_history
        
        self.history.append({
            'role': 'user',
            'content': user_input
        })
        
        actual_model = custom_model if custom_model else model_name
        
        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key=api_key,
        )

        try:
            reasoning_models = [
                "deepseek-ai/DeepSeek-V3.1",
                "deepseek-ai/DeepSeek-V3.2-Exp",  # 新增推理模型
                "Qwen/Qwen3-Next-80B-A3B-Thinking",
                "deepseek-ai/DeepSeek-R1-0528"
            ]
            
            if actual_model in reasoning_models:
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=self.history,
                    stream=True
                )
                
                full_response = ""
                reasoning_content = ""
                done_reasoning = False
                
                for chunk in response:
                    reasoning_chunk = getattr(chunk.choices[0].delta, 'reasoning_content', '')
                    answer_chunk = getattr(chunk.choices[0].delta, 'content', '')
                    
                    if reasoning_chunk != '':
                        reasoning_content += reasoning_chunk
                    elif answer_chunk != '':
                        if not done_reasoning and reasoning_content:
                            full_response += f"\n\n=== 推理过程 ===\n{reasoning_content}\n\n=== 最终回答 ===\n"
                            done_reasoning = True
                        full_response += answer_chunk
                
                if not reasoning_content:
                    full_response = answer_chunk
            else:
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=self.history,
                    stream=True
                )

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content

            self.history.append({
                'role': 'assistant',
                'content': full_response
            })

            return (full_response, self.history)
        except Exception as e:
            return (f"API调用出错: {str(e)}", self.history)

class ConversationHistorySaver_Sevr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conversation_history": ("CONVERSATION_HISTORY",),
                "file_name": ("STRING", {"default": "conversation_history.json"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_history"
    OUTPUT_NODE = True
    CATEGORY = "AI接口_Sevr"

    def save_history(self, conversation_history, file_name):
        if not file_name.endswith('.json'):
            file_name += '.json'
        
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, ensure_ascii=False, indent=2)
        
        print(f"对话历史已保存到: {file_name}")
        return ()

class ConversationHistoryLoader_Sevr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": ("STRING", {"default": "conversation_history.json"}),
            }
        }

    RETURN_TYPES = ("CONVERSATION_HISTORY",)
    RETURN_NAMES = ("对话历史",)
    FUNCTION = "load_history"
    CATEGORY = "AI接口_Sevr"

    def load_history(self, file_name):
        if not file_name.endswith('.json'):
            file_name += '.json'
        
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            print(f"已从 {file_name} 加载对话历史")
            return (history,)
        except FileNotFoundError:
            print(f"文件 {file_name} 不存在，返回空历史")
            return ([],)
        except Exception as e:
            print(f"加载对话历史时出错: {str(e)}")
            return ([],)

# ==================== 视觉多模态理解节点 ====================
class ModelScopeVisionPromptInversion_Sevr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt_text": ("STRING", {"multiline": True, "default": "详细描述这幅图像内容，并生成适合AI绘图的提示词"}),
                "model_name": ([
                    # 新增最新模型
                    "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    "Qwen/Qwen3-VL-30B-A3B-Instruct",  # 新增模型
                    "Qwen/Qwen3-VL-30B-A3B-Thinking",  # 新增模型
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                    "Qwen/Qwen2.5-VL-32B-Instruct",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "stepfun-ai/step3",
                    "Qwen/Qwen2-VL-7B-Instruct",
                    "PaddlePaddle/ERNIE-4.5-VL-28B-A3B-PT",
                    "Qwen/QVQ-72B-Preview"
                ], {"default": "Qwen/Qwen3-VL-235B-A22B-Instruct"}),  # 更新默认模型为最新模型
            },
            "optional": {
                "api_key": ("STRING", {"default": "ms-73ac35eb-38eb-4571-80ad-42e40956e1e5"}),
                "custom_model": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "AI接口/视觉_Sevr"

    def generate_prompt(self, image, prompt_text, model_name, api_key="ms-73ac35eb-38eb-4571-80ad-42e40956e1e5", custom_model=""):
        actual_model = custom_model if custom_model else model_name
        
        pil_image = Image.fromarray(torch.clamp(image[0] * 255, 0, 255).byte().numpy(), 'RGB')
        width, height = pil_image.size
        ratio = width / height
        
        if width * height > 1024 * 1024:
            target_width = math.sqrt(ratio * 1024 * 1024)
            target_height = target_width / ratio
            target_width = int(target_width)
            target_height = int(target_height)
            pil_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        data_url = f"data:image/png;base64,{img_base64}"

        client = OpenAI(
            base_url='https://api-inference.modelscope.cn/v1',
            api_key=api_key,
        )

        try:
            # 处理思考模型
            reasoning_models = [
                "Qwen/Qwen3-VL-30B-A3B-Thinking"
            ]
            
            if actual_model in reasoning_models:
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=[{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': prompt_text,
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': data_url,
                                },
                            }
                        ],
                    }],
                    stream=True
                )
                
                full_response = ""
                reasoning_content = ""
                done_reasoning = False
                
                for chunk in response:
                    reasoning_chunk = getattr(chunk.choices[0].delta, 'reasoning_content', '')
                    answer_chunk = getattr(chunk.choices[0].delta, 'content', '')
                    
                    if reasoning_chunk != '':
                        reasoning_content += reasoning_chunk
                    elif answer_chunk != '':
                        if not done_reasoning and reasoning_content:
                            full_response += f"\n\n=== 推理过程 ===\n{reasoning_content}\n\n=== 最终回答 ===\n"
                            done_reasoning = True
                        full_response += answer_chunk
                
                if not reasoning_content:
                    full_response = answer_chunk
            else:
                response = client.chat.completions.create(
                    model=actual_model,
                    messages=[{
                        'role': 'user',
                        'content': [
                            {
                                'type': 'text',
                                'text': prompt_text,
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': data_url,
                                },
                            }
                        ],
                    }],
                    stream=True
                )

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content

            return (full_response,)
        except Exception as e:
            return (f"API调用出错: {str(e)}",)

# ==================== 图生图编辑节点 v3 ====================
class ModelScopeImageEditorV3_Sevr:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_options = [
            "AutoSize",
            "928x1664",
            "1104x1472",
            "1328x1328",
            "1472x1104",
            "1664x928",
            "1024x1024",
            "2048x2048",
            "1024x576",
            "576x1024",
        ]
        
        image_hosting_options = [
            "Uguu.se",
            "imgbb"
        ]
        
        model_options = [
            "Qwen/Qwen-Image-Edit",
            "MusePublic/FLUX.1-Kontext-Dev"
        ]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "manual_image_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "直接输入图片URL（优先级最高，留空则使用图床）"
                }),
                "prompt": ("STRING", {"multiline": True, "default": "turn the object into a different color"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "色调艳丽，过曝，细节模糊不清，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，杂乱的背景，三条腿"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
                "size": (resolution_options, {"default": "AutoSize"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "modelscope_api_key": ("STRING", {"default": "your-modelscope-api-key-here"}),
                "model": (model_options, {"default": "Qwen/Qwen-Image-Edit"}),
                "image_hosting_service": (image_hosting_options, {"default": "Uguu.se"}),
                "imgbb_api_key": ("STRING", {"default": "your-imgbb-api-key-here", "multiline": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("edited_image", "usage_instruction")
    FUNCTION = "edit_image"
    CATEGORY = "AI接口/图像编辑_Sevr"

    def get_image_size(self, image):
        image_tensor = image[0]
        height, width = image_tensor.shape[0], image_tensor.shape[1]
        return f"{width}x{height}"

    def upload_to_uguu(self, image):
        try:
            image_tensor = image[0]
            image_array = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            buffered.seek(0)
            
            files = {'files[]': ('image.png', buffered, 'image/png')}
            response = requests.post("https://uguu.se/upload", files=files)
            
            if response.status_code == 200:
                response_json = response.json()
                if 'files' in response_json and len(response_json['files']) > 0:
                    url_without_backslashes = response_json['files'][0]['url'].replace('\\', '')
                    return url_without_backslashes
                else:
                    raise Exception("Uguu.se返回格式异常，未找到图片URL")
            else:
                raise Exception(f"Uguu.se上传失败，HTTP状态码: {response.status_code}")
                
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                raise Exception("Uguu.se图片上传超时，请检查网络连接")
            else:
                raise Exception(f"Uguu.se图片上传失败: {error_msg}")

    def upload_to_imgbb(self, image, imgbb_api_key):
        try:
            image_tensor = image[0]
            image_array = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=95)
            buffered.seek(0)
            
            image_size = buffered.getbuffer().nbytes
            if image_size > 10 * 1024 * 1024:
                quality = max(30, 95 - int((image_size - 10*1024*1024) / (1024*1024) * 5))
                pil_image.save(buffered, format="JPEG", quality=quality, optimize=True)
                buffered.seek(0)
            
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            response = requests.post(
                'https://api.imgbb.com/1/upload',
                data={
                    'key': imgbb_api_key,
                    'image': img_str
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    image_url = result['data']['url']
                    return image_url
                else:
                    error_msg = result.get('error', {}).get('message', '未知错误')
                    raise Exception(f"imgbb上传失败: {error_msg}")
            else:
                raise Exception(f"imgbb上传失败，HTTP状态码: {response.status_code}")
                
        except Exception as e:
            if "key" in str(e).lower():
                raise Exception("imgbb API密钥无效，请检查密钥是否正确")
            elif "timeout" in str(e).lower():
                raise Exception("图片上传超时，请尝试减小图片尺寸或检查网络连接")
            else:
                raise Exception(f"图片上传失败: {e}")

    def edit_image(self, image, manual_image_url, prompt, negative_prompt, batch_size, size, steps, guidance, seed, 
                  modelscope_api_key, model, image_hosting_service, imgbb_api_key):
        base_url = 'https://api-inference.modelscope.cn/'
        
        common_headers = {
            "Authorization": f"Bearer {modelscope_api_key}",
            "Content-Type": "application/json",
        }

        final_image_url = None
        if manual_image_url and manual_image_url.strip():
            final_image_url = manual_image_url.strip()
        else:
            final_size = size
            if size == "AutoSize":
                final_size = self.get_image_size(image)

            if image_hosting_service == "Uguu.se":
                final_image_url = self.upload_to_uguu(image)
            elif image_hosting_service == "imgbb":
                if not imgbb_api_key or imgbb_api_key.strip() == "your-imgbb-api-key-here":
                    raise Exception("请提供有效的imgbb API密钥")
                final_image_url = self.upload_to_imgbb(image, imgbb_api_key.strip())
        
        if not final_image_url:
            raise Exception(f"图片URL获取失败")
        
        request_data = {
            "model": model,
            "prompt": prompt,
            "image_url": final_image_url,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images_per_prompt": batch_size,
        }
        
        # 从旧版本迁移的AutoSize逻辑 - 当没有手动输入URL时应用尺寸设置
        if not manual_image_url or not manual_image_url.strip():
            final_size = size
            if size == "AutoSize":
                final_size = self.get_image_size(image)
            request_data["size"] = final_size
        
        if seed != -1:
            if seed < -1 or seed > 2147483647:
                seed = -1
            else:
                base_seed = seed
                seed_list = []
                for i in range(batch_size):
                    safe_seed = (base_seed + i) % 2147483647
                    seed_list.append(safe_seed)
                request_data["seed"] = seed_list
        
        if model == "Qwen/Qwen-Image-Edit":
            request_data["negative_prompt"] = negative_prompt

        instruction_text = ""

        try:
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(request_data, ensure_ascii=False).encode('utf-8'),
                timeout=30
            )

            response.raise_for_status()
            response_data = response.json()
            task_id = response_data["task_id"]

            max_attempts = 60
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                
                try:
                    result = requests.get(
                        f"{base_url}v1/tasks/{task_id}",
                        headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                        timeout=15
                    )
                    result.raise_for_status()
                    data = result.json()

                    if data["task_status"] == "SUCCEED":
                        output_images = data["output_images"]
                        images = []
                        
                        for i, url in enumerate(output_images):
                            image_response = requests.get(url, timeout=30)
                            edited_image = Image.open(BytesIO(image_response.content))
                            edited_image = edited_image.convert("RGB")
                            
                            image_tensor = torch.from_numpy(np.array(edited_image).astype(np.float32) / 255.0)
                            images.append(image_tensor)
                        
                        batch_tensor = torch.stack(images)
                        return (batch_tensor, instruction_text)
                        
                    elif data["task_status"] == "FAILED":
                        error_msg = data.get("error_message", "未知错误")
                        raise Exception(f"图像编辑失败: {error_msg}")
                        
                    time.sleep(5)
                    
                except requests.exceptions.RequestException as e:
                    if attempt >= max_attempts:
                        raise Exception(f"查询任务状态超时: {e}")
                    time.sleep(5)
                    
            raise Exception("任务处理超时")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求错误: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = f"API错误: {error_detail}"
                except:
                    error_msg = f"HTTP错误: {e.response.status_code} - {e.response.text}"
            raise Exception(error_msg)

# ==================== 文生图节点 ====================
class ModelScopeImageGeneratorV2_Sevr:
    @classmethod
    def INPUT_TYPES(cls):
        resolution_options = [
            "928x1664",
            "1104x1472", 
            "1328x1328",
            "1472x1104",
            "1664x928",
            "1024x1024",
            "2048x2048",
            "1024x576",
            "576x1024",
            "custom",
        ]
        
        # 调整后的基础模型选项列表
        base_model_options = [
            "Qwen/Qwen-Image",
            "MusePublic/Qwen-image",
            "Qwen/Qwen-Image-Edit",
            "black-forest-labs/FLUX.1-Krea-dev",
            "MusePublic/FLUX.1-Kontext-Dev",
            "DiffSynth-Studio/FLUX.1-Kontext-dev-lora-highresfix",
            "DiffSynth-Studio/FLUX.1-Kontext-dev-lora-ArtAug",
            "MusePublic/489_ckpt_FLUX_1",
            "DiffSynth-Studio/ArtAug-lora-FLUX.1dev-v1",
            "MusePublic/806_lora_FLUX_1",
            "MAILAND/majicflus_v1",
        ]
        
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A golden cat"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "色调艳丽，过曝，细节模糊不清，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，杂乱的背景，三条腿"}),
                "base_model": (base_model_options, {"default": "Qwen/Qwen-Image"}),
                "custom_model_id": ("STRING", {"default": "", "multiline": False}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),  # 修改最大值为8
                "size": (resolution_options, {"default": "1144x2048"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 4, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "custom_width": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "custom_height": ("INT", {"default": 1024, "min": 64, "max": 4096}),
                "api_key": ("STRING", {"default": "your-modelscope-api-key-here"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "图像/生成_Sevr"

    def generate_image(self, prompt, negative_prompt, base_model, custom_model_id, batch_size, size, steps, guidance, seed, 
                      custom_width, custom_height, api_key):
        base_url = 'https://api-inference.modelscope.cn/'
        
        common_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if custom_model_id and custom_model_id.strip():
            model_id = custom_model_id.strip()
        else:
            model_id = base_model

        if size == "custom":
            final_size = f"{custom_width}x{custom_height}"
        else:
            final_size = size

        request_data = {
            "model": model_id,
            "prompt": prompt,
            "size": final_size,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images_per_prompt": batch_size,
        }
        
        # 为支持负面提示词的模型添加负面提示词
        qwen_models = [
            "Qwen/Qwen-Image", 
            "Qwen/Qwen-Image-Edit", 
            "MusePublic/Qwen-image",
        ]
        
        if negative_prompt and negative_prompt.strip() and model_id in qwen_models:
            request_data["negative_prompt"] = negative_prompt.strip()
        elif negative_prompt and negative_prompt.strip():
            # 对于不支持负面提示词的模型，可以选择忽略或者给出警告
            print(f"警告: 模型 {model_id} 可能不支持负面提示词，已忽略负面提示词参数")
            
        if seed != -1:
            base_seed = seed
            request_data["seed"] = [base_seed + i for i in range(batch_size)]

        print(f"发送请求到ModelScope API，模型: {model_id}")
        print(f"分辨率: {final_size}")
        print(f"请求数据: {json.dumps(request_data, ensure_ascii=False, indent=2)}")

        try:
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps(request_data, ensure_ascii=False).encode('utf-8'),
                timeout=30
            )

            response.raise_for_status()
            response_data = response.json()
            task_id = response_data["task_id"]
            print(f"任务已创建，任务ID: {task_id}")

            max_attempts = 60
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                print(f"检查任务状态 ({attempt}/{max_attempts})...")
                
                try:
                    result = requests.get(
                        f"{base_url}v1/tasks/{task_id}",
                        headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                        timeout=15
                    )
                    result.raise_for_status()
                    data = result.json()
                    
                    print(f"任务状态: {data['task_status']}")

                    if data["task_status"] == "SUCCEED":
                        image_urls = data["output_images"]
                        images = []

                        for i, url in enumerate(image_urls):
                            print(f"下载图像 {i+1}/{len(image_urls)}...")
                            image_response = requests.get(url, timeout=30)
                            image = Image.open(BytesIO(image_response.content))
                            image = image.convert("RGB")
                            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
                            images.append(image_tensor)

                        batch_tensor = torch.stack(images)
                        print("图像生成完成!")
                        return (batch_tensor,)
                        
                    elif data["task_status"] == "FAILED":
                        error_msg = data.get("error_message", "未知错误")
                        raise Exception(f"图像生成失败: {error_msg}")
                        
                    time.sleep(5)
                    
                except requests.exceptions.RequestException as e:
                    print(f"查询任务状态时出错: {e}")
                    if attempt >= max_attempts:
                        raise Exception(f"查询任务状态超时: {e}")
                    time.sleep(5)
                    
            raise Exception("任务处理超时")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求错误: {e}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = f"API错误: {error_detail}"
                except:
                    error_msg = f"HTTP错误: {e.response.status_code} - {e.response.text}"
            raise Exception(error_msg)

# ==================== TXT文件合并器节点 ====================
class 本地txt输入_Sevr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入文件夹路径，例如: C:/documents 或 ./input"
                }),
                "scan_subfolders": ("BOOLEAN", {
                    "default": False,
                    "label_on": "启用",
                    "label_off": "禁用"
                }),
                "text_filter": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入要过滤的文本（留空则不过滤）"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("merged_text", "file_count")
    FUNCTION = "merge_txt_files"
    CATEGORY = "AI接口_Sevr"
    DESCRIPTION = "扫描指定文件夹中的txt文件并合并文本内容"

    def merge_txt_files(self, folder_path, scan_subfolders, text_filter):
        if not folder_path or not os.path.exists(folder_path):
            return ("错误：文件夹路径不存在或为空", "0")
        
        # 构建搜索模式
        if scan_subfolders:
            search_pattern = os.path.join(folder_path, "**", "*.txt")
        else:
            search_pattern = os.path.join(folder_path, "*.txt")
        
        # 获取所有txt文件
        txt_files = []
        try:
            if scan_subfolders:
                txt_files = glob.glob(search_pattern, recursive=True)
            else:
                txt_files = glob.glob(search_pattern)
        except Exception as e:
            return (f"扫描文件时出错: {str(e)}", "0")
        
        if not txt_files:
            return ("未找到任何txt文件", "0")
        
        # 读取并处理所有txt文件内容
        merged_content = []
        processed_files = 0
        filtered_count = 0
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    
                    # 应用文本过滤（如果设置了过滤文本）
                    if text_filter and text_filter.strip():
                        original_length = len(content)
                        content = content.replace(text_filter, "")
                        if original_length != len(content):
                            filtered_count += 1
                    
                    # 替换回车换行为逗号
                    content = content.replace('\r\n', ',').replace('\n', ',')
                    content = content.replace('\r', ',')
                    
                    # 清理多余的逗号
                    while ',,' in content:
                        content = content.replace(',,', ',')
                    content = content.strip(',')
                    
                    if content:
                        merged_content.append(content)
                        processed_files += 1
                        
            except UnicodeDecodeError:
                # 如果UTF-8解码失败，尝试其他编码
                try:
                    with open(file_path, 'r', encoding='gbk') as file:
                        content = file.read().strip()
                        
                        # 应用文本过滤（如果设置了过滤文本）
                        if text_filter and text_filter.strip():
                            original_length = len(content)
                            content = content.replace(text_filter, "")
                            if original_length != len(content):
                                filtered_count += 1
                        
                        content = content.replace('\r\n', ',').replace('\n', ',')
                        content = content.replace('\r', ',')
                        while ',,' in content:
                            content = content.replace(',,', ',')
                        content = content.strip(',')
                        if content:
                            merged_content.append(content)
                            processed_files += 1
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {str(e)}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        if not merged_content:
            return ("所有txt文件均为空或无法读取", "0")
        
        # 用换行符连接所有处理后的内容
        final_text = '\n'.join(merged_content)
        
        # 准备文件数量信息
        file_count_info = f"成功合并 {processed_files} 个txt文件 (总共找到 {len(txt_files)} 个文件)"
        
        # 如果有过滤操作，添加过滤信息
        if text_filter and text_filter.strip():
            file_count_info += f", 过滤文本 '{text_filter}' 在 {filtered_count} 个文件中被移除"
        
        # 输出统计信息到控制台
        print(file_count_info)
        print(f"总字符数: {len(final_text)}")
        
        return (final_text, file_count_info)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

# 节点显示名称映射
NODE_CLASS_MAPPINGS = {
    "ModelScope多轮对话_Sevr": ModelScopeMultiTurnChat_Sevr,
    "对话历史保存_Sevr": ConversationHistorySaver_Sevr,
    "对话历史加载_Sevr": ConversationHistoryLoader_Sevr,
    "ModelScope视觉多模态理解_Sevr": ModelScopeVisionPromptInversion_Sevr,
    "ModelScope图生图编辑V3_Sevr": ModelScopeImageEditorV3_Sevr,
    "ModelScope文生图V2_Sevr": ModelScopeImageGeneratorV2_Sevr,
    "本地txt输入_Sevr": 本地txt输入_Sevr,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScope多轮对话_Sevr": "ModelScope多轮对话_Sevr",
    "对话历史保存_Sevr": "对话历史保存_Sevr", 
    "对话历史加载_Sevr": "对话历史加载_Sevr",
    "ModelScope视觉多模态理解_Sevr": "ModelScope视觉多模态理解_Sevr",
    "ModelScope图生图编辑V3_Sevr": "ModelScope图生图编辑V3_Sevr",
    "ModelScope文生图V2_Sevr": "ModelScope文生图V2_Sevr",
    "本地txt输入_Sevr": "本地txt输入_Sevr",
}