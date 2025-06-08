import os
import random
import base64
import io

import folder_paths
from PIL import Image
import numpy as np
import torch
import hashlib

# set the models directory
if "conditionings" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "conditionings")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["conditionings"]
folder_paths.folder_names_and_paths["conditionings"] = (current_paths, ".bin")

class SaveConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"conditionings": ("CONDITIONING", ),},
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_conditioning"
    CATEGORY = "SaveCondition"

    def save_conditioning(self, conditionings): # conditionings : [[text, {"pooled_output"}]...]
        # 使用内存缓冲区而不是文件
        buffer = io.BytesIO()
        torch.save(conditionings[0], buffer)
        buffer.seek(0)
        
        # 转换为base64编码的字符串
        encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # 打印一些调试信息
        conditioning = conditionings[0]
        print(f"conditioning[0].shape:{conditioning[0].shape}")
        for key, value in conditioning[1].items():
            if(type(value) == torch.Tensor):
                print(f"key:{key}, type:{type(value)}, shape:{value.shape}")
            else:
                print(f"key:{key}, type:{type(conditioning[1][key])}")

        if(hasattr(conditioning[0], "addit_embeds")):
           for key, value in conditioning[0].addit_embeds.items():
                if(type(value) == torch.Tensor):
                    print(f"key:{key}, type:{type(value)}, shape:{value.shape}")
                else:
                    print(f"key:{key}, type:{type(value)}")
                    
        return (encoded_string,)

class LoadContditioning():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "conditioning_string": ("STRING", {"multiline": True})}}

    CATEGORY = "SaveCondition"
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "load_conditioning"

    def load_conditioning(self, conditioning_string):
        try:
            # 从base64字符串解码
            decoded_bytes = base64.b64decode(conditioning_string)
            buffer = io.BytesIO(decoded_bytes)
            
            # 从内存缓冲区加载
            conditioning_list = torch.load(buffer)
            
            # 确保所有张量都在CPU上
            conditioning_list[0] = conditioning_list[0].cpu()
            for key, value in conditioning_list[1].items():
                if(type(value) == torch.Tensor):
                    conditioning_list[1][key] = value.cpu()
            if(hasattr(conditioning_list[0], "addit_embeds")):
                for key, value in conditioning_list[0].addit_embeds.items():
                    if(type(value) == torch.Tensor):
                        conditioning_list[0].addit_embeds[key] = value.cpu()
            return ([conditioning_list], )
        except Exception as e:
            print(f"Error loading conditioning from string: {e}")
            # 返回一个空的conditioning作为fallback
            return ([], )
    

NODE_CLASS_MAPPINGS = {
    "SaveConditioning": SaveConditioning,
    "LoadContditioning": LoadContditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveConditioningAsString": "SaveConditioning",
    "LoadConditioningFromString": "LoadContditioning"
}