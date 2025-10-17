from .modelscope_nodes import *

NODE_CLASS_MAPPINGS = {
    "ModelScopeMultiTurnChat_Sevr": ModelScopeMultiTurnChat_Sevr,
    "ConversationHistorySaver_Sevr": ConversationHistorySaver_Sevr,
    "ConversationHistoryLoader_Sevr": ConversationHistoryLoader_Sevr,
    "ModelScopeVisionPromptInversion_Sevr": ModelScopeVisionPromptInversion_Sevr,
    "ModelScopeImageEditorV3_Sevr": ModelScopeImageEditorV3_Sevr,
    "ModelScopeImageGeneratorV2_Sevr": ModelScopeImageGeneratorV2_Sevr,
    "本地txt输入_Sevr": 本地txt输入_Sevr,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeMultiTurnChat_Sevr": "ModelScope多轮对话_Sevr",
    "ConversationHistorySaver_Sevr": "保存对话历史_Sevr",
    "ConversationHistoryLoader_Sevr": "加载对话历史_Sevr",
    "ModelScopeVisionPromptInversion_Sevr": "ModelScope视觉提示词生成_Sevr",
    "ModelScopeImageEditorV3_Sevr": "ModelScope图生图编辑v3_Sevr",
    "ModelScopeImageGeneratorV2_Sevr": "ModelScope文生图_Sevr",
    "本地txt输入_Sevr": "本地txt输入_Sevr",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']