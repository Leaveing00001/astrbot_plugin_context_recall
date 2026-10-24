"""读空气撤回插件的内部实现。

本包内的模块不依赖 astrbot，便于脱离 AstrBot 单独做逻辑测试。
"""

from .models import ChatLine, message_to_text, normalize_messages

__all__ = ["ChatLine", "message_to_text", "normalize_messages"]
