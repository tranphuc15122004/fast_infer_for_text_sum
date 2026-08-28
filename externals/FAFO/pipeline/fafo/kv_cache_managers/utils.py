from pipeline.fafo.kv_cache_managers.stream_manager import StreamManager
from pipeline.fafo.kv_cache_managers.quest_manager import QuestManager

KV_CACHE_MANAGERS = {
    'stream-llm': StreamManager,
    'quest': QuestManager,
}
