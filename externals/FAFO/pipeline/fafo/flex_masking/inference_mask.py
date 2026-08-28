from pipeline.fafo.flex_masking.stream_inference_mask import generate_stream_inference_mask
from pipeline.fafo.flex_masking.quest_inference_mask import generate_quest_inference_mask

INFERENCE_MASKS = {
    'stream-llm': generate_stream_inference_mask,
    'quest': generate_quest_inference_mask,
}
