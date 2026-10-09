from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
import importlib
import os

if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == "sglang_models":
    Qwen3_5TextConfig = importlib.import_module("sglang.srt.configs.qwen3_5").Qwen3_5TextConfig


class YANchorConfig(Qwen3_5TextConfig):
    model_type = "yanchor"

    def __init__(self, backend="torch", prefill_chunk_size=1024, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.prefill_chunk_size = prefill_chunk_size
        if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == "sglang_models":
            self.sliding_window = getattr(self, 'fixed_memory_swa_window', 1024)
            self.is_hybrid_swa = True
            layers = getattr(self, 'fixed_memory_swa_layers', range(3, self.num_hidden_layers, 4))
            self.hybrid_layer_pattern = [1 if i in layers else -1 for i in range(self.num_hidden_layers)]
