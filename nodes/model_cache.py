"""Shared model cache for Fish Audio S2 nodes."""

import gc
import logging
from typing import Any

import torch

logger = logging.getLogger("FishAudioS2")

# Module-level cache: keyed by (model_path, device, precision)
_cached_engine: Any = None
_cached_key: tuple = ()


def get_cache_key(model_path: str, device: str, precision: str, attention: str) -> tuple:
    return (model_path, device, precision, attention)


def get_cached_engine():
    return _cached_engine, _cached_key


def set_cached_engine(engine: Any, key: tuple):
    global _cached_engine, _cached_key
    _cached_engine = engine
    _cached_key = key


def unload_engine():
    global _cached_engine, _cached_key
    if _cached_engine is not None:
        logger.info("Unloading Fish S2 model from memory...")
        try:
            # Stop the llama queue thread if running
            engine = _cached_engine
            if hasattr(engine, "llama_queue"):
                engine.llama_queue.put(None)  # sentinel to stop thread
        except Exception:
            pass
        del _cached_engine
        _cached_engine = None
        _cached_key = ()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model unloaded and VRAM freed.")
