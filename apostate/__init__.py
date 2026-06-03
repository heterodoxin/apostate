
import warnings as _w
import os as _os
import logging as _log

_w.filterwarnings("ignore", message=".*urllib3.*")
_w.filterwarnings("ignore", message=".*chardet.*")
_w.filterwarnings("ignore", message=".*charset_normalizer.*")
_w.filterwarnings("ignore", category=UserWarning, module="requests")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _n in ("huggingface_hub", "urllib3", "datasets", "transformers"):
    _log.getLogger(_n).setLevel(_log.ERROR)


def _text_only():
    if _os.environ.get("APOSTATE_ALLOW_TORCHVISION", "").lower() in ("1", "true", "yes"):
        return
    try:
        import transformers.utils as _tu
        import transformers.utils.import_utils as _iu

        def _false():
            return False

        _iu.is_torchvision_available = _false
        _iu.is_torchvision_v2_available = _false
        _tu.is_torchvision_available = _false
        _tu.is_torchvision_v2_available = _false
    except Exception:
        pass


_text_only()

from .config import ApostateConfig

__version__ = "0.1.0"
__all__ = ["ApostateConfig"]
