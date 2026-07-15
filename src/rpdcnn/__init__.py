import sys

import torch.serialization

from .cfg import RPDCFG, get_default_cfg, get_preset_cfg
from .rpdcnn import RPDCNN

__all__ = ["RPDCFG", "get_default_cfg", "get_preset_cfg", "RPDCNN"]

main_module = sys.modules.get("__main__")
if main_module is not None and not hasattr(main_module, "RPDCFG"):
    main_module.RPDCFG = RPDCFG

torch.serialization.add_safe_globals([(RPDCFG, "__main__.RPDCFG")])
