from omegaconf import OmegaConf

# `${len:${data.classes}}` -> number of output channels; 1 when no class list is configured (single binary mask).
OmegaConf.register_new_resolver("len", lambda x: len(x) if x else 1, replace=True)
