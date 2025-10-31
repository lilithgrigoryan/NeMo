import os

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import clean_exp_ckpt, exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from nemo.collections.asr.models.fastconformer_ctc_with_adapter_model import FastConformerCTCWithAdapterModel

@hydra_runner(config_path="conf", config_name="ctc_alignment_adapter")
def main(cfg: DictConfig):
    logging.info(f'Config loaded:\n{OmegaConf.to_yaml(cfg)}')
    # import pdb; pdb.set_trace()
    print(cfg)

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    model = FastConformerCTCWithAdapterModel(cfg.model, trainer=trainer)

    if 'train_ds' in cfg.model:
        model.setup_training_data(cfg.model.train_ds)
    if 'validation_ds' in cfg.model and cfg.model.validation_ds is not None:
        model.setup_multiple_validation_data(cfg.model.validation_ds)
    if 'optim' in cfg.model and cfg.model.optim is not None:
        model.setup_optimization(cfg.model.optim)

    trainer.fit(model)    
    

    # adapter_cfg = cfg.model.get('adapter', None)
    # if adapter_cfg is not None:
    #     adapter_state_dict_name = adapter_cfg.get('adapter_state_dict_name', None)
    #     if adapter_state_dict_name:
    #         state_path = exp_dir if exp_dir is not None else os.getcwd()
    #         ckpt_path = os.path.join(state_path, "checkpoints")
    #         if os.path.exists(ckpt_path):
    #             state_path = ckpt_path
    #         state_path = os.path.join(state_path, adapter_state_dict_name)
    #         model.save_adapters(str(state_path))

    # if getattr(cfg, 'delete_ckpt_after_train', False):
    #     clean_exp_ckpt(exp_dir, remove_ckpt=True, remove_nemo=False)


if __name__ == '__main__':
    main()
