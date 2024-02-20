# from typing import Literal


# import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm


from args import ArgParse
from logger import configure_logger_pl
# from callback import configure_callbacks
from dataset import TrainValDataModule
from model import SimpleLightningModel


# SupportedStrategy = Literal["auto", "dp", "ddp"]


def main():
    args = ArgParse.get()

    loggers, exp_name = configure_logger_pl(
        model_name=args.model_name,
        disable_logging=args.disable_comet,
        save_dir=args.comet_log_dir,
    )
    # callbacks = configure_callbacks()

    data_module = TrainValDataModule(
        command_line_args=args,
        dataset_name=args.dataset_name,
    )
    model_lightning = SimpleLightningModel(
        command_line_args=args,
        n_classes=data_module.n_classes,
        exp_name=exp_name
    )

    # strategy: SupportedStrategy = "auto"
    # if args.gpu_strategy == 'dp':
    #     strategy = 'dp'
    # if args.gpu_strategy == "ddp":
    #     strategy = "ddp"

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    trainer = pl.Trainer(
        # devices=args.gpus,
        # accelerator="auto",
        # strategy=strategy,
        max_epochs=args.num_epochs,
        logger=loggers,
        log_every_n_steps=args.log_interval_steps,
        accumulate_grad_batches=args.grad_accum,
        num_sanity_val_steps=0,
        # precision=16,  # for FP16 training
        # fast_dev_run=True, # only for debug
        # fast_dev_run=5, # only for debug
        # limit_train_batches=5, # only for debug
        # limit_val_batches=5, # only for debug
        # callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
        # profiler="simple",
    )

    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=args.checkpoint_to_resume,
    )


if __name__ == "__main__":
    main()
