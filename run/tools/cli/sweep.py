import os
import ast
import sys
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

# ugly hack to enable configs inside the package to be run
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# for ray
os.environ["TUNE_ORIG_WORKING_DIR"] = os.getcwd()
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")

from tools.src.model import ModelInterface
from tools.src.data import DataInterface
from argparse import ArgumentParser
from tools.src.utils import load_config
from tools.src.model.utils import OptimizerResumeHook

# hyperparameter search
import ray
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

ray.init(ignore_reinit_error=True, num_cpus=8)
torch.multiprocessing.set_sharing_strategy("file_system")


def configure_arguments():
    """
    Configure the arguments from command line
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--cfg", type=str, help="config file path")
    parser.add_argument("--hp_cfg", type=str, help="config file path for search")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu_ids", default="0", type=str)
    args = parser.parse_args()
    return args


def run(config):
    os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])

    args, hp_config = config["args"], config["hp_config"]
    # device setting
    args.accelerator = "auto"
    # training setting for distributed training
    # args.gpu = '[0, 1, 2, 3]'
    args.gpus = ast.literal_eval(args.gpu_ids)
    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    args.devices = len(args.gpus)
    if args.devices > 1:
        args.sync_batchnorm = True
        args.strategy = "ddp"

    # load config file
    cfg = load_config(args.cfg)

    # ****************************** Set Hyper Parameters from Arguments ******************************** #
    cfg.model.arch.patch_size = hp_config["patch_size"]
    cfg.model.arch.losses[0].alpha = hp_config["alpha"]
    cfg.model.arch.losses[0].gamma = hp_config["gamma"]
    cfg.data.train_batch_size = hp_config["train_batch_size"]
    cfg.optimization.optimizer.lr = hp_config["lr"]
    # *************************************************************************************************** #

    # fix random seed (seed in config has higher priority )
    seed = cfg.seed if cfg.get("seed", None) is not None else args.seed
    pl.seed_everything(seed)

    # ignore warning
    if not cfg.get("warning", True):
        warnings.filterwarnings("ignore")

    # save config file to log directory
    cfg.base_name = args.cfg.split("/")[-1]
    if os.path.exists(os.path.join(cfg.log.work_dir, cfg.log.exp_name)):
        cfg.dump(os.path.join(cfg.log.work_dir, cfg.log.exp_name, cfg.base_name))
    else:
        os.makedirs(os.path.join(cfg.log.work_dir, cfg.log.exp_name))
        cfg.dump(os.path.join(cfg.log.work_dir, cfg.log.exp_name, cfg.base_name))

    # 5 part: model(arch, loss, ) -> ModelInterface /data(file i/o, preprocess pipeline) -> DataInterface
    # /optimization(optimizer, scheduler, epoch/iter...) -> ModelInterface/
    # log(logger, checkpoint, work_dir) -> Trainer /other

    # other setting
    if cfg.get("cudnn_benchmark", True):
        args.benchmark = True

    if cfg.get("deterministic", False):
        if cfg.get("cudnn_benchmark", True):
            print("cudnn_benchmark will be disabled")
        args.deterministic = True
        args.benchmark = False

    # data
    data_module = DataInterface(cfg.data)
    # set ddp sampler
    if args.devices > 1:
        # if cfg.data.train.get('sampler', None) is None:
        args.replace_sampler_ddp = True

    # optimization
    if cfg.optimization.type == "epoch":
        args.max_epochs = cfg.optimization.max_iters
    elif cfg.optimization.type == "iter":
        args.max_steps = cfg.optimization.max_iters
    else:
        raise NotImplementedError(
            "You must choose optimziation update step from (epoch/iter)"
        )

    # for models need setting readout layer with dataloader informatios
    if cfg.model.get("archs", None) is not None:
        for arch in cfg.model.archs:
            if arch.pop("need_dataloader", False):
                data_module.setup(stage="fit")
                arch.dataloader = data_module.train_dataloader()
    else:
        if cfg.model.pop("need_dataloader", False):
            data_module.setup(stage="fit")
            cfg.model.dataloader = data_module.train_dataloader()

    if cfg.get("resume_from", None) is None:
        model = ModelInterface(cfg.model, cfg.optimization)

    # log
    callbacks = []
    if cfg.get("resume_from", None) is not None:
        callbacks.append(OptimizerResumeHook())

    # accumulation of gradients
    if cfg.optimization.get("accumulation_steps", 1) != 1:
        if isinstance(cfg.optimization.accumulation_steps, int):
            callbacks.append(
                plc.GradientAccumulationScheduler(
                    scheduling={0: cfg.optimization.accumulation_steps}
                )
            )
        else:
            # dict of scheduling {epoch: accumulation_steps, ...}
            callbacks.append(
                plc.GradientAccumulationScheduler(
                    scheduling=cfg.optimization.accumulation_steps
                )
            )

    # used to control early stopping
    if cfg.log.earlystopping is not None:
        callbacks.append(
            plc.EarlyStopping(
                monitor=cfg.log.get("monitor", "val_loss"),
                mode=cfg.log.earlystopping.get("mode", "max"),
                strict=cfg.log.earlystopping.get("strict", False),
                patience=cfg.log.earlystopping.get("patience", 5),
                min_delta=cfg.log.earlystopping.get("min_delta", 1e-5),
                check_finite=cfg.log.earlystopping.get("check_finite", True),
                verbose=cfg.log.earlystopping.get("verbose", True),
            )
        )

    # Disable ProgressBar
    callbacks.append(plc.progress.TQDMProgressBar(
        refresh_rate=0,
    ))
    metrics = hp_config["monitor"].metrics
    callbacks.append(TuneReportCallback(metrics, on="validation_end"))
    args.callbacks = callbacks

    # load trainer
    trainer = Trainer.from_argparse_args(args)

    trainer.fit(model, data_module, ckpt_path=cfg.get("resume_from", None))


def sweep():
    args = configure_arguments()
    hp_config = load_config(args.hp_cfg)
    exp_name = hp_config.exp_name
    n_trials = hp_config.n_trials
    cpu_limit = hp_config.get("cpu", 4)
    gpu_limit = hp_config.get("gpu", 1)
    metric = hp_config.monitor

    hp_config = {k: getattr(tune, v["type"])(v["range"]) for k, v in hp_config.hyper_parameters.items()}
    hp_config["monitor"] = metric

    scheduler = ASHAScheduler(
        max_t=load_config(args.cfg).optimization.max_iters,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            run,
            resources={
            "cpu": cpu_limit,
            "gpu": gpu_limit
            }
        ),
        tune_config=tune.TuneConfig(
            metric=hp_config["monitor"].target,
            mode=hp_config["monitor"].direction,
            scheduler=scheduler,
            num_samples=n_trials,
        ),
        run_config=air.RunConfig(
            name=exp_name,
        ),
        param_space={"args": args,
                     "hp_config": hp_config},
    )
    analysis = tuner.fit()

    print("best result:", analysis.get_best_result())


if __name__ == "__main__":
    sweep()
