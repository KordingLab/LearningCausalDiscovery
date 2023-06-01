# model settings
num_classes = 2
loss = [dict(type="FocalLoss", alpha=0.7, gamma=3, loss_weight=1.0)]

model = dict(
    arch=dict(
            type="Transformer",
            in_channels=2,
            hidden_size=192*4,
            input_length=30*128,
            embedding_size=192,
            patch_size=32,
            num_layers=8,
            nhead=3,
            num_classes=num_classes,
            losses=loss
    ),
    evaluation=dict(
        metrics=[
            dict(
                type='TorchMetrics',
                metric_name='AveragePrecision',
                prob=True, pos_label=1
            ),
            dict(
                type="TorchMetrics",
                metric_name="AUROC",  # step_reduction='mean',
                prob=True, pos_label=1,
            ),
        ]
    ),
)

# dataset settings
dataset_type = "NMOS6502"

interval = 1

data = dict(
    train_batch_size=1024,  # for single card
    val_batch_size=2048,
    test_batch_size=2048,
    num_workers=8,
    train=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_896_1024/Regular_3510_step_128_rec_30_window_896_1024.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_896_1024/csv/fold_seed_42/train_ds_3.0_unique_True.csv",
        interval=interval,
        shift_range=(-1200, 1200),
    ),
    val=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_768_896/Regular_3510_step_128_rec_30_window_768_896.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_768_896/csv/fold_seed_42/val_ds_1.0_unique_True.csv",
        interval=interval,
    ),
    test=dict(
        type=dataset_type,
        data_root='.cache/sim_data/DonkeyKong/HR/window_640_768/Regular_3510_step_128_rec_30_window_640_768.npy',
        split=".cache/sim_data/DonkeyKong/HR/window_640_768/csv/fold_seed_42/test_ds_1.0_unique_True.csv",
        interval=interval,
    ),
)

log = dict(
    project_name="nmos6502_official_v2",
    work_dir="work_dir",
    exp_name="mos6502_big_transformer_nhead_3_depth_8_dim_192_bs_1024_lr_1e-3",
    logger_interval=50,
    monitor="val_average_precision",
    logger=True,  # use wandb logger
    checkpoint=dict(
        type="ModelCheckpoint",
        top_k=1,
        mode="max",
        verbose=True,
        save_last=True,
    ),
)

resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    type="epoch",
    max_iters=100,
    optimizer=dict(type="AdamW",
                   lr=1e-3,
                   weight_decay=0.05),
    scheduler=dict(
        warmup_steps=10,
        min_lr=0.0,
    ),
)
