# model settings
num_classes = 2
loss = [dict(type="FocalLoss", alpha=0.7, gamma=3, loss_weight=1.0)]

model = dict(
    arch=dict(type="Transformer",
              in_channels=2,
              num_classes=num_classes,
              input_length=200,
              embedding_size=128,
              patch_size=16,
              hidden_size=128*4,
              num_layers=8,
              nhead=4,
              # input_norm=True,
              losses=loss,
    ),
    evaluation=dict(
        metrics=[
            dict(
                type='TorchMetrics',
                metric_name='AveragePrecision',
                prob=True,
                pos_label=1,
            ),
            dict(
                type="TorchMetrics",
                metric_name="AUROC",
                prob=True,
                pos_label=1,
            ),
            dict(
                type="TorchMetrics",
                metric_name="HammingDistance",
                prob=True,
                threshold=0.7,
            ),
        ]
    ),
)

# dataset settings
dataset_type = "NMOS6502"

interval = 10

# dataset settings
dataset_type = "NetSim"
train_dir = [
    ".cache/netsim/sim1.mat",
    ".cache/netsim/sim2.mat",
    ".cache/netsim/sim3.mat",
    ".cache/netsim/sim4.mat",
    ".cache/netsim/sim8.mat",
]

val_dir = [
    ".cache/netsim/sim1.mat",
    ".cache/netsim/sim2.mat",
    ".cache/netsim/sim3.mat",
    ".cache/netsim/sim8.mat",
    ".cache/netsim/sim10.mat",
    ".cache/netsim/sim11.mat",
    ".cache/netsim/sim12.mat",
    ".cache/netsim/sim13.mat",
    ".cache/netsim/sim4.mat",
    ".cache/netsim/sim14.mat",
    ".cache/netsim/sim15.mat",
    ".cache/netsim/sim16.mat",
    ".cache/netsim/sim17.mat",
    ".cache/netsim/sim18.mat",
    ".cache/netsim/sim21.mat",
]

test_dir = [
    ".cache/netsim/sim1.mat",
    ".cache/netsim/sim2.mat",
    ".cache/netsim/sim3.mat",
    ".cache/netsim/sim8.mat",
    ".cache/netsim/sim10.mat",
    ".cache/netsim/sim11.mat",
    ".cache/netsim/sim12.mat",
    ".cache/netsim/sim13.mat",
    ".cache/netsim/sim4.mat",
    ".cache/netsim/sim14.mat",
    ".cache/netsim/sim15.mat",
    ".cache/netsim/sim16.mat",
    ".cache/netsim/sim17.mat",
    ".cache/netsim/sim18.mat",
    ".cache/netsim/sim21.mat",
]


data = dict(
    train_batch_size=256,  # for single card
    val_batch_size=1024,
    test_batch_size=1024,
    num_workers=4,
    train=dict(
        type=dataset_type,
        multiple=True,
        multiple_key="data_root",
        data_root=train_dir,
        percentage=0.6,
        shift_range=[-100, 100]
    ),
    val=dict(
        type=dataset_type,
        multiple=True,
        multiple_key="data_root",
        data_root=val_dir,
        percentage=[0.6, 0.8],
    ),
    test=dict(
        type=dataset_type,
        multiple=True,
        multiple_key="data_root",
        data_root=test_dir,
        percentage=[0.8, 1.0],
    ),
)

log = dict(
    project_name="netsim",
    work_dir="work_dir",
    exp_name="netsim_transformer_hnead_3_depth_8_dim_128_bs_256_lr_5e-4",
    logger_interval=50,
    monitor="val_average_precision",
    logger=True,  # use wandb logger
    checkpoint=dict(
        type="ModelCheckpoint",
        top_k=1,
        mode="max",
        verbose=True,
        save_last=False,
    ),
    earlystopping=dict(
        mode="max",
        strict=False,
        patience=20,
        min_delta=0.0001,
        check_finite=True,
        verbose=True,
    ),
)

resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    type="epoch",
    max_iters=100,
    optimizer=dict(type="AdamW",
                   lr=5e-4,
                   weight_decay=0.05),
    scheduler=dict(
        min_lr=0.0,
    ),
)
