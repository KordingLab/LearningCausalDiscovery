# model settings
num_classes = 2
loss = [dict(type="FocalLoss", alpha=0.7, gamma=3, loss_weight=1.0)]

model = dict(
    arch=dict(
        type="Transformer",
        in_channels=2,
        num_classes=num_classes,
        input_length=966,
        embedding_size=64,
        patch_size=21,
        hidden_size=64,
        num_layers=2,
        nhead=4,
        input_norm=True,
        losses=loss,
    ),
    evaluation=dict(
        metrics=[
            dict(
                type='TorchMetrics',
                metric_name='AveragePrecision',
                prob=True, num_classes=2
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
dataset_type = "Dream3"

train_dir = [('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli1.pt',
             '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli1.tsv'),
             ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast1.pt',
              '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast1.tsv')
             ]

val_dir = ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli2.pt',
             '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv')

test_dir = ('/project/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli2.pt',
            '/project/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv')


data = dict(
    train_batch_size=128,  # for single card
    val_batch_size=512,
    test_batch_size=512,
    num_workers=0,
    train=dict(
        type=dataset_type,
        multiple=True,
        multiple_key="data_root",
        data_root=train_dir,
        shift_range=[-322, 322]
    ),
    val=dict(
        type=dataset_type,
        data_root=val_dir,
    ),
    test=dict(
        type=dataset_type,
        data_root=test_dir,
    ),
)

log = dict(
    project_name="nmos6502_official",
    work_dir="work_dir",
    exp_name="dream3_ecoli_transformer_128",
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
    # earlystopping=dict(
    #     mode="max",
    #     strict=False,
    #     patience=5,
    #     min_delta=0.0001,
    #     check_finite=True,
    #     verbose=True,
    # ),
)

resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    type="epoch",
    max_iters=100,
    optimizer=dict(type="AdamW",
                   lr=0.0005,
                   weight_decay=0.05),
    scheduler=dict(
        min_lr=0.0,
    ),
)
