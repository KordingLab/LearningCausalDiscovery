hyper_parameters = dict(
    lr=dict(type='choice', range=[0.00005, 0.0001, 0.0005, 0.001]),
    train_batch_size=dict(type='choice', range=[256, 512, 1024, 2048, 4096]),
    alpha=dict(type='choice', range=[0.1, 0.25, 0.5, 0.7, 0.9]),
    gamma=dict(type='choice', range=[2, 3, 4, 5, 7, 9]),
    patch_size=dict(type='choice', range=[4, 8, 16, 24]),
)

exp_name = "search_netsim"
n_trials = 32
monitor = dict(metrics=["val_average_precision", "val_auroc"],
               target="val_average_precision",
               direction="max")
