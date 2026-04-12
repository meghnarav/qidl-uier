class Config:
    img_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 30
    num_workers = 4
    seed = 42

    base_ch = 64
    n_res = 4

    device = "cuda"