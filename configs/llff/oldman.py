_base_ = '../default.py'

expname = 'llff_oldman'
basedir = './logs/llff'

data = dict(
    datadir='./data/llff/oldman',
    dataset_type='llff',
    factor=1
)


fine_train = dict(
    N_iters=10000,                # number of optimization steps
    N_rand=2048,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    pervoxel_lr=True,             # view-count-based lr
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.01,     # weight of background entropy loss
    weight_rgbper=0.1,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_from=0,                    # count total variation loss from tv_from step
    weight_tv_density=0.0,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,             # weight of total variation loss of color/feature voxel grid
    pg_scale=[],                  # checkpoints for progressive scaling
)
