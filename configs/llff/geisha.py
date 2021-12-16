_base_ = '../default.py'

expname = 'llff_geisha'
basedir = './logs/llff'

data = dict(
    datadir='./data/llff/geisha',
    dataset_type='llff',
    factor=2
)

