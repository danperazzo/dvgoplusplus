_base_ = '../default.py'

expname = 'llff_pinecone'
basedir = './logs/llff'

data = dict(
    datadir='./data/llff/pinecone',
    dataset_type='llff',
    factor=1
)

