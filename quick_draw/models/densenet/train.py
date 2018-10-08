from quick_draw.models.densenet.model import model_fn
from quick_draw.models.train import train

if __name__ == '__main__':
    train('densenet', model_fn)
