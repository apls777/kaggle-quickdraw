from quick_draw.models.simple_cnn.model import model_fn
from quick_draw.models.train import train

if __name__ == '__main__':
    train('simple_cnn', model_fn)
