import os
import sys
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config.config import get_config
from utils.dataloader import prepare_data

def test(ctrl, data_path, model_path):
    test_x, test_y = prepare_data(data_path.replace('train', 'eval'), ctrl.replace('train', 'eval'))

    test_model = tf.keras.models.load_model(model_path)
    _, acc = test_model.evaluate(test_x, test_y, verbose=2)

    return acc

if __name__ == "__main__":
    config = get_config()
    ctrl = config.ctrl.replace('train', 'eval')
    data_path = config.feat_path + '/eval'
    model_path = config.save_path

    acc = test(ctrl, data_path, model_path)
    with open('result.txt', 'w') as f:
        f.write(str('Accuracy : {:.3f} %'.format(acc * 100)))