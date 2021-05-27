import os

from config.config import get_config

config = get_config()

if __name__ == "__main__":
    step = config.step
    if step < 0 or step > 2:
        raise ValueError("Please enter only between 0 and 2.")

    elif step == 0:
        print('Extracting features...')
        os.system('python ./utils/feature_extractor.py')

    elif step == 1:
        print('Training...')
        os.system('python ./utils/train.py')

    else:
        print('Evaluating...')
        os.system('python ./utils/test.py')