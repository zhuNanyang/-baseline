import os
class config(object):
    def __init__(self):
        self.image_path = os.path.join(os.getcwd(), 'photonpic')
        self.learning_rate = 0.001
        self.train_layers = [ 'fc7', 'fc8']
        self.train_dir = './train_dir'
        self.checkpoint_dir = os.path.join(self.train_dir, 'checkpoint')
        self.tensorboard_dir = os.path.join(self.train_dir, 'tensorboard')
        self.tensorboard_train_dir = os.path.join(self.tensorboard_dir, 'train')
        self.path = './photonpic'
        self.epoch = 5000
        self.batch_size = 16
        self.log_step = 10



