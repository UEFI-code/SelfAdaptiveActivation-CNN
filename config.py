#可以根据自己的情况进行修改
class MyConfigs():
    data_folder = '/DataSet/train/'
    num_worker = 4
    model_name = "savemodel" 
    weights = "./checkpoints/"
    logs = "./logs/"
    epochs = 99500
    batch_size = 16
    img_height = 128
    img_width = 128
    num_classes = 4
    lr = 0.001
    lr_decay = 0.000001
    weight_decay = 2e-4
    ratio = 0.3
config = MyConfigs()
