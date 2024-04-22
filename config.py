cfgs = {}

cfgs['model_name'] = ["ResNet50", "Inception", "SEResNet50", "MobileNet", "ShuffleNet", "DenseNet", "SwinTransformer", "EfficientNet"]

cfgs['ResNet50'] = {
    "BATCH_SIZE" : 64,
    "INPUT_SHAPE" : (64, 3, 256, 256),
    "NUM_CLASSES": 10, # CIFAR-10

}

cfgs['Inception'] = {
    "BATCH_SIZE" : 32,
    "INPUT_SHAPE" : (32, 3, 256, 256),
    "NUM_CLASSES": 10, # CIFAR-10
}