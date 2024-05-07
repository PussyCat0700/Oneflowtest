cfgs = {}

cfgs['image_classification'] = ["ResNet50", "Inception", "SEResNet50", "MobileNet", "ShuffleNet", "DenseNet", "SwinTransformer", "EfficientNet"]
cfgs['token_classification'] = ["BERT-Large", "T5"]
cfgs['llm_sft'] = ["GPT2", "LLaMa-7B"]
cfgs['model_name'] = cfgs['image_classification'] + cfgs['token_classification'] + cfgs['llm_sft']

cfgs['ResNet50'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 256,
    "INPUT_SHAPE" : (32, 3, 256, 256),
    "NUM_CLASSES": 10, # CIFAR-10
}

cfgs['Inception'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 299,
    "INPUT_SHAPE" : (32, 3, 299, 299),
    "NUM_CLASSES": 10, # CIFAR-10
}

cfgs['SEResNet50'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 224,
    "INPUT_SHAPE" : (32, 3, 224, 224),
    "NUM_CLASSES": 10, # CIFAR-10
}

cfgs['MobileNet'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 256,
    "INPUT_SHAPE" : (32, 3, 256, 256),
    "NUM_CLASSES": 10, # CIFAR-10
}

cfgs['ShuffleNet'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 256,
    "INPUT_SHAPE" : (32, 3, 256, 256),
    "NUM_CLASSES": 10, # CIFAR-10
}

cfgs['DenseNet'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 256,
    "INPUT_SHAPE" : (32, 3, 256, 256),
    "NUM_CLASSES": 10, # CIFAR-10
}

cfgs['SwinTransformer'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 224,
    "INPUT_SHAPE" : (32, 3, 224, 224),
    "NUM_CLASSES": 10, # CIFAR-10
} # TODO flowflop

cfgs['EfficientNet'] = {
    "BATCH_SIZE" : 32,
    "IMAGE_SIZE" : 256,
    "INPUT_SHAPE" : (32, 3, 256, 256),
    "NUM_CLASSES": 10, # CIFAR-10
}

cfgs['BERT-Large'] = {
    "BATCH_SIZE" : 32,
    "NUM_CLASSES": 9,
    "TEXT_LENGTH": 56,
    "VOCAB_SIZE": 30522,
    "INPUT_SHAPE" : (32, 56),
}
