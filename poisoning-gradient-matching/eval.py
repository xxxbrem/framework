import torch
from torchvision import transforms
from torch import nn
from forest.victims.training import run_validation
from PIL import Image 
import numpy as np
import forest
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()

def normalize(image):

    mean = np.mean(image)
    var = np.mean(np.square(image-mean))
    image = (image - mean)/np.sqrt(var)
    return image



if __name__ == "__main__":

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    model = torch.load(args.model_path)
    criterion = nn.CrossEntropyLoss()
    valid_acc, valid_loss = run_validation(model, criterion, data.validloader, setup)
    print(f"valid_acc: {valid_acc}, valid_loss: {valid_loss}")


    image = Image.open(args.pic_path).convert('RGB')
    image_arr = np.array(image) 
    image_n = normalize(image_arr)
    tensor = transforms.ToTensor()
    img_tensor = tensor(image_n) 
    img_tensor1 = img_tensor.unsqueeze(0) 
    inputs = img_tensor1.to(**setup)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(f"predict: {predicted[0]}, target: 1")