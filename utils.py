import torchvision
from torchvision import transforms
import torch



def get_test_image(D,num=10,PATH = "./imgnet-small"):
    TRANSFORM_IMG = transforms.Compose([
        torchvision.transforms.Resize(128),
        transforms.CenterCrop(D),
        transforms.ToTensor(),
        ])
    test_data = torchvision.datasets.ImageFolder(root=PATH, transform=TRANSFORM_IMG)
    img_loader = torch.utils.data.DataLoader(test_data, batch_size=num,shuffle = False)
    for i in img_loader:
        img_batch=(i[0]*255).to(torch.int32)
        break
    return img_batch
