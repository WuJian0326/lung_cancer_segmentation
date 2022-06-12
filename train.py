from models.model import ResNet
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from trainer import trainer
from DataLoader import *
from utils_tool import *
from loss_fn import *
from F1_score import *


lr = 5e-3   #學習率
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 30
num_epoch = 500
num_worker = 4
pin_memory = True
resume = False
train_image_path = 'SEG_Train_Datasets/Mix_858_471/Train_Images/'
train_mask_path = 'SEG_Train_Datasets/Mix_858_471/Train_Mask/'

def main():
    train_transform = get_train_transform() #取得影像增強方式
    vaild_transform = get_vaild_transform() #取得測試影像增強
    #將資料送進dataloader中
    train_data = ImageMaskDataset(train_image_path, train_mask_path, train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, pin_memory=True)

    #建立模型
    model = ResNet().to(device)
    model = nn.DataParallel(model)
    # model = load_checkpoint(model)
    loss_function = Subloss()  #Subloss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96)

    train = trainer(train_loader, model, optimizer, scheduler, loss_function, epochs=num_epoch,best_acc=None)
    #訓練
    model = train.training()

if __name__ == '__main__':
    #訓練
    # main()

    #預測
    model = ResNet().to(device)
    model = load_checkpoint(model, path='checkpoint/ckpt_Mix_resize.pth')
    model.eval()

    vaild_transform = get_vaild_transform()  #取得影像前處理方法

    test_image = 'SEG_Train_Datasets/Test/trainset_input_471_858/'
    test_name = os.listdir(test_image)

    for idx, n in enumerate(tqdm(test_name)):
        path = test_image + n
        img = plt.imread(path)   #讀取影像
        img = vaild_transform(image=img)['image'].unsqueeze(0).to(device)   #影像前處理
        out = F.sigmoid(model(img)).cpu().detach().numpy()[0].transpose(1, 2, 0)    #預測

        #設定閥值
        out[out > 0.5] = 255
        out[out <= 0.5] = 0
        out = cv2.resize(out, (1716, 942))
        #儲存
        cv2.imwrite('SEG_Train_Datasets/Test/output_mask/' + n.replace('jpg', 'png'), out)

    mask = 'SEG_Train_Datasets/Test/output_mask'
    truth = 'SEG_Train_Datasets/Test/traintarget'
    calculate(mask, truth)



