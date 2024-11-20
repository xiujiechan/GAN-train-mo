import os
import sys
import ipdb
import torch as t
import torchvision as tv
import tqdm
from model import NetG, NetD
from torchnet.meter import AverageValueMeter 
from IPython.core.debugger import Pdb
from visualize import Visualizer
ipdb = Pdb()
from PIL import Image


# coding:utf8

# print('sys.path : ',sys.path)
#sys.path.append('/lib/python3.11/site-packages')
#c:\users\user\anaconda3\lib\site-packages

#加入路徑，確保可以找到model.py
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

#配置類定義
class Config(object):
    data_path = r'C:\Users\user\Desktop\GAN 20241117\data\faces'  # 路徑 把偵測好的臉部影像存放在faces 開一個新的data folder 把faces放進去 
    print(f"Using data path: {data_path}") 
    print(f"Contents of data path: {os.listdir(data_path)}")
    num_workers = 4  # 多進程加載數據所用的進程數
    image_size = 96  # 圖片尺寸
    batch_size = 256
    max_epoch = 3000
    lr1 = 2e-4  # 生成器的學習率 
    lr2 = 2e-4  # 判别器的學習率
    beta1 = 0.5  # Adam優化器的beta1參數
    gpu = False  # 是否使用GPU
    nz = 100  # 噪聲維度
    ngf = 64  # 生成器feature map數
    ndf = 64  # 判别器feature map數
    save_path = 'result/'  # 生成圖片保存路徑
    if os.path.exists('result') is False:
        os.mkdir('result')
    vis = False  # 是否使用visdom可視化
    env = 'GAN'  # visdom的env
    plot_every = 2  # 每間隔20 batch，visdom畫圖一次
    debug_file = '/tmp/debuggan'  # 存在該文件則進入debug模式
    d_every = 1  # 每1個batch訓練一次判别器
    g_every = 5  # 每5個batch訓練一次生成器
    save_every = 10  # 每10個epoch保存一次模型
    netd_path = None  # 'checkpoints/netd_.pth' #預訓練模型
    netg_path = None  # 'checkpoints/netg_211.pth'
    # 只測試不訓練
    gen_img = 'result0420.png'
    # 從512張生成的圖片中保存最好的64張
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0  # 噪聲的均值
    gen_std = 1  # 噪聲的方差

opt = Config()
#opt.gpu = True # 確保使用 GPU

# 確保生成器和判別器初始化 
def main(): 
    generator = NetG(opt) 
    discriminator = NetD(opt) 
    print("Generator and Discriminator initialized successfully")

if __name__ == "__main__":
    main()

#訓練函數
def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    if opt.vis:      
        vis = Visualizer(opt.env)       

    # 數據
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    print('~~~dataset : ', dataset)
    dataloader = t.utils.data.DataLoader(dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=opt.num_workers,
                                        drop_last=True
                                        )

    # 網絡
    netg, netd = NetG(opt), NetD(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    netd.to(device)
    netg.to(device)

    # 定義優化器和損失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)


    # 真圖片label為1，假圖片label為0
    # noises為生成網絡的輸入
    true_labels = t.ones(opt.batch_size).to(device)
    fake_labels = t.zeros(opt.batch_size).to(device)
    fix_noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)
    noises = t.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    epochs = range(opt.max_epoch)
    for epoch in iter(epochs):
        for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
            real_img = img.to(device)

            #ipdb.set_trace()  #置斷點

            if ii % opt.d_every == 0:
                # 訓練判别器
                optimizer_d.zero_grad()
                ## 盡可能的把真圖片判别為正確
                output = netd(real_img)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()

                ## 盡可能把假圖片判别為錯誤 (這裡面放了一些非人臉的圖片)
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 根據噪聲生成假圖
                output = netd(fake_img)
                error_d_fake = criterion(output, fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            if ii % opt.g_every == 0:
                # 訓練生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                output = netd(fake_img)
                error_g = criterion(output, true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                ## 可視化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs = netg(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64] * 0.5 + 0.5, win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64] * 0.5 + 0.5, win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])

        if (epoch + 1) % opt.save_every == 0:
            # 保存模型、圖片
            fix_fake_imgs = netg(fix_noises)
            tv.utils.save_image(fix_fake_imgs.data[:64], f'{opt.save_path}/{epoch}.png', 
                                normalize=True,value_range=(-1, 1))
            # 檢查並創建目錄
            os.makedirs('checkpoints', exist_ok=True)
            
            # 保存模型狀態
            t.save(netd.state_dict(), f'checkpoints/netd_{epoch}.pth')
            t.save(netg.state_dict(), f'checkpoints/netg_{epoch}.pth')
            
            # 重置計量器
            errord_meter.reset()
            errorg_meter.reset()


@t.no_grad()     #PyTorch 中的一個裝飾器（Decorator），用來在執行某些操作時禁用自動微分機制。這通常用於推理（inference）階段，即在不需要計算梯度的情況下運行模型，以節省內存和加速運行。在進行模型推理時，不需要計算梯度，只需執行前向傳遞。
def generate(**kwargs):
    """
    随機生成動漫頭像，並根據netd的分數選擇較好的
    """
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda'if opt.gpu else 'cpu')  

    netg, netd = NetG(opt).to(device).eval(), NetD(opt).to(device).eval()
    noises = t.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std)
    noises = noises.to(device)

    map_location = lambda storage, loc: storage
    netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    # 生成圖片，並計算圖片在判别器的分數
    fake_img = netg(noises)
    scores = netd(fake_img).detach()

    # 挑選最好的幾張圖
    indexs = scores.topk(opt.gen_num)[1]
    result = []
    for ii in indexs:
        result.append(fake_img.data[ii])

    # 設定生成圖片的輸出目錄 
    output_dir = "generated_images" 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
        
    # 保存生成的圖片 
    for i, img in enumerate(result): 
        img_path = os.path.join(output_dir, f"generated_image_{i}.png")
        tv.utils.save_image(img, img_path, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    generate()
    #train()
