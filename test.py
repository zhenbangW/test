import torch
import numpy as np
from osgeo import gdal
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import colorsys

def getFileName(filePath, target_dir, i, j):
  fileNameType = filePath.split('/')[-1].split('.')
  fileName = fileNameType[0]+'_'+str(i)+str(j)+'.'+fileNameType[1]
  target_file = os.path.join(target_dir,fileName)
  # path = target_file.split('.')
  # filetype = path[1]
  # name = path[0]
  # target_file = name+str(i)+str(j)+'.'+filetype
  if os.path.exists(target_file) and os.path.isfile(target_file):  # 如果已存在同名影像
    os.remove(target_file)  # 则删除之
  return target_file




def resampling(filePath, target_dir, scale, out_size, oribox, expandTxt):
    """
        影像重采样后裁剪为多幅指定像元大小的影像
        :param filePath: 待缩放源文件路径
        :param target_dir: 输出影像文件夹路径
        :param out_scale: 像元缩放比例
        :param out_size: 输出单幅影像h或w
        :param oribox: 单幅影像原始标签
        :return:
    """
    dataset = gdal.Open(filePath)
    band_count = dataset.RasterCount  # 波段数
    if band_count == 0 or not scale > 0:
        print("参数异常")
        return
    oriCols = dataset.RasterXSize  # 列数
    oriRows = dataset.RasterYSize  # 行数
    cols = int(oriCols * scale)  # 计算新的行列数
    rows = int(oriRows * scale)
    scale = cols/oriCols
    
    geotrans = list(dataset.GetGeoTransform())
    print(dataset.GetGeoTransform())
    print(geotrans)
    geotrans[1] = geotrans[1] / scale  # 像元宽度变为原来的scale倍
    geotrans[5] = geotrans[5] / scale  # 像元高度变为原来的scale倍
    print(geotrans)

    

    band1 = dataset.GetRasterBand(1)
    data_type = band1.DataType
    # target = dataset.GetDriver().Create(target_file, xsize=input_size, ysize=input_size, bands=band_count,
    #                                     eType=data_type)
    # print(cols,rows)
    # target.SetProjection(dataset.GetProjection())  # 设置投影坐标
    # target.SetGeoTransform(geotrans)  # 设置地理变换参数
    # total = band_count + 1
    # for index in range(1, total):
    #     # 读取波段数据
    #     print("正在写入" + str(index) + "波段")
    #     data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
    #     # data = dataset.GetRasterBand(index).ReadAsArray(0,0,cols, rows)
    #     # data = np.flip(data,0)   #翻转图像
    #     #只选取局部像元（得到缩放效果）
    #     # for i in range(cols//input_size):
    #     #   for j in range(rows//input_size):
    #     #     data[:(i+1)*input_size,:]
    #     arrwidth = data.shape[0]//2
    #     arrheight = data.shape[1]//2
    #     data = data[:arrwidth,:arrheight]

    #     out_band = target.GetRasterBand(index)
    #     #out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
    #     out_band.WriteArray(data)  # 写入数据到新影像中
    #     out_band.FlushCache()
    #     out_band.ComputeBandStats(False)  # 计算统计信息
    
    i_strid = math.ceil(cols/out_size)
    j_strid = math.ceil(rows/out_size)

    #只选取局部像元（得到缩放效果）
    for i in range(cols//out_size):
      for j in range(rows//out_size):
        out_file = getFileName(filePath, target_dir, i, j)
        
        #-------------------------------------#
        #  计算仿射变换参数值
        #-------------------------------------#
        top_left_x           = geotrans[0]  #左上角X坐标
        w_e_pixel_resolution = geotrans[1] # 东西方向像素分辨率
        top_left_y           = geotrans[3] # 左上角y坐标
        n_s_pixel_resolution = geotrans[5] # 南北方向像素分辨率
        top_left_x           = top_left_x + i*out_size*w_e_pixel_resolution
        top_left_y           = top_left_y + j*out_size*n_s_pixel_resolution  
        out_geotrans         = (top_left_x,geotrans[1],geotrans[2],top_left_y,geotrans[4],geotrans[5])

        #-------------------------------------#
        #  创建待生成的缩放影像
        #-------------------------------------#
        target = dataset.GetDriver().Create(out_file, xsize=out_size, ysize=out_size, bands=band_count,
                                            eType=data_type)
        target.SetProjection(dataset.GetProjection())  # 设置投影坐标
        target.SetGeoTransform(out_geotrans)  # 设置地理变换参数

        #-------------------------------------#
        #  计算缩放后的影像和样本标签
        #-------------------------------------#
        data = dataset.ReadAsArray(buf_xsize=cols, buf_ysize=rows)
        xmin = i*out_size
        xmax = (i+1)*out_size
        ymin = j*out_size
        ymax = (j+1)*out_size
        tempBox = oribox.copy()
        #计算缩放后样本标签
        box  = resampleBox(tempBox,scale,xmin,xmax,ymin,ymax)
        #通过影像元数据的数组进行裁剪
        data = data[:,j*out_size:(j+1)*out_size,i*out_size:(i+1)*out_size]

        writeAnnotation(out_file,box,expandTxt)

        #写入缩放后影像
        for index in range(1,band_count+1):
          out_band = target.GetRasterBand(index)
          temp = data[index-1]
          out_band.WriteArray(data[index-1])  # 写入数据到新影像中
          out_band.FlushCache()
          out_band.ComputeBandStats(False)  # 计算统计信息

    print("正在写入完成")
    del dataset
    del target


def resampleBox(box,scale,xmin,xmax,ymin,ymax):
  """
    影像重采样后标签数据更新
    :param box: 原始标签数组
    :param scale: 缩放比例
    :param xmin,xmax,ymin,ymax: 裁剪影像后的左上角和右下角
    :return: 缩放更新后的标签数据
  """
  if len(box) == 0 :
    return []
  box[:, [0,2]] = box[:, [0,2]]*scale
  box[:, [1,3]] = box[:, [1,3]]*scale
  #if flip: box[:, [0,2]] = w - box[:, [2,0]]
  # leftP = torch.tensor([xmin,ymin])
  box[:,0][box[:,0]<xmin] = xmin
  box[:,1][box[:,1]<ymin] = ymin
  box[:,2][box[:,2]>xmax] = xmax
  box[:,3][box[:,3]>ymax] = ymax
  box_w = box[:,2] - box[:,0]
  box_h = box[:,3] - box[:,1]
  box = box[np.logical_and(box_w>1, box_h>1)]
  if(len(box)) != 0:
    box[:,[0,2]] = box[:,[0,2]] - xmin
    box[:,[1,3]] = box[:,[1,3]] - ymin
  return box

  # box[:, 0:2][box[:, 0:2]<0] = 0
  # box[:, 2][box[:, 2]>w] = w
  # box[:, 3][box[:, 3]>h] = h
  # box_w = box[:, 2] - box[:, 0]
  # box_h = box[:, 3] - box[:, 1]
  # box = box[np.logical_and(box_w>1, box_h>1)]

# def tif2Rgb(tif_path,rgb_dir):
#   """
#   """
#   image = gdal.Open(tif_path)
#   bands_num = image.RasterCount
#   img_width = image.RasterXSize
#   img_height = image.RasterYSize
#   bands = image.ReadAsArray()
#   image = stretch_16to8(bands)
  
#   image = np.transpose(image,(1,2,0))[:,:,0:3]
#   rgbName = tif_path.split('.')[0]+'.jpg'
#   rgbPath = os.path.join(rgb_dir,rgbName)
#   cv2.imwrite(rgbPath,image)


def tif2Rgb(tif_dir,rgb_dir):
  """
    将tif影像转为rgb图像
    :param tif_dir: 待转换tif的文件夹路径
    :param rgb_dir: 生成rgb图像文件夹路径
    :return: 
  """
  tifList  =[]
  for i in os.listdir(tif_dir):
    if i.endswith('tif'):
      tifList.append(i)

  rgbPaths = []
  for j in tifList:
    tifPath = os.path.join(tif_dir,j)
    image = gdal.Open(tifPath)
    bands_num = image.RasterCount
    img_width = image.RasterXSize
    img_height = image.RasterYSize
    #band1 = image.GetRasterBand(1).ReadAsArray()
    #bands = image.ReadAsArray(int(img_width/2),0,int(img_width/2),int(img_height/2))
    bands = image.ReadAsArray()
    image = stretch_16to8(bands)
    
    image = np.transpose(image,(1,2,0))[:,:,0:3]
    rgbName = j.split('.')[0]+'.jpg'
    rgbPath = os.path.join(rgb_dir,rgbName)
    cv2.imwrite(rgbPath,image)
    rgbPaths.append(rgbPath)
  return rgbPaths


def writeAnnotation(imagePath, box, expandTxt):
  with open(expandTxt,'a',encoding= 'utf-8') as f:
    f.write(imagePath+' ')
    if len(box) != 0:
      for i in box:
        f.write(" " + ",".join([str(a) for a in i[0:4]]) + ',' + str(i[-1]))
    f.write('\n')

def drawBox(boxes,image):
    """
    绘制影像中标签框用于检验结果框是否正确
    :param boxes: 当前影像中所有框(array([[xmin,ymin,xmax,ymax]]))
    :param image: 当前影像
    :return: 
    """
    hsv_tuples = [(x / 1, 1., 1.)for x in range(1)]  # 获得hsv格式的不同色度

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))  # 获得rgb格式的不同颜色

    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))  # 通过hsv格式来调整不同类别对应边框的色度

    #---------------------------------------------------------#
    #   设置字体与边框厚度
    #---------------------------------------------------------#
    font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness   = int(max((image.size[0] + image.size[1]) // np.mean(320), 1))
    
    #---------------------------------------------------------#
    #   图像绘制
    #---------------------------------------------------------#
    for i, c in list(enumerate(boxes)):
        box   = boxes.copy()
        box   = box[i][:4]
        top, left, bottom, right = box

        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
        right   = min(image.size[0], np.floor(right).astype('int32'))
        draw = ImageDraw.Draw(image)
        for i in range(thickness):
            draw.rectangle([top + i, left + i, bottom - i,  right - i], outline=colors[0])
        del draw
    image.show()


def stretch_16to8(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[0]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)
        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out.astype(np.uint8)

# 如果需要进行浮点数除法，则需要float32
# def stretch_16to8(bands, lower_percent=2, higher_percent=98):
#     out = np.zeros_like(bands, dtype=np.float32)
#     n = bands.shape[0]
#     for i in range(n):
#         a = 0  # np.min(band)
#         b = 1  # np.max(band)
#         c = np.percentile(bands[i, :, :], lower_percent)
#         d = np.percentile(bands[i, :, :], higher_percent)
#         t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
#         t[t < a] = a
#         t[t > b] = b
#         out[i, :, :] = t
#     return out.astype(np.uint8)

def preprocess_input(image):
    image /= 255.0
    return image


def clip_by_tensor( t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def MSELoss(pred, target):
    return torch.pow(pred - target, 2)

def BCELoss(pred, target):
    epsilon = 1e-7
    pred    = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

if __name__ == "__main__":
  
  #--------------------------------------------------------------------------------------------------------------------------------#
  #   此处与voc_annotation.py文件中保持一致
  #   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
  #   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1  
  #   仅在annotation_mode为0和1的时候有效
  #--------------------------------------------------------------------------------------------------------------------------------#
  # trainval_percent    = 0.9
  # train_percent       = 0.7

  # a = torch.ones(2,2) 
  # b = torch.tensor([[1,2],[3,4]],dtype=torch.float, requires_grad = True)
  # c = torch.tensor([[-1,2],[3,4]])
  # print(c.clamp(0))
  # d = a + b
  # e = d*5
  # print(e)
  # d.backward(torch.ones_like(d))
  # print(a.grad,b.grad)
  # e.backward(torch.ones_like(e))
  # print(a.grad,b.grad)



  # a = torch.rand((3,4))
  # b = torch.tensor([[0,1,0,0],[1,0,0,0],[0,0,1,0]])
  # # print(a,b)
  # c = BCELoss(a,b)
  # # print(c,torch.mean(c))
  # # # cNum = [2,3,4]
  # cNum = torch.tensor([2,3,4,5]).repeat(a.size()[0],1)
  # # # print(cNum)
  # # # print(torch.max(b,1))
  # # _, obj = torch.max(b,1,keepdim=True)
  # # print(obj,cNum,cNum[:,obj])
  # # print(cNum,b*cNum)
  # balance, _ = (b*cNum).max(1,keepdim=True)
  # balance = balance.float()
  # weigth = (1-0.9)/(1-torch.pow(0.9,balance))
  # print(c,c*balance,torch.mean(c*weigth))

  # cNum1 = torch.tensor([2,3,4,5]).repeat(3,2,3,1)
  # a = a.repeat(3,2,1,1)
  # b = b.repeat(3,2,1,1)
  # cNum1 = torch.tensor([2,3,4,5]).repeat(3,2,3,1)
  # c = BCELoss(a,b)
  # balance, _ = (b*cNum1).max(-1,keepdim=True)
  # balance = balance.float()
  # weigth = (1-0.9)/(1-torch.pow(0.9,balance))
  
  # print(c,weigth,c*weigth,torch.mean(c*weigth))

  # sample_path = r'F:\Study\毕业论文\testgit\yolo3-pytorch\2007_train.txt'
  # classNum = 3
  # classes = [x for x in range(classNum)]
  # classNums = [0 for x in range(classNum)]
  # with open(sample_path,encoding= 'utf-8') as f:
  #   lines = f.readlines()
  #   for line in lines:
  #     sample = line.split()[1:]
  #     for num in sample:
  #       classtype = num.split(',')[-1]
  #       classNums[int(classtype)]+=1
  # print(torch.tensor(classNums))





######
  # expand_num = 10
  # expand_tif_dir = r'F:\temp2\test\ExpandImage\TIFImage'
  # expand_rgb_dir = r'F:\temp2\test\ExpandImage\RGBImage'
  # expandTxt      = r'F:\temp2\test\ExpandImage\expand.txt'

  # txtPath = r'F:\temp2\test\2007_train.txt'

  
  # with open(txtPath,encoding = 'utf-8') as f:
  #   sampleLine = f.readlines()
  # boxes = []
  # for line in sampleLine:
  #   line = line.split()
  #   box  = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
  #   resampling(line[0],expand_tif_dir,2.2,320,box,expandTxt)
    

  # rgbPaths = tif2Rgb(expand_tif_dir,expand_rgb_dir)
####

  # for rgb_path in rgbPaths:
  #   image    = Image.open(rgb_path)
  #   drawBox(box,image)


  # num = len(sampleLine)
  # sampleList = range(num)
  # tr = int(num*)



  # with open(txtPath,'a',encoding='utf-8') as f:
  #   f.write('123 231\n')


  # xmlStr = '107,276,156,320,0 158,289,206,320,0'
  # line      = xmlStr.split()
  # box       = np.array([np.array(list(map(int,box.split(',')))) for box in line[0:]])
  # image = Image.open(r'F:\temp2\test\ExpandImage\RGBImage\1239_01.jpg')
  # drawBox(box,image)

  # input_path = r'F:\temp2\tifImage\0726.tif'
  # tif_dir   = r'F:\temp2'
  # rgb_dir   = r'F:\temp2'
  # xmlStr = '156,267,188,301,0 176,240,204,269,0 196,215,226,245,0 197,175,226,204,0 202,146,230,173,0 231,140,252,163,0 226,164,252,193,0 223,195,244,215,0 215,92,244,123,0 246,88,277,119,0 279,92,308,122,0 306,95,320,122,0 250,56,281,87,0 269,78,286,96,0 281,59,313,90,0 233,16,264,48,0 265,10,295,42,0 296,7,320,47,0 252,2,272,20,0 230,1,261,14,0 263,1,292,9,0'
  # line      = xmlStr.split()
  # box       = np.array([np.array(list(map(int,box.split(',')))) for box in line[0:]])
  
  # box       =resampling(input_path,tif_dir,2.2,320,box)
  # tif2Rgb(tif_dir,rgb_dir)
  # image = Image.open(r'F:\temp2\0726_11.jpg')
  # drawBox(box,image)

  # path = r'F:\Study\毕业论文\testgit\yolo3-pytorch\img\1.jpg'
  # image = Image.open(path)
  # # image.show()
  # b = gdal.Open(path).ReadAsArray()
  # a = np.array(image)




  
 


  # tif2Rgb(tif_path,rgb_path)
  # path = r'F:\Study\毕业论文\testgit\yolo3-pytorch\VOCdevkit\temp1.tif'
  # dataset = gdal.Open(path)
  # imageArray = dataset.ReadAsArray()
  # array = stretch_16to8(imageArray)
  # array = preprocess_input(array)
  # print(array.max(),array.min(),len(array==array.max()),len(array==array.min()))

  # path = 'VOCdevkit/VOC2007/JPEGImages/'+'0726.jpg'
  # img = Image.open(path)
  # # size = img.size
  # # w = size[0]*2
  # # img = img.resize((w,size[1]),Image.BICUBIC)
  # img.show()
  # image = np.array(img)

  # a = torch.linspace(1,2,2).repeat(3,1)
  # b = a.index_select(1,torch.tensor(0))
  # c = a[:,[0]]
  # print(a,b,c)
 



  pass





  # oriImg = gdal.Open(input_path)
  # img = gdal.Open(out_path)
  # pass
  

    
    