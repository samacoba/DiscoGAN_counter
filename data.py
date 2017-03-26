# coding: utf-8

import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from numpy import random
from skimage import io
from skimage import draw
from skimage import transform
import math
import pickle


#bokeh用のイメージを取得img = [3][imgH][imgW]
def get_view_img(img):
 
    ch, imgW, imgH= img.shape    
    img = np.clip(img, 0, 255).transpose((1, 2, 0))
    img_plt = np.empty((imgH,imgW), dtype=np.uint32)
    view = img_plt.view(dtype=np.uint8).reshape((imgH, imgW, 4))
    view[:, :, 0:3] = np.flipud(img[:, :, 0:3])#上下反転あり
    view[:, :, 3] = 255
    
    return img_plt

#極大値を取得、in_imgs = ndarray[N][0][imgH][imgW]
def get_local_max_point(in_imgs, threshold = 50):
    m_imgs = chainer.Variable(in_imgs)
    m_imgs = F.max_pooling_2d(m_imgs, ksize=9 ,stride=1, pad=4)
    m_imgs = m_imgs.data
    p_array = (in_imgs == m_imgs)#極大値判定（True or False）の配列
    out_imgs = in_imgs * p_array
    
    out_imgs[out_imgs >= threshold] = 255
    out_imgs[out_imgs < threshold] = 0
    
    return out_imgs

#データの切り出し        
def get_data_N_rand(DataO, N_pic =1, imgH = 256, imgW = 256):
  
    Data={}
    
    #切り出したデータの保存先 dim=[N][0][imgH][imgW] ,float32       
 
    Data['x'] = np.zeros((N_pic, 3, imgH, imgW),dtype= np.float32)
    
    #切り出し限界を設定
    xlim = DataO['x'].shape[3] - imgW + 1
    ylim = DataO['x'].shape[2] - imgH + 1     

    for i in range(0, N_pic):            
        im_num =np.random.randint(0, DataO['x'].shape[0])#切り取る写真の番号
        rotNo = np.random.randint(8) #回転No
        cutx = np.random.randint(0, xlim)
        cuty = np.random.randint(0, ylim)

        Data['x'][i] = rand_rot((DataO['x'][im_num][:, cuty:cuty+imgH, cutx:cutx+imgW]), rotNo)
    
    return Data   
                
#np配列をもらって左右上下の反転・90、180、270°の回転した配列を返す
def rand_rot(a,rotNo):    
    #i = np.random.randint(8)
    i=rotNo
    b = np.zeros_like(a)
    for k in range(0,3):
        if i==0:
            b[k]=a[k]
        elif i==1:
            b[k]=np.fliplr(a[k])
        elif i==2:
            b[k]=np.flipud(a[k])        
        elif i==3:
            b[k]=np.rot90(a[k],1)     
        elif i==4:
            b[k]=np.rot90(a[k],2)        
        elif i==5:
            b[k]=np.rot90(a[k],3)
        elif i==6:
            b[k]=np.fliplr(np.rot90(a[k],1))
        elif i==7:
            b[k]=np.flipud(np.rot90(a[k],1))         
    return b 


#circleを描画 in_imgs = ndarray[N][0][imgH][imgW]
def draw_circle(in_imgs):
        
    cir = np.zeros((1,1,15,15), dtype= np.float32)
    rr, cc = draw.circle_perimeter(7,7,5)
    cir[0][0][rr, cc] = 1      
    decon_cir = L.Deconvolution2D(1, 1, 15, stride=1, pad=7)
    decon_cir.W.data = cir
    out_imgs  = decon_cir(chainer.Variable(in_imgs))
    out_imgs = out_imgs.data
    return out_imgs

#コアを描画、in_imgs = ndarray[N][0][imgH][imgW]
def draw_core(in_imgs, max_xy = 15, sig=3.0):
    
    sig2=sig*sig
    c_xy=7
    core=np.zeros((max_xy, max_xy),dtype = np.float32)
    for px in range(0, max_xy):
        for py in range(0, max_xy):
            r2 = float((px-c_xy)*(px-c_xy)+(py-c_xy)*(py-c_xy))
            core[py][px] = math.exp(-r2/sig2)*1
    core = core.reshape((1,1,core.shape[0],core.shape[1]))
    
    decon_core = L.Deconvolution2D(1, 1, max_xy, stride=1, pad=7)
    decon_core.W.data = core
    out_imgs  = decon_core(chainer.Variable(in_imgs))
    out_imgs = out_imgs.data
    return out_imgs


#点と球状のimageを作成
def get_rand_core(N_pic=1, imgH=512, imgW=512):

    Data={} 

    #ランダムに0.05％の点を作る
    img_p = np.random.randint(0, 10000, size = N_pic*imgW*imgH)
    img_p[img_p < 9995] = 0
    img_p[img_p >= 9995] = 255

    img_p = img_p.reshape((N_pic,1, imgH, imgW)).astype(np.float32)

    #点⇒球に変換
    img = draw_core(img_p)    
    
    img_3ch = np.zeros((N_pic,3, imgH, imgW),dtype = np.float32)
    img_3ch[:,0,:,:] = img
    img_3ch[:,1,:,:] = img  
    img_3ch[:,2,:,:] = img
    
    Data['x'] = img_3ch

    return Data


#xのデータ取得
def get_ori_data_x_1pic(N_pic = 1, imgH = 512, imgW = 512 ,fpath = 'imgA1.png'):
    
    Data={}
    
    Data['x'] = np.zeros((N_pic, 3, imgH, imgW), dtype= np.float32) 
    img = io.imread(fpath)
    img = transform.resize(img,(imgH,imgW),preserve_range=True)    
    #img = img[0:imgH, 0:imgW, :]
    Data['x'][0] = img.astype(np.float32).transpose((2, 0, 1))   
       
    return Data
