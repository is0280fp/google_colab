#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:56:22 2019

@author: Yume
"""

import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        #親クラスのnn.Moduleのコンストラクタを呼ぶ
        super(Net,self).__init__()
        #畳み込み層を定義する
        #引数は順番に、サンプル数、チャネル数、フィルタのサイズ
        self.conv1=nn.Conv2d(1,6,(5,5))
        #フィルタのサイズは正方形であればタプルではなく整数でも可（8行目と10行目は同じ意味）
        self.conv2=nn.Conv2d(6,16,5)
        #全結合層を定義する
        #fc1の第一引数は、チャネル数*最後のプーリング層の出力のマップのサイズ=特徴量の数
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        #入力→畳み込み層1→活性化関数(ReLU)→プーリング層1(2*2)→出力
        x=F.max_pool2d(F.relu(self.conv1(x)),2)
        #入力→畳み込み層2→活性化関数(ReLU)→プーリング層2(2*2)→出力
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        #ここまでの出力を全結合層へ繋げるために1次元へ展開
        #-1を引数とすることで、self.num_flat_features(x)の値から自動的に値が決まる(ここでは結局1になる)
        x=x.view(-1, self.num_flat_features(x))
        #入力→全結合層1→活性化関数(ReLU)→出力
        x=F.relu(self.fc1(x))
        #入力→全結合層2→活性化関数(ReLU)→出力
        x=F.relu(self.fc2(x))
        #入力→全結合層3→出力
        x=self.fc3(x)
        return x

    def num_flat_features(self,x):
        #Conv2dは入力を4階のテンソルとして保持する(サンプル数*チャネル数*縦の長さ*横の長さ)
        #よって、特徴量の数を数える時は[1:]でスライスしたものを用いる
        size=x.size()[1:]
        #特徴量の数=チャネル数*縦の長さ*横の長さを計算する
        num_features=1
        for s in size:
            num_features*=s
        return num_features
    
    #出力
    output=net(input)
    #教師データ(今回は適当に作成)
    target=Variable(t.arange(1,11))
    #損失関数(平均二乗誤差)のインスタンスを生成
    criterion=nn.MSELoss()
    #損失関数を計算
    loss=criterion(output,target)
    print(loss)
