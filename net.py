import numpy as np
class NeuralNet(object):
    __activeFunc = ['sigmoid', 'tanh', 'relu']
    __costFunc = ['MS', 'CE']
    __initMethod = ['uniform', 'normal']

    def __init__(self, node, func=[], initMethod='normal', learningRate=0.1, costFunc='MS'):
        self.node = np.array(node)
        self.layerCount = self.node.shape[0]
        self.inDim = self.node[0]  ##输入维度
        self.outDim = self.node[self.layerCount - 1]  ##输出维度
        for layerFunc in func:
            if layerFunc not in NeuralNet.__activeFunc:
                raise ValueError('激活函数错误')
        if len(func)>self.layerCount-1:
            raise  ValueError('激活函数输入数目过多')
        for i in range(len(func),self.layerCount-1):
            func.append('sigmoid')
        self.func=func
        self.initMethod = initMethod
        self.initWandB()
        self.learningRate = learningRate
        if costFunc not in NeuralNet.__costFunc:
            raise ValueError('损失函数错误！')
        else:
            self.costFunc=costFunc

    def initWandB(self):
        if self.initMethod not in NeuralNet.__initMethod:
            raise KeyError(self.initMethod + ':权重初始化方法不存在')
        np.random.seed(0)
        mu = 0
        sigma = 1
        self.weight = []
        self.bias = []
        for i in range(1, self.layerCount):
            size = [self.node[i - 1], self.node[i]]
            layerWeight = np.random.normal(loc=mu, scale=sigma, size=size)
            self.weight.append(layerWeight)
        for i in range(1, self.layerCount):
            layerBias = np.random.normal(loc=mu, scale=sigma, size=self.node[i])
            self.bias.append(layerBias)

    def train(self, input, output, iterNum=100):
        dealInput = np.array(input)
        dealOutput = np.array(output)
        self.CheckInput(input=dealInput)
        if len(dealOutput.shape) != 2:
            raise TypeError('输出值类型错误')
        if dealOutput.shape[1] != self.outDim:
            raise TypeError('输出值类型错误')
        if dealInput.shape[0] != dealOutput.shape[0]:
            raise TypeError("训练数据输入个数和输出个数不一致")
        sampleCount = dealInput.shape[0]
        for iter in range(0,iterNum):
            print('iter:'+str(iter))
            # 对训练样本进行预测
            preOutput = self.predict(input=dealInput, complete=True)
            preMyOutput=np.array([z for z in preOutput[:,self.layerCount-1]])
            costFunc=[]
            if self.costFunc==NeuralNet.__costFunc[0]:
                ##平方损失函数
                costFunc=np.sum(np.power(dealOutput-preMyOutput,2))/(2*sampleCount*self.outDim)
            elif self.costFunc==NeuralNet.__costFunc[1]:
                ##交叉熵损失函数
                costFunc=-np.sum(dealOutput*np.log(preMyOutput)+(1-dealOutput)*np.log(1-preMyOutput))/(sampleCount*self.outDim)
            print('costFunc:'+str(costFunc))
            Gz = []  ##定义损失函数对输出的梯度
            Ga = []  ##定义损失函数对输入的梯度，和对阈值的梯度一样
            Gw = []  ##权重
            for i in range(0, self.layerCount):
                layerIndex = self.layerCount - i - 1
                bGz = []  ##定义损失函数对输出的梯度
                bGa = []  ##定义损失函数对输入的梯度
                bGw = []  ##权重
                # 如果为最后一层
                if layerIndex == self.layerCount - 1:
                    for sampIndex in range(0, sampleCount):
                        ##计算损失函数对输出的导数singleGz
                        preLastOutput = np.array(preOutput[sampIndex, layerIndex])
                        if self.costFunc==NeuralNet.__costFunc[0]:
                            ##平方损失函数的导数
                            sGz=(preLastOutput-dealOutput[sampIndex])/self.outDim
                        elif self.costFunc==NeuralNet.__costFunc[1]:
                            ##交叉熵损失函数的导数
                            sGz = (preLastOutput - dealOutput[sampIndex]) / (preLastOutput * (1 - preLastOutput))
                        bGz.append(sGz)
                        ##计算输出对输入的导数
                        sGza = preLastOutput * (1 - preLastOutput)
                        sGa = sGz * sGza
                        bGa.append(sGa)
                        ##由于sGa为向量
                        zBefore = np.array(preOutput[sampIndex, layerIndex - 1]).reshape(-1, 1)
                        sGw = np.dot(zBefore, sGa.reshape(1, -1))
                        bGw.append(sGw)
                    Gz.append(bGz)
                    Ga.append(bGa)
                    Gw.append(bGw)
                elif layerIndex!=0:
                    for sampIndex in range(0,sampIndex):
                        ##计算对输出的梯度
                        sGz=np.dot(self.weight[layerIndex],Ga[i-1][sampIndex])
                        bGz.append(sGz)
                        ##计算输出对输入的梯度
                        preLastOutput=np.array(preOutput[sampIndex,layerIndex])
                        sGza=preLastOutput*(1-preLastOutput)
                        sGa=sGz*sGza
                        bGa.append(sGa)
                        ##由于sGa为向量
                        zBefore = np.array(preOutput[sampIndex, layerIndex - 1]).reshape(-1, 1)
                        sGw = np.dot(zBefore, sGa.reshape(1, -1))
                        bGw.append(sGw)
                    Gz.append(bGz)
                    Ga.append(bGa)
                    Gw.append(bGw)
                    ##对上一层梯度求和,并更新权重和阈值
                    sumGw=np.sum(Gw[i-1],axis=0)/sampleCount
                    self.weight[layerIndex]-=self.learningRate*sumGw
                    sumGa=np.sum(Ga[i-1],axis=0)/sampleCount
                    self.bias[layerIndex]-=self.learningRate*sumGa
                else:
                    ##对上一层梯度求和,并更新权重和阈值
                    sumGw = np.sum(Gw[i - 1], axis=0)/sampleCount
                    self.weight[layerIndex] -= self.learningRate * sumGw
                    sumGa = np.sum(Ga[i - 1], axis=0)/sampleCount
                    self.bias[layerIndex] -= self.learningRate * sumGa

###layer 为0,1，...，layerCount-1,complete表示是否输出中间结果
    def predict(self, input, layer=-1, complete=False):
        input = np.array(input)
        ##对输入值进行类型检查
        self.CheckInput(input=input)
        ##-1表示最后一层的输出
        if layer == -1:
            layer = self.layerCount - 1
        elif layer > self.layerCount - 1:
            raise ValueError('预测的输出层数超限')
        elif layer == 0:
            output = input
            return output
        ##定义输出
        output = []
        ##遍历输入集合
        for index, inputContent in enumerate(input):
            ##定义内部输出
            interOutput = inputContent
            completeOutput = [inputContent]
            for i in range(1, layer + 1):
                interInput = np.dot(interOutput, self.weight[i - 1]) + self.bias[i - 1]
                ##默认采用sigmoid函数
                interOutput = 1 / (1 + np.exp(-interInput))
                ##记录中间输出值
                if complete:
                    completeOutput.append(interOutput.tolist())
            if complete:
                output.append(completeOutput)
            else:
                output.append(interOutput.tolist())
        return np.array(output)

    def CheckInput(self, input):
        if len(input.shape) != 2:
            raise TypeError('输入值类型错误')
        if input.shape[1] != self.inDim:
            raise TypeError('输入值类型错误')
        pass

neural = NeuralNet([1,5,5, 1],learningRate=1, costFunc='CE')
dataX=np.arange(start=0,stop=2*np.pi,step=0.01)
dataX=dataX.reshape(-1,1)
dataY=np.sin(dataX)

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY=train_test_split(dataX,dataY,test_size=0.2)

##对输入训练集进行白化处理
meanX=np.mean(trainX)
varX=np.std(trainX)
newtrainX=(trainX-meanX)/varX
##对输出训练集进行归一化处理
minY=np.min(trainY)
maxY=np.max(trainY)
newtrainY=(trainY-minY)/(maxY-minY)

neural.train(input=newtrainX,output=newtrainY,iterNum=10000)
newtestX=(testX-meanX)/varX
pretestY=neural.predict(input=newtestX)
newpretestY=pretestY*(maxY-minY)+minY
pretrainY=neural.predict(input=newtrainX)
newpretrainY=pretrainY*(maxY-minY)+minY

import matplotlib.pyplot as plt
plt.subplot(121)
plt.plot(trainX,trainY)
plt.plot(trainX,newpretrainY)
plt.subplot(122)
plt.plot(testX,testY)
plt.plot(testX,newpretestY)



