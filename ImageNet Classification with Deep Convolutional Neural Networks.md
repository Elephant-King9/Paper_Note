# ImageNet Classification with Deep Convolutional Neural Networks

***

📅 出版年份:2012\ 📖 出版期刊:\ 📈 影响因子:\ 🧑 文章作者:Krizhevsky Alex,Sutskever Ilya,Hinton Geoffrey E

***

## 🔎 摘要:

We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7% and 18.9% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of convolutional nets. To reduce overfitting in the globally connected layers we employed a new regularization method that proved to be very effective.

***

## 🌐 研究目的:

## 📰 研究背景:

## 🔬 研究方法:

***

## 🔩 模型架构:

“The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax.”



***

## 🧪 实验:

###  📇  数据集:

“ILSVRC-2010 and ILSVRC-2012”

“ILSVRC uses a subset of ImageNet with roughly 1000 images in each of 1000 categories. In all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images.”

“Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then cropped out the central 256×256 patch from the resulting image. We did not pre-process the images in any other way, except for subtracting the mean activity over the training set from each pixel. So we trained our network on the (centered) raw RGB values of the pixels.” (Krizhevsky 等, 2012, p. 2)

###  🧠 神经网络创新:

1.“ReLU Nonlinearity” (Krizhevsky 等, 2012, p. 3)



“A four-layer convolutional neural network with ReLUs (solid line) reaches a 25% training error rate on CIFAR-10 six times faster than an equivalent network with tanh neurons (dashed line).”

2.“Training on Multiple GPUs” (Krizhevsky 等, 2012, p. 3)

“Therefore we spread the net across two GPUs. Current GPUs are particularly well-suited to cross-GPU parallelization, as they are able to read from and write to one another’s memory directly, without going through host machine memory.”

3.“Local Response Normalization”



“Response normalization reduces our top-1 and top-5 error rates by 1.4% and 1.2%, respectively. We also verified the effectiveness of this scheme on the CIFAR-10 dataset: a four-layer CNN achieved a 13% test error rate without normalization and 11% with normalization3.” (Krizhevsky 等, 2012, p. 4)

4.“Overlapping Pooling” (Krizhevsky 等, 2012, p. 4)

“This is what we use throughout our network, with s = 2 and z = 3. This scheme reduces the top-1 and top-5 error rates by 0.4% and 0.3%, respectively,”

“We generally observe during training that models with overlapping pooling find it slightly more difficult to overfit.” (Krizhevsky 等, 2012, p. 4)

###  📉 优化器超参数:

“We trained our models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005.”

“We initialized the weights in each layer from a zero-mean Gaussian distribution with standard deviation 0.01. We initialized the neuron biases in the second, fourth, and fifth convolutional layers, as well as in the fully-connected hidden layers, with the constant 1.” “We initialized the neuron biases in the remaining layers with the constant 0.” (Krizhevsky 等, 2012, p. 6)

“The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate. The learning rate was initialized at 0.01 and” (Krizhevsky 等, 2012, p. 6)

“reduced three times prior to termination. We trained the network for roughly 90 cycles” (Krizhevsky 等, 2012, p. 7)

###  ⚠️ 过拟合问题:

1.“Data Augmentation” (Krizhevsky 等, 2012, p. 5)

“The first form of data augmentation consists of generating image translations and horizontal reflections.” (Krizhevsky 等, 2012, p. 5)

“The first form of data augmentation consists of generating image translations and horizontal reflections. We do this by extracting random 224 × 224 patches (and their horizontal reflections) from the 256×256 images and training our network on these extracted patches4” (Krizhevsky 等, 2012, p. 5)

“At test time, the network makes a prediction by extracting five 224 × 224 patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions made by the network’s softmax layer on the ten patches.” (Krizhevsky 等, 2012, p. 5)

“The second form of data augmentation consists of altering the intensities of the RGB channels in training images.” (Krizhevsky 等, 2012, p. 5)

use PCA

“This scheme approximately captures an important property of natural images, namely, that object identity is invariant to changes in the intensity and color of the illumination. This scheme reduces the top-1 error rate by over 1%.” (Krizhevsky 等, 2012, p. 6)

2.“Dropout” (Krizhevsky 等, 2012, p. 6)

“The neurons which are “dropped out” in this way do not contribute to the forward pass and do not participate in backpropagation. So every time an input is presented, the neural network samples a different architecture, but all these architectures share weights. This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on the presence of particular other neurons.” (Krizhevsky 等, 2012, p. 6)

“At test time, we use all the neurons but multiply their outputs by 0.5, which is a reasonable approximation to taking the geometric mean of the predictive distributions produced by the exponentially-many dropout networks.” (Krizhevsky 等, 2012, p. 6)

###  💻  实验设备:

“Our network takes between five and six days to train on two GTX 580 3GB GPUs” (Krizhevsky 等, 2012, p. 2)

###  📊  模型提升:

We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.”

###  📋  实验结果:

“ILSVRC-2010” (Krizhevsky 等, 2012, p. 7)



“ILSVRC-2012” (Krizhevsky 等, 2012, p. 7)



###  📈  定性分析:

“The kernels on GPU 1 are largely color-agnostic, while the kernels on on GPU 2 are largely color-specific.”





###  📉  定量分析:

***

## 🚩 研究结论:

***

## 📝 总结

### 💡 创新点:

提出了一种新的网络AlexNet，在ImageNet上取得了突破性的进步

使用了深层的卷积神经网络

*   ReLU作为非线性激活函数
*   双GPU并行训练
*   使用了局部响应归一化LRN
*   使用了重叠池化

在过拟合的问题上，使用了

*   数据增强

    *   对图像进行四角或中心裁剪
    *   对图像根据PCA主成分分析进行变化

*   使用了Dropout

在得出的结果中定性分析得知，神经网络是真正的学习到了知识，而不是简单的判断

###  ⚠ 局限性:

###  🔧 改进方法:

###  🖍️ 知识补充:

***

## 💬 讨论:
