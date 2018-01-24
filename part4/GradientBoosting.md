## GBDT 概述
###GBDT V.S. AdaBoost

GBDT也是集成学习Boosting家族的成员，但是却和传统的AdaBoost有很大不同。

* AdaBoost利用前一轮迭代弱学习器的误差率来更新训练集的权重，这样一轮轮的迭代下去。AdaBoost是按分类的对错，分配不同的weight，并且在计算cost function的时候使用这些weight。从而将"错分类的样本权值越来越大，越被重视"。
* Boostrap也有类似思想，在每一步迭代时**不改变模型本身，也不计算残差**，而是从N个样本训练集中按照一定概率重新抽取N个样本出来，并对这N个新样本再训练一轮。训练集变了，迭代模型训练结果也不一样，对于被分错的样本，sample概率就越高，这样就能同样达到逐步关注被分错的样本，达到逐步完善的效果。
* GBDT也是迭代，但是弱学习器限定了只能使用**CART回归树模型**。假设前一轮迭代得到的弱学习器是$$f_{t-1}(x)$$，损失函数是$$L(y, f_{t-1}(x))$$，本轮迭代目标是找到一个CART树的若学习器$$h_t(x)$$，让本轮的损失函数$$L(y, f_t(x)) = L(y, f_{t-1}(x)+h_t(x))$$最小，也就是本轮迭代找到的决策树，要让样本损失尽量变得更小。
* GBDT也可以在使用残差的同时引入Boostrap re-sampling，GBDT多数实现版本中也增加了这个选型，但是否一定使用则有很多不同观点，re-sampling一个缺点是其随机性，即同样的数据集合训练两边结果是不一样的，也就是模型不可能稳定复现，这对评估是很大的挑战。


### Gradient Boosting

基于boosting框架的整体模型可以用线性组成式来描述，其中$$h_i(x)$$为基模型与其权值的乘积:
$$
F(x) = \sum_{i}^mh_i(x)
$$
根据上式，整体模型的训练目标是使得预测值F(x)逼近真实值y，也就是说要让每一个基模型的预测值逼近各自要预测的部分真实值。由于要同时考虑所有基模型，导致了整体模型的训练变成了一个非常复杂的问题。所以，研究者们想到了一个贪心的解决手段：每次只训练一个基模型。那么，现在改写整体模型为迭代式：
$$
F^i(x)=F^{i-1}(x)+h_i(x)=F^{i-1}(x)+\underbrace{arg min}_{f \in H}\sum_{i=1}^nL(y_i, F^{i-1}(x)+f(x))
$$
这样一来，在每一轮迭代中，只要集中解决一个基模型的训练问题：使得$$F^i(x)$$逼近真实值y。



## GBDT负梯度拟合

Freidman提出了用损失函数的负梯度来拟合本轮损失的近似值，进而拟合一个CART回归树。第t轮的第i个样本的损失函数的负梯度表示为：
$$
r_{ti}=-[\frac{\partial L(y, f(x_i))}{\partial f(x_i)}]_{f(x)=f_{t-1}(x)}
$$
利用$$(x_i, r_{ti})\quad (i=1,2,\cdots, m)$$，我们可以拟合一颗CART回归树，得到了第t颗回归树，其对应的叶节点区域为$$R_{tj},j=1,2,\cdots,J$$。其中J为叶子节点的个数。

针对每一个叶子节点里的样本$$(j=1,2,\cdots,J)$$，我们求出使得损失函数最小，也就是拟合叶子节点最好的输出值$$c_{tj}$$如下：
$$
c_{tj}=\underbrace{arg min}_c\sum_{x_i \in R_{tj}}L(y_i, f_{t-1}(x_i)+c)
$$
从而本轮最终得到的强学习器的表达式如下：
$$
f_t(x)=f_{t-1}(x)+\sum_{j=1}^Jc_{tj}I(x \in R_{tj})
$$
通过损失函数的负梯度来拟合，我们找到了一个通用的拟合损失函数的办法，这样无论是分类问题还是回归问题，我们通过其损失函数的负梯度的拟合，就可以用GBDT来解决分类回归问题。区别仅仅在于损失函数的不同导致负梯度的不同而已。

## GBDT回归算法

输入是输入样本$$T={(x_1, y_1),(x_2, y_2),\cdots,(x_m, y_m)}$$，最大迭代次数T，损失函数L。

输出是强学习器$$f(x)$$。

1) 初始化弱学习器
$$
f_0(x) = \underbrace{argmin}_c\sum_{i=1}^mL(y_i, c)
$$
2) 对迭代论述$$t=1,2,\cdots,T$$有：

​		a) 对样本$$i=1,2,\cdots,m$$,计算负梯度
$$
r_{ti}=-[\frac{\partial L(y, f(x_i))}{\partial f(x_i)}]_{f(x)=f_{t-1}(x)}
$$
​		b) 利用$$(x_i, r_{ti})\quad (i=1,2,\cdots, m)$$，我们可以拟合一颗CART回归树，得到了第t颗回归树，其对应的叶节点区域为$$R_{tj},j=1,2,\cdots,J$$。其中J为叶子节点的个数。

​		c) 对叶子区域$$j=1,2,\cdots,J$$，计算最佳拟合值:
$$
c_{tj}=\underbrace{arg min}_c\sum_{x_i \in R_{tj}}L(y_i, f_{t-1}(x_i)+c)
$$
​		d) 更新强学习器
$$
f_t(x)=f_{t-1}(x)+\sum_{j=1}^Jc_{tj}I(x \in R_{tj})
$$
3) 得到强学习器$$f(x)$$的表达式：
$$
f(x)=f_T(x)=f_0(x)+\sum_{t=1}^T\sum_{j=1}^Jc_{tj}I(x \in R_{tj})
$$

## GBDT分类算法

GBDT的分类算法从思想上和GBDT回归算法没有区别，但是由于样本输出不是连续的值，而是离散的类别，导致我们无法直接从输出类别去拟合类别输出的误差。

为了解决这个问题，主要有两个方法：

* 一个是用指数损失函数，此时GBDT退化为Adaboost算法。
* 另一种方法是用类似于逻辑回归的对数似然损失函数的方法。也就是说，我们用的是类别的预测概率值和真实概率值的差来拟合损失。

本文仅讨论用对数似然损失函数的GBDT分类。而对于对数似然损失函数，我们又有二元分类和多元分类的区别。

### 二元GBDT分类算法

对于二元GBDT，如果使用类似于逻辑回归的对数似然损失函数，则损失函数为:
$$
L(y, f(x))=log(1+exp(-yf(x)))\qquad y \in \{-1, +1\}
$$
此时负梯度误差为：
$$
r_{ti} = -\bigg[\frac{\partial L(y, f(x_i)))}{\partial f(x_i)}\bigg]_{f(x) = f_{t-1}\;\; (x)} =\frac{ y_i}{(1+exp(y_if(x_i))}
$$
对于生成的决策树，我们各个叶子节点的最佳残差拟合值为：
$$
c_{tj} = \underbrace{arg\; min}_{c}\sum\limits_{x_i \in R_{tj}} log(1+exp(-y_i(f_{t-1}(x_i) +c)))
$$
由于上式比较难优化，我们一般使用近似值代替：
$$
c_{tj} = \sum\limits_{x_i \in R_{tj}}r_{ti}\bigg /  \sum\limits_{x_i \in R_{tj}}|r_{ti}|(1-|r_{ti}|)
$$
除了负梯度计算和叶子节点的最佳残差拟合的线性搜索，二元GBDT分类和GBDT回归算法过程相同。

### 多元GBDT分类算法

多元GBDT要比二元GBDT复杂一些，对应的是多元逻辑回归和二元逻辑回归的复杂度差别。假设类别数为K，则此时我们的对数似然损失函数为：
$$
L(y, f(x)) = -  \sum\limits_{k=1}^{K}y_klog\;p_k(x)
$$
其中如果样本输出类别为k，则$$y_k=1$$。第k类的概率$$p_k(x)$$的表达式为：
$$
p_k(x) = exp(f_k(x)) \bigg / \sum\limits_{l=1}^{K} exp(f_l(x))
$$
集合上两式，我们可以计算出第tt轮的第$$i$$个样本对应类别$$l$$的负梯度误差为
$$
r_{til} = -\bigg[\frac{\partial L(y, f(x_i)))}{\partial f(x_i)}\bigg]_{f_k(x) = f_{l, t-1}\;\; (x)} = y_{il} - p_{l, t-1}(x_i)
$$
观察上式可以看出，其实这里的误差就是样本ii对应类别ll的真实概率和$$t−1$$轮预测概率的差值。

对于生成的决策树，我们各个叶子节点的最佳残差拟合值为：
$$
c_{tjl} = \underbrace{arg\; min}_{c_{jl}}\sum\limits_{i=0}^{m}\sum\limits_{k=1}^{K} L(y_k, f_{t-1, l}(x) + \sum\limits_{j=0}^{J}c_{jl} I(x_i \in R_{tj})
$$
由于上式比较难优化，我们一般使用近似值代替：
$$
c_{tjl} =  \frac{K-1}{K} \; \frac{\sum\limits_{x_i \in R_{tjl}}r_{til}}{\sum\limits_{x_i \in R_{til}}|r_{til}|(1-|r_{til}|)}
$$
除了负梯度计算和叶子节点的最佳残差拟合的线性搜索，多元GBDT分类和二元GBDT分类以及GBDT回归算法过程相同。

## XGBOOST

Xgboost是GB算法的高效实现，xgboost中的基学习器除了可以是CART树也可以是线性分类器。

* xgboost在目标函数中显式的增加了正则化项，基学习为CART时，正则化项与树的叶子节点的数量T和叶子节点的权值有关。

$$
L(\phi)=\sum_{i}l(\hat y_i, y_i)+\sum_k \Omega(f_k) \\
where \; \Omega(f) = \gamma T+\frac{1}{2}\lambda ||w||^2
$$

* GB中使用的Loss Function 对$$f(x)$$的一阶导数计算出的残差用于学习生成$$f_t(x)$$,xgboost则使用了二阶导数。第t轮的Loss Function。

$$
L^{(t)}=\sum_{i=1}^nl(y_i, \hat y_i^{(t-1)}+f_t(x_i))+\Omega(f_t)
$$

对上式做二阶泰勒展开：g为一阶导数，h为二阶导数
$$
L^{(t)} \cong \sum_{i=1}^n[l(y_i, \hat y^{(t-1)})+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)\\
where \; g_i =\frac{\partial l(y_i, \hat y_i^{(t-1)})}{\partial \hat y_i^{(t-1)}}\;, h_i =\frac{\partial ^2l(y_i, \hat y_i^{(t-1)})}{\partial \hat y_i^{(t-1)}}
$$

* xgboost寻找分割点的标准是最大化，枚举可行的分割点，选择增益最大的划分，继续同样的操作，直到满足某阈值或得到纯节点。

$$
L_{split}=\frac{1}{2}[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}]-\gamma \\
where\; G_L=\sum_{i\in I_L}g_i \; G_R=\sum_{i\in I_R}g_i \;H_L=\sum_{i\in I_L}h_i\;H_R=\sum_{i\in I_R}h_i
$$

xgboost与GBDT除了上述三点不同之外，xgboost在实现上还做了许多优化：

1. 在寻找最佳分割点时，考虑传统的枚举每个特征的所有可能分割点的贪心法效率太低，xgboost实现了一种近似的算法。大致的思想是根据百分位法列举几个可能成为分割点的候选者，然后从候选者中根据上面求分割点的公式计算找出最佳的分割点。
2. xgboost考虑了训练数据为稀疏值的情况，可以为缺失值或者指定的值指定分支的默认方向，这能大大提升算法的效率，paper提到50倍。
3. 特征列排序后以块的形式存储在内存中，在迭代中可以重复使用；虽然boosting算法迭代必须串行，但是在处理每个特征列时可以做到并行。
4. 按照特征列方式存储能优化寻找最佳的分割点，但是当以行计算梯度数据时会导致内存的不连续访问，严重时会导致cache miss，降低算法效率。paper中提到，可先将数据收集到线程内部的buffer，然后再计算，提高算法的效率。
5. xgboost 还考虑了当数据量比较大，内存不够时怎么有效的使用磁盘，主要是结合多线程、数据压缩、分片的方法，尽可能的提高算法的效率。



## GBDT小结

GBDT主要的优点有：

1) 可以灵活处理各种类型的数据，包括连续值和离散值。

2) 在相对少的调参时间情况下，预测的准备率也可以比较高。这个是相对SVM来说的。

3) 使用一些健壮的损失函数，对异常值的鲁棒性非常强。比如 Huber损失函数和Quantile损失函数。

GBDT的主要缺点有：

4) 由于弱学习器之间存在依赖关系，难以并行训练数据。不过可以通过自采样的SGBT来达到部分并行。

 

## 参考阅读

[GBDT（MART） 迭代决策树入门教程](http://blog.csdn.net/w28971023/article/details/8240756)

[使用sklearn进行集成学习——理论](http://www.cnblogs.com/jasonfreak/p/5657196.html)

[梯度提升树(GBDT)原理小结](http://www.cnblogs.com/pinard/p/6140514.html)

[GB, XGBOOST](http://blog.csdn.net/shenxiaoming77/article/details/51542982)