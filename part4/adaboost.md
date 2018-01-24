##  Adaboost

### Boosting

#### Boosting基本原理

Boosting方法就是从弱学习器算法触发，反复学习，得到一系列弱分类器/基本分类器，然后组合这些弱分类器，构成一个强分类器。大多数boosting方法都是改变训练数据的概率分布（训练数据的权值分布），针对不同的训练数据分布调用弱学习器算法学习一些列弱分类器。

* 每一轮如何改变数据的权值或概率分布？

AdaBoosting做法是提高那些被前一轮弱分类器错误分类样本的权值，而降低那些被正确分类样本的权值。则那些没有得到正确分类的数据由于权值加大而受到后一轮的弱分类器的更大关注

* 如何将弱分类器组成成一个强分类器？

AdaBoost采取加权多数表决方法。具体地，加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用，减小无分类误差率大的弱分类器的权值，使其在表决中起较小的作用。

#### Boosting几大参数

Boosting算法族中需要解决的4大问题如下：

* $$e$$—分类误差率？
* $$\alpha$$—弱学习器的权重系数？
* $$D$$—数据权值分布/样本权重？
* 使用何种结合策略？

### AdaBoost

#### AdaBoost二元分类问题算法流程

输入：训练数据集$$T=\{(x_1, y_1),(x_2, y_2),\cdots,(x_N, y_N)\}$$，其中$$x_i \subseteq R^n, y_i \in {-1, +1}$$。假设迭代次数为K。

输出：最终强分类器$$f(x)$$。

1）初始化训练样本集的权重
$$
D(1) = (w_{11}, w_{12},\cdots,w_{1N});\quad w_{1i} = \frac{1}{N};\quad i=1,2,\cdots,m
$$
2）对于$$k=1,2,\cdots,K$$

&ensp;&ensp;a)使用具有权重$$D_k$$的样本集来训练数据，得到弱分类器$$G_k(x)$$：

&ensp;&ensp;b)计算$$G_k(x)$$在训练数据集上的分类误差率
$$
e_k=P(G_k(x_i) \ne y_i)=\sum_{i=1}^N w_{ki}I(G_k(x_i)\ne y_i )
$$
&ensp;&ensp;c)计算弱分类器的系数
$$
\alpha_k = \frac{1}{2}log\frac{1-e_k}{e_k}
$$
&ensp;&ensp;d)更新样本集的权重分布
$$
w_{k+1,i}=\frac{w_{ki}}{Z_k}exp(-\alpha_ky_iG_k(x_i)) \qquad i=1,2,\cdots,m
$$
&ensp;&ensp;这里$$Z_k$$为规范化因子
$$
Z_k = \sum_{i=1}^Kw_{ki}exp(-\alpha_ky_iG_k(x_i))
$$
3）构建最终分类器
$$
f(x) = sign(\sum_{k=1}^K\alpha_kG_k(x))
$$
对于Adaboost多元分类问题，原理和二元分类类似，主要区别是在弱分类器系数上。比如Adaboost SAMME算法，它的弱分类器系数：
$$
\alpha_k = \frac{1}{2}log\frac{1-e_k}{e_k}+log(R-1)
$$
其中，R为类别数。

在上述算法流程 c) 中，$$\alpha_k$$表示弱分类器$$G_k(x)$$在最终分类器中的重要性。当$$e_k \le \frac{1}{2}$$时，$$\alpha_k \ge 0$$，并且$$\alpha_k$$随着$$e_k$$的减小而增大，所以 

> 分类误差率越小的基本分类器在最终分类器中的作用越大。

在步骤 d) 的计算公式中可以看出，如果第i个样本分类错误，则$$y_iG_k(x_i) \lt 0$$，导致样本的权重在第$$k+1$$个弱分类器中增大(即$$w_{k+1, i}$$值增大)；如果分类正确，则$$y_iG_k(x_i) \gt 0$$，则权重在第$$k+1$$个弱分类器中减少。

#### AdaBoost算法的解释

AdaBoost算法

> 模型为加法模型，损失函数为指数函数、学习算法为前向分布算法时的二分类学习方法。

* 模型—加法模型

模型为加法模型好理解，我们的最终的强分类器是若干个弱分类器加权平均而得到的。

* 学习算法—前向学习

AdaBoost算法是通过一轮轮的弱学习器学习，利用前一个弱学习器的记过来更新后一个弱学习器的训练集权重，即第$$k-1$$轮的强学习器为：
$$
f_{k-1}(x) = \sum_{k=1}^{k-1}\alpha_iG_i(x)
$$
而k轮的强学习器为:
$$
f_{k}(x) = \sum_{k=1}^{k}\alpha_iG_i(x)
$$
则上面两公式比较得：
$$
f_k(x) =f_{k-1}(x)+\alpha_kG_k(x)
$$
可见强学习器是通过前向分布学习算法一步步而得到的。

* 损失函数—指数函数

 Adaboost损失函数为指数函数，即定义**损失函数**为：
$$
(\alpha_k, G_k(x)) = arg\underbrace {min}_{\alpha, G}\sum_{i=1}^mexp(-y_i f_k(x))
$$
利用前向分布学习算法的关系，可以得到损失函数为：
$$
(\alpha_k, G_k(x)) = arg\underbrace {min}_{\alpha, G}\sum_{i=1}^mexp(-y_i (f_{k-1}(x)+\alpha G(x)))
$$
令$$w_{ki}^{'} = exp(-y_i f_{k-1}(x))$$，它的值不依赖于$$\alpha, G$$，因此与最小化无关，仅仅依赖于$$f_{k-1}(x)$$，随着每一轮迭代而改变。

带入损失函数得到下式：
$$
(\alpha_k, G_k(x)) = arg\underbrace {min}_{\alpha, G}\sum_{i=1}^mw_{ki}^{'}exp(-y_i\alpha G(x))
$$
首先，求解$$G_k(x)$$，对于任意的$$\alpha \gt 0$$，上式最小的$$G_k(x)$$可以表示为：
$$
G_k(x) = arg\underbrace {min}_{G}\sum_{i=1}^mw_{ki}^{'}I(y_i \ne G(x_i))
$$
将$$G_k(x)$$带入损失函数，并对$$\alpha$$求导，使其等于0：
$$
\begin{aligned}
\sum_{i=1}^mw_{ki}^{'}exp(-y_i\alpha G(x))) \\
&=\sum_{y_i = G_k(x_i)}^mw_{ki}^{'}e^{-\alpha} + \sum_{y_i \ne G_k(x_i)}^mw_{ki}^{'}e^{\alpha} \\
&=e^{-\alpha}(1-e_m) + e^{\alpha}e_m\\
\end{aligned}
$$
求导，等于0：
$$
\alpha_k = \frac{1}{2}log\frac{1-e_k}{e_k}
$$

#### AdaBoost回归问题算法流程

AdaBoost回归算法变种很多，下面的算法为Adaboost R2回归算法过程。

输入：训练数据集$$T=\{(x_1, y_1),(x_2, y_2),\cdots,(x_N, y_N)\}$$，假设迭代次数为K。

输出：最终强分类器$$f(x)$$。

1）初始化训练样本集的权重
$$
D(1) = (w_{11}, w_{12},\cdots,w_{1N});\quad w_{1i} = \frac{1}{N};\quad i=1,2,\cdots,m
$$
2）对于$$k=1,2,\cdots,K$$

&ensp;&ensp;a)使用具有权重$$D_k$$的样本集来训练数据，得到弱分类器$$G_k(x)$$：

&ensp;&ensp;b)计算计算训练集上的最大误差
$$
E_k = max|y_i - G_k(x_i)| \quad i=1,2,\cdots,m
$$
&ensp;&ensp;c)计算每个样本的相对误差：

&ensp;&ensp;&ensp;&ensp;如果是线性误差：$$e_{ki}=\frac{|y_i - G_k(x_i)|}{E_k}$$ 

&ensp;&ensp;&ensp;&ensp;如果是平方误差：$$e_{ki}=\frac{(y_i - G_k(x_i))^2}{E_k}$$ 

&ensp;&ensp;&ensp;&ensp;如果是指数误差：$$e_{ki}=1-exp(\frac{-y_i+G_k(x_i)}{E_k})$$ 

&ensp;&ensp;d)计算回归误差率：
$$
e_k = \sum_{i=1}^mw_{ki}e_{ki}
$$
&ensp;&ensp;e)计算弱学习器的系数：
$$
\alpha_k = \frac{1}{2}log\frac{1-e_k}{e_k}
$$
&ensp;&ensp;f)更新样本集的权重分布：
$$
w_{k+1,i} = \frac{w_{k,i}}{Z_k}\alpha_k^{1-e_{ki}}
$$
&ensp;&ensp;这里$$Z_k$$为规范化因子：
$$
Z_k =\sum_{k=1}^Kw_{ki}\alpha_k^{1-e_{ki}}
$$
3）构造最终学习器为：
$$
f(x) = \sum_{k=1}^K(ln\frac{1}{\alpha_k})G_k(x)
$$

#### AdaBoost小结

理论上任何学习器都可以用于Adaboost.但一般来说，使用最广泛的Adaboost弱学习器是决策树和神经网络。对于决策树，Adaboost分类用了CART分类树，而Adaboost回归用了CART回归树。

Adaboost的主要优点有：

1）Adaboost作为分类器时，分类精度很高

2）在Adaboost的框架下，可以使用各种回归分类模型来构建弱学习器，非常灵活。

3）作为简单的二元分类器时，构造简单，结果可理解。

4）不容易发生过拟合

Adaboost的主要缺点有：

1）对异常样本敏感，异常样本在迭代中可能会获得较高的权重，影响最终的强学习器的预测准确性。

### 参考阅读

[集成学习之Adaboost算法原理小结](http://www.cnblogs.com/pinard/p/6133937.html)



### 