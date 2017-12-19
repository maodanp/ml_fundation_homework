## 线性回归

### 线性回归

#### 最大似然估计解释MLE（最小二乘法）

假设
$$
y^{(i)} = \theta^Tx^{(i)} + \epsilon^{(T)}
$$
其中误差$$\epsilon^{(i)}(1 \le i \le m)$$是独立同分布的，服从均值为0，方差为某定值$$\theta^2$$的**高斯分布**.
$$
p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}) \\
p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}) \\
L(\theta) = \prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}) \\
lnL(\theta) = log\prod_{i=1}^m\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}) \\
=\sum_{i=1}^mlog\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}) \\
=mln\frac{1}{\sqrt{2\pi}\sigma}-\sum_{i=1}^m\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}
$$
则代价函数/代价函数可以表示为:
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$

#### 目标函数求最小值—解析式求解

前面解析出目标函数为:
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2=\frac{1}{2}(X\theta-y)^T(X\theta-y)
$$
梯度:
$$
\nabla{J(\theta)} = \nabla{[\frac{1}{2}(X\theta-y)^T(X\theta-y)]}=\nabla{[\frac{1}{2}(\theta^TX^T-y^T)(X\theta-y)]} \\
=\nabla{\frac{1}{2}(\theta^TX^TX\theta - \theta^TX^Ty - y^TX\theta+y^Ty)} \\
=\frac{1}{2}(2X^TX\theta - X^Ty - (y^TX)^T) = X^TX\theta-X^Ty
$$
令偏导数为0, 得到参数的最优解:
$$
\theta = (X^TX)^{-1}X^Ty
$$
若$$X^TX$$ **不可逆**或者防止**过拟合**，
$$
\theta = (X^TX + \lambda{I})^{-1}X^Ty
$$

#### 目标函数求最小值—梯度下降法

* 初始化$$\theta$$（随机初始化）


* 沿着负梯度方向迭代，更新后的$$\theta$$,使得$$J(\theta)$$更小
  $$
  \theta = \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta}
  $$
  其中，$$\alpha$$ 为学习率，步长

  梯度方向为:
  $$
  \frac{\partial J(\theta)}{\partial \theta_j} = \frac{\partial}{\partial \theta_j} \frac{1}{2}(h_\theta(x) - y)^2 \\
  =2 \cdot \frac{1}{2}(h_\theta(x) - y) \cdot \frac{\partial}{\partial \theta_j}(h_{\theta}(x)-y) \\
  =(h_{\theta}(x)-y) \cdot \frac{\partial}{\partial \theta_j}(\sum_{i=0}^n\theta_ix_i-y)\\
  =(h_{\theta}(x)-y)x_j
  $$


### 回归正则化方法

线性回归的目标函数为:
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2
$$
正则化项(regularizer)或罚项(penalty term)一般是模型复杂度的单调递增函数，模型越复杂，则正则化值就越大。正则化项可以是模型参数向量的范数，一般形式:
$$
min \frac{1}{N}\sum_{i=1}^NL(y_i, f(x_i)) + \lambda{J(f)}
$$
其中第1项是**经验风险**，第2项是**正则化项**。$$\lambda$$为调整两者之间关系的系数。第1项的经验风险较小的模型可能较为复杂(有多个非零参数，或者参数较大)，这时第2项d额模型复杂度会较大。

>正则化的作用是选择经验风险与模型复杂度同时较小的模型

Ridge将目标函数加入平方和损失（L1正则化）:
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda \sum_{j=1}^n\theta_j^2
$$
Lasso将目标函数加入平方和损失（L2正则化）:
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda \sum_{j=1}^n|\theta_j|
$$
Elastic Net:
$$
J(\theta) = \frac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2 + \lambda_1  \sum_{j=1}^n|\theta_j| +\lambda_2 \sum_{j=1}^n\theta_j^2 
$$
回归正则化方法在高纬和数据集变量之间多重共线性情况下运行良好。

正则化符合奥卡姆剃刀原理：在所有可能选择的模型中，能够很好的解释一直数据并且十分简单才是最好的模型，也就是应该选择的模型；从贝叶斯角度看，正则化项对应于模型的先验概率，可以假设负责的模型有较大的先验概率，简单的模型有较小的先验概率。

### 交叉验证

———————————————————————————————————

|———训练数据-> $$\theta$$———|———验证数据->  $$\lambda$$———|———测试数据———|

———————————————————————————————————

训练数据集: 用该集合中的数据计算参数$$\theta$$；

验证数据集：$$\lambda$$ 无法通过计算获得，为了计算 $$\theta$$所引入的新的参数(超参数)；通过计算出的一组$$\theta$$，通过比较cost function, 找到一个最优的$$\theta$$，比较同一model不同参数的**优劣**；

测试数据集: 用于比较model的优劣，比如线性回归model1, 决策树model2, **比较不同model的优劣**。

CV—>$$\lambda$$—>$$\theta$$—>$$\hat y$$

如果给定样本数据充足，进行选择的一种简单方法是随机将数据集切分成三部分，在学习到的不同复杂度的参数对应模型中，选择对验证集有最小预测误差的模型，由于验证集有足够多的数据，用它对模型进行选择也是游戏哦啊的。

但是实际应用中数据是不充足的，为了选择好的模型可以采用交叉验证的方法。交叉验证的基本思想是重复的使用数据；把给定的数据进行切分，将切分的数据集组合为训练集与测试集，在此基础上反复进行训练、测试以及模型选择。该过程也是为了选择出泛化能力强的模型。

* 简单交叉验证
* S折交叉验证

首先随机将数据切分成S个互不相交的大小相同的子集；然后利用S-1个子集数据训练模型，利用余下的自己测试模型；将这一过程对可能的S中选择重复进行；最后选出S次评测中平均测试误差最小的模型。

* 留一交叉验证

S折交叉验证的特殊形式S=N。

### 梯度下降算法

#### 批量梯度下降算法（batch gradient descent）

$$
Repeat \quad until\quad  convergence \{ \\
\theta_j := \theta_j + \alpha\sum_{i=1}^m(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}\\
\}
$$

我们每一轮的参数迭代更新都用到了所有的训练数据，如果训练数据非常多的话，是**非常耗时的**。

#### 随机梯度下降算法（stochastic gradient descent）

$$
Repeat\{ \\
 for \quad i=1 \quad to \quad m, \{ \\
   \theta_j := \theta_j + \alpha (y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)} \\
 \} \\
\}
$$

随机梯度下降是通过每个样本迭代更新一次，对比上面的BGD，迭代一次需要用到所有训练样本。SGD伴随的一个问题是噪音教BGD要多，使得SGD并不是每次迭代都向着整体最优化方向。 但大体上是往最优值方向移动的。

#### 折中梯度下降算法（min-batch gradient descent）

如果不是每拿到一个样本即更改梯度，而是若干个样本的平均梯度作为更新方向，则是mini-batch梯度下降算法。

### 统计意义下的参数

对于m个样本$$(x_1, y_1), (x_2, y_2),\cdots,(x_m, y_m)$$，某模型的估计值为$$(x_1, \hat y_1), (x_2,\hat  y_2),\cdots,(x_m, \hat y_m)$$。

* 样本总平方和TSS (Total Sum of Squares): $$TSS = \sum_{i=1}^m(y_i - \overline y)$$
* 残差平方和RSS (Residual Sum of Squares): $$RSS = \sum_{i=1}^m(\hat y_i - y_i)$$
* 定义$$R^2 = 1-RSS/TSS$$, $$R^2$$越大，拟合效果越好。