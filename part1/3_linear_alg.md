# 线性代数

## 矩阵

### SVD

奇异值分解(Singular Value Decomposition)是一种重要的矩阵分解方法，可以看做对称方阵在任意矩阵上的推广。

假设A是一个$$m \times n$$阶实矩阵，则存在一个分解使得:
$$
A_{m\times n} = U_{m \times m}\Sigma_{m \times n}V^T_{n \times n}
$$
证明关键步骤:
$$
(A^TA)v_i = \lambda_iv_i \Rightarrow \\
\begin{cases}  
\sigma_i = \sqrt{\lambda_i}\\
\mu_i = \frac{1}{\sigma_i}A\cdot v_i
\end{cases}
\Rightarrow \\
A=U\Sigma V^T
$$
其中，

$$\Sigma$$上的元素称为矩阵A的奇异值；

$$U$$的第$$i$$列称为A的关于$$\sigma_i$$的左奇异向量；

$$V$$的第$$i$$列称为A的关于$$\sigma_i$$的右奇异向量。

### 矩阵的乘法

#### 状态转移矩阵

A为$$m×s$$阶的矩阵，B为$$s×n$$阶的矩阵，那么，$$C=A×B$$是$$m×n$$阶的矩阵，其中:
$$
c_{ij}=\sum_{k=1}^sa_{ik}b_{kj}
$$
转移矩阵实例: 		
假定按照经济状况将人群分成上、中、下三个阶层，用1、2、3表示。假定当前处于某阶层只和上一代有关，即:考察父代为第$$i$$阶层，则子代为第$$j$$阶层的概率。假定为如下**转移概率矩阵**: 
$$
\begin{pmatrix}
0.65 & 0.28 & 0.07 \\
0.15 & 0.67 & 0.18 \\
0.12 & 0.36 & 0.52 
\end{pmatrix}
$$
第$$n+1$$代中处于第$$j$$个阶层的概率为:
$$
\pi(X_{n+1}=j)=\sum_{i=1}^K\pi(X_n=i)\cdot P(X_{n+1}=j|X_n=i) \\
\Rightarrow \pi^{(n+1)}=\pi^{(n)}\cdot P \\
\Rightarrow (\pi^{(n+1)}_1, \pi^{(n+1)}_2, \pi^{(n+1)}_3)_{1\times 3} = (\pi^{(n)}_1, \pi^{(n)}_2, \pi^{(n)}_3)_{1\times 3}  \cdot P_{3\times 3}
$$
其中，$$P$$即为条件概率转移矩阵。

最终得到的概率为:

| 第一阶层  | 第二阶层  | 第三阶层  |
| ----- | ----- | ----- |
| 0.286 | 0.489 | 0.225 |

因为$$P$$的特征值分别为: $$0.32, 0.51, 1$$。 则经过$$n$$次迭代，$$P^n$$最终保留的是$$1$$的特征。

#### 矩阵和向量的乘法

$$A$$为$$m\times n$$的矩阵，$$X$$为$$n \times 1$$的列向量，则AX为$$m \times 1$$的列向量，记: $$\overrightarrow y = A \cdot \overrightarrow x$$

上式实际给出了从$$n$$维空间的点到$$m$$维空间点的线性变化(旋转、平移)。

若$$m=n$$，则$$AX$$完成了$$n$$维空间内的线性变换。

#### 秩与线性方程组解的关系

对于$$n$$元方程组$$A \overrightarrow x = \overrightarrow b$$,

* 无解的充要条件:  $$R(A) \lt R(A, b)$$
* 有唯一解的充要条件： $$R(A) = R(A, b) = n$$
* 有无限多解的充要条件： $$R(A) =R(A, b) \lt n$$

#### 系数矩阵

将向量组A和B所构成的矩阵依次记做$$A=(a_1,a_2,\ldots,a_m)$$和$$B=(b_1,b_2,\ldots,b_n)$$，B组能由A组线性表示，即对每个向量$$b_j$$，存在$$k_{1j},k_{2j},\ldots,k_{mj}$$，使得:
$$
b_j = k_{1j}a_1+k_{2j}a_2 + \ldots + k_{mj}a_m = (a_1, a_2, \ldots, a_m)
\begin{pmatrix}
k_{1j} \\
k_{2j} \\
\ldots \\
k_{mj} 
\end{pmatrix}
$$
从而得到系数矩阵K:
$$
\begin{pmatrix} b_1 & b_2 & \ldots & b_n\end{pmatrix} = (a_1, a_2, \ldots, a_m)
\begin{pmatrix}
k_{11} & k_{12} &\ldots & k_{1n} \\
k_{21} & k_{22} &\ldots & k_{2n} \\
\ldots & \ldots  &\ldots & \ldots \\
k_{m1} & k_{m2} &\ldots & k_{mn} 
\end{pmatrix}
$$

#### C=AB重认识

若$$C=AB$$,则矩阵C的列向量能由A的列向量线性表示，B即为这一表示的系数矩阵。

## 特征值和特征向量

若$$n$$阶矩阵A满足$$A^TA=I$$，成A为正交矩阵，简称**正交阵**。 若A是正交阵，X为向量，则$$A\cdot X$$称作**正交变换**， 不改变向量长度!!

**正交阵和对称阵，能够通过何种操作获得一定意义下的联系??**

$$A$$是$$n$$阶矩阵，若数$$\lambda$$和$$n$$维非0列向量$$x$$满足$$Ax=λx$$,那么, 数$$\lambda$$称为$$A$$的特征值，$$x$$称为$$A$$的对应于特征值$$\lambda$$的**特征向量**。

* 特征值的性质:

$$
\lambda_1+\lambda_2+\ldots + \lambda_n = a_{11}+a_{22}+\ldots+a_{nn} (trace)\\
\lambda_1\lambda_2\ldots \lambda_n = |A|
$$

* 不同的特征值对应的特征向量一定是**线性无关**的。


* **若方阵A是对称阵，线性无关这个结论会加强吗??**  — 实对称阵的特征值是实数，实对称阵不同特征值的特征向量**正交**。
*  设A为n阶对称阵，则必有正交阵P，使得$$P^{-1}AP=P^TAP=\Lambda$$

### 数据白化

计算观测数据X的$$n \times n$$的对称阵$$x \cdot x^T$$的特征值和特征向量，用特征值形成对角阵D，特征向量形成正交阵U, 则$$x \cdot x^ = U^TDU$$。

令$$\hat x = U^TD^{-0.5}U \cdot x$$

则:
$$
\hat x \hat x^T = (U^TD^{-0.5}U \cdot x)( U^TD^{-0.5}U \cdot x)^T = I
$$
**将不相关的数据喂给模型，达到较好效果。**

### 正定阵

对于$$n$$阶方阵$$A$$，若任意$$n$$阶向量$$x$$，都有$$xTAx \gt 0$$，则称$$A$$是正定阵。

正定阵的判定：

* 对称阵A为正定阵
* A的特征值都为正
* A的顺序主子式大于0

### 标准正交基

任何一个矩阵都可以求标准正交基。

查阅: **Schmidt正交化/Givens变换/HouseHolder变换**

### QR分解

* 定义

对于$$m \times n$$的列满秩矩阵A，必有:
$$
A_{m \times n} = Q_{m \times n} \cdot R_{n \times n}
$$
其中，$$Q^T \cdot Q = I$$(即列正交矩阵)，$$R$$为非奇异上三角矩阵。当要求$$R$$的对角线元素为正时，该分解唯一。

该分解为**QR**分解，可用于求解矩阵$$A$$的特征值，$$A$$的逆等问题。

* 计算特征值

$$
A=Q \cdot R \Rightarrow A_1 = Q^TAQ = R \cdot Q \\
\ldots \\
A_k = Q_k \cdot R_k \Rightarrow A_{k+1} = R_k \cdot Q_k \\
\ldots \\
A_k \to diag\{ \lambda_1, \lambda_2, \ldots, \lambda_n\}
$$



正交矩阵乘以原始矩阵，得到的新矩阵，不破坏原始矩阵特征值。

