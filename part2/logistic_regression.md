## 逻辑回归

### Logistic/sigmoid 函数

$$
h_{\theta}(x) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}} \\
g'(x) = (\frac{1}{1+e^{-x}})'=\cdots=g(x)\cdot(1-g(x))
$$

### Logistic回归参数估计

假定:
$$
P(y=1|x;\theta) = h_{\theta}(x) \\
P(y=0|x;\theta) = 1- h_{\theta}(x) \\
P(y|x;\theta) = (h_{\theta}(x))^y (1-h_{\theta}(x))^{1-y}
$$
对数似然函数:
$$
\begin{align}
L(\theta) 
&= \prod_{i=1}^mp(y^{(i)}|x^{(i)};\theta) \\
&= \prod_{i=1}^m(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}} 
\end{align}
$$

$$
l(\theta) = lnL(\theta) = \sum_{i=1}^my^{(i)}logh(x^{(i)})+(1-y^{(i)})log(1-h(x^{(i)}))
$$

$$
\begin{align}
\frac{\partial l(\theta)}{\partial \theta_j} 
&= \sum_{i=1}^m[\frac{y^{(i)}}{h(x^{(i)})} - \frac{1-y^{(i)}}{1-h(x^{(i)})}]\cdot\frac{\partial h(x^{(i)})}{\partial \theta_j} \\
&= \sum_{i=1}^m[\frac{y^{(i)}}{h(x^{(i)})} - \frac{1-y^{(i)}}{1-h(x^{(i)})}]\cdot\frac{\partial g(\theta^Tx^{(i)})}{\partial \theta_j} \\
&= \sum_{i=1}^m[\frac{y^{(i)}}{h(x^{(i)})} - \frac{1-y^{(i)}}{1-h(x^{(i)})}]\cdot h(x^{(i)}) \cdot (1-h(x^{(i)}))\frac{\partial \theta^Tx^{(i)}}{\partial \theta_j} \\
&=\sum_{i=1}^m(y^{(i)}(1-g(\theta^Tx^{(i)}))-(1-y^{(i)})g(\theta^Tx^{(i)}))\cdot x_j^{(i)} \\
&=\sum_{i=1}^m(y^{(i)} - g(\theta^Tx^{(i)}))\cdot x_j^{(i)}
\end{align}
$$

### 梯度下降法表示

与线性回归的参数估计一样，可以将批量梯度下降法表示如下：
$$
\begin{align}
Repeat \quad until\quad  convergence \{ \\
& \theta_j := \theta_j + \alpha\sum_{i=1}^m(y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)}\\
&\}
\end{align}
$$
随机梯度下降法表示如下：
$$
\begin{align}
Repeat\{ \\
 &for \quad i=1 \quad to \quad m, \{ \\
&   \theta_j := \theta_j + \alpha (y^{(i)}-h_\theta(x^{(i)}))x_j^{(i)} \\
&\} \\
\}
\end{align}
$$
其中:
$$
h_\theta(x)=
\begin{cases}
\theta^Tx, &\text{linear regression} \\[2ex]
\frac{1}{1+e^{-\theta^Tx}}, &\text{logistic regression}
\end{cases}
$$
