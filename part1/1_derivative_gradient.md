## 导数

### 导数概念

* 简单而言，导数就是曲线的斜率，是曲线变化快慢的反应
* 二阶导数是斜率变化快慢的反应，表征曲线凸凹性

### 常用函数的导数

$$
(sinx)'=cosx \qquad (cosx)'=-sinx \\
(a^x)'=a^xlna \qquad (e^x)=e^x \\
(log_ax)'=\frac{1}{x}log_ae \qquad (lnx)'=\frac{1}{x} \\
(u+v)'=u'+v' \qquad (uv)'=u'v+uv'
$$

### 具体应用

* 已知函数$$f(x)=x^x, x \gt 0$$, 求$$f(x)$$的最小值

$$
t=x^x \\
\to lnt=xlnx \\
 \stackrel {两边对x求导} \to \frac{1}{t}t'=lnx+1 \\
 \stackrel {t'=0} \to lnx+1=0 \\
 \to x=e^{-1}\\
 \to t=e^{-\frac{1}{e}}
$$



* 当$$N \to \infty$$, $$lnN!$$ 值？

$$
lnN!=\sum_{i=1}^Nlni \approx \int_{1}^Nlnxdx \\
=xlnx|_1^N-\int_{1}^Nxdlnx \\
=NlnN-\int_{1}^Nx \cdot \frac{1}{x}dx \\
=NlnN - N + 1 \to NlnN - N
$$

## Taylor 公式-Maclaurin公式

Taylor公式可以把一个可导的函数拆成若干个多项式之和，当n越大，若干个多项式之和逼近于原函数的值
$$
f(x)=f(x_0)+f'(x-x_0)+\frac{f''(x_0)}{2!}(x-x_0)^2+\ldots+\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n+R_n(x) \\
f(x)=f(x_0)+f'(0)x+\frac{f''(x_)}{2!}(x)^2+\ldots+\frac{f^{(n)}(0)}{n!}(x)^n+o(x^n)
$$

### 具体应用

数值计算: 初等函数值的计算（在原点展开）
$$
sin(x)=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+\ldots+(-1)^{m-1}\cdot\frac{x^{2m-1}}{(2m-1)!}+R_{2m}\\
e^x=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\ldots+\frac{x^n}{n!}+R_n
$$


* 计算$$e^x$$

求整数k和小数r，使得 $$x=k \cdot ln2 + r, |r|\lt0.5 \cdot ln2$$
$$
e^x=e^{k\cdot ln2+r} \\
=e^{k\cdot ln2}e^{r} \\
=2^k\cdot e^r
$$

* Gini系数

