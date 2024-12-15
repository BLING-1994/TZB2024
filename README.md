### 2024天智杯 智慧地球领域科目二 
#### 
以影像金字塔为为基础，结合DINOV2检索和GIM匹配
##### 底图库构建
- [00-buildpyarmia](IRSA_Match/00-buildpyarmia.py)   建立金字塔
- [01-bulidbox](IRSA_Match/01-bulidbox.py)     为避免重复区域，建立全范围多尺度网格
- [02-genfeature](IRSA_Match/02-genfeature.py) 特征库构建，每个尺度建立特征库（匹配和检索）
##### 测试数据处理
- [10-roateimg](IRSA_Match/10-roateimg.py) 去除黑白并对长宽比例过大图像切分
- [10-roateimg_new](IRSA_Match/10-roateimg_new.py) 12.15完成，发现一些长宽比例均衡但是在16m的尺度很难匹配，因此针对这些图像切分并保存原始影像4点相对坐标
- [11-processtestimg](IRSA_Match/11-processtestimg.py) 将切分完成的测试数据切分到对应的尺度，包括4m 8m 16m
- [12-genNtestnpy](IRSA_Match/12-genNtestnpy.py)  补漏洞脚本，最开始方案[10-roateimg](10-roateimg.py)只针对切分的生成原始影像4点坐标
- [13-rotefine](IRSA_Match/13-rotefine.py) 生成4米尺度的旋转测试数据，这里解释下，匹配分为粗匹配和精细匹配阶段，此结果也是针对精细匹配阶段[R13finematch_new.py](R13finematch_new.py)
##### 匹配
- [R1-match](IRSA_Match/R1-match.py) 匹配的主文件，按照设定尺度进行检索，然后根据检索结果进行粗匹配和精细匹配
- [R11search](IRSA_Match/R11search.py) 检索，没啥，就dinov2加faiss
- [R12croasematch](IRSA_Match/R12croasematch.py) 粗匹配，将GIM算法的粗匹配部分先计算，其实时间少不了多少，但是分步还是优化空间更大，这个版本是提前准备的，并没考虑到大旋转问题
- [R12croasematch_new](IRSA_Match/R12croasematch_new.py) 最终版本粗匹配，旋转匹配，不是0度左右的不要！，然后将判别的角度传递给精细匹配部分，精细匹配就一定在正向图上面，由于精细匹配在4m进行，因此需要[13-rotefine](13-rotefine.py)、
- [R13croasematch](IRSA_Match/R13croasematch.py) 精细匹配，这个版本是赛前几天准备的，忽视了2个很大的问题，旋转和数据越界，本地模拟数据是用的一张很大的底图，因此随便切就可以，但是赛场数据就重新判断相交区域，挺麻烦的。也想过用vrt做虚拟数据集裁切，但是貌似不能直接读金字塔，效率有问题，因此14号放弃此方案
- [R13finematch_new](IRSA_Match/R13finematch_new.py) 精细匹配, 直接重构，几个坐标算来算去是真头痛，也发现了有些地方点减一，有些地方没减，影像不大（最终翻车？）
##### 最终问题
在15号早上提交之后，我意识到一个掉精度的问题，同样50张图，067的全色数据是8.2分，200的数据是7.2，这里直接导致了5个点的分差。原因在于精细匹配策略的问题，精细匹配是在4m尺度随机选择10个512的相交区域（其实叫精细匹配，也不能算，或许是匹配的二次确认比较靠谱）。问题就来了，4m分辨率512的图是不足以获取200m的全图精确范围的，而067还能凑合，稳定很多。
解决其实不难,找到测试数据的范围，将图像缩放到合适大小二次匹配校正就好
- [F1_down.py](IRSA_Match/F1_down.py) 所有数据下采样
- [F2-buildvrt.py](IRSA_Match/F2-buildvrt.py) 构建虚拟数据集
- [F3-cutdata.py](IRSA_Match/F3-cutdata.py) 切割底图对应区域，保留对应的投影信息
- [F4-finematch.py](IRSA_Match/F4-finematch.py) 二次匹配
然后。。。奇葩的事情出现了，我本地跑的好好的，确实要准确很多，但是但是，拷贝到比赛计算机，[F3-cutdata.py](F3-cutdata.py)出来的图和测试图对不上？？？离谱离谱，然后就打扰了呗。

#### 最后瞎扯淡
这次比赛确实是挺难的，其实初赛之后也就有空的时候看看匹配的算法，检索的算法看看效果。当前段时间群里面赛题慢慢公布，我意识到，这次竞赛的考验不是算法，是工程能力。我大概11号晚上左右开始全力投入。看着代码不是很多，但是需要考虑如何快速解决问题，因此将代码完全分片，然后将所有的配置信息集中（这避免很多的问题和节省时间），还有就是在每个步骤中尽可能把可能有用的信息都流转。
这几天连话都没咋说，真的是怕说几句话就断开了思路，希望后续可以多多交流沟通！
最后，天智杯，我还会回来的！！！

#### 参考:
([GIM:Learning Generalizable Image Matcher From Internet Videos](https://github.com/xuelunshen/gim))
([DINOv2: Learning Robust Visual Features without Supervision](https://github.com/facebookresearch/dinov2))
