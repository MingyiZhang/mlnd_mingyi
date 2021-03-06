# 机器学习纳米学位
## 毕业项目
张鸣一</br>
2017年08月09日

## 用深度神经网络侦测走神司机</br>(Using Deep Neural Network to Detect Distracted Driver)

## I. 问题的定义

### 项目概述

驾驶员未能完全将精力集中于道路上。致使他们的判断出现偏差，谓之“分心驾驶(distracted driving)”。导致分心驾驶的原因很多，其中包括：驾驶员用手机通话，传短信， 阅读， 使用全球定位系统，观看视频或电影，吃东西，吸烟，补妆，跟乘客聊天，身心疲劳等.

驾驶期间，驾驶员必须在任何时候都要全神贯注。否则会带来严重后果。研究表明，驾驶过程中使用移动电话的驾驶员发生交通事故的风险大约是不使用者的四倍。在所有交通事故中，近三成涉及分心驾驶.

世界卫生组织WHO的[资料](http://www.who.int/violence_injury_prevention/publications/road_traffic/distracted_driving/en/)显示，随着移动电话用户的增长和新型车载通讯系统的迅速采用，分心驾驶对安全造成的严重威胁正逐年增加。无人驾驶虽然取得了长足进步，但完全替代人类尚需时日。如何有效地侦测司机走神并加以提醒就变得尤为重要。通过车载监视器监测驾驶员状态，通过深度卷积神经网络的分析，可以一定程度上达到这个目的.


### 问题陈述
项目取自Kaggle的[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)比赛。目标是根据车载监视器显示的画面自动判断驾驶员是否走神，若驾驶员走神，则给予提醒。其中根据不同的走神原因又将走神状态进行了细分（具体分类信息将在“分析”一节进行详细描述）。通过对监控镜头中二维图像的分析，我们期望可以判断司机走神与否，如若走神，判断是什么原因导致的。

为了达成项目目标，我将建立一个深度卷积神经网络模型。因为重新训练一个卷积神经网络将是一个费力不讨好的方法，所以我将采用几个预训练模型，综合起来建立模型。训练所用数据集为Kaggle提供的监控图像与其对应的状态。用训练集数据训练模型，精细调节参数。

我期望训练的模型，对于输入的图像，可以分析图像中的特征，将图像内容分入相应的类中，并显示出分类结果。分类的好坏将以准确率和LogLoss函数来衡量.


### 评价指标
评估指标采用准确率和LogLoss来衡量. 准确率可以粗略评估模型的好坏. 而LogLoss可以更加精确地评估模型对于分类的确定程度. Kaggle的LeaderBoard采用的方式为LogLoss.
LogLoss的表达式为
$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^M y_{ij}\ln p_{ij}
$$

其中是测试集的图片数, 是图像类别数(在此项目中), 是自然对数,当第张图属于第类, 是1, 反之则为0. 是预测第张图属于第类的概率, 取值在0到1之间. 从中我们可以看出, 当模型预测得越准确, 置信概率越高, 那么Logloss就越小. 因此Logloss的值越小表明模型越好.
根据我对Kaggle上面讨论的浏览, 我决定将我的目标定在准确率大于90%, LogLoss小于0.5. 希望可以达成.


## II. 分析

### 数据的探索

数据集来自Kaggle的[State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)竞赛。数据集分为三个文件。

- `imgs.zip` 所有训练集和测试集图片的压缩文件：解压后包含`train`和`test`两个文件夹，其中`train`文件夹包含`c0`到`c9`十个文件夹，每个文件夹对应一种驾驶状态。具体信息统计如下。
    ```
    ├── train [22424 images]
    │   ├── c0 安全驾驶 [2489 images]
    │   ├── c1 右手手机打字 [2267 images]
    │   ├── c2 右手打电话 [2317 images]
    │   ├── c3 左手手机打字 [2346 images]
    │   ├── c4 左手打电话 [2326 images]
    │   ├── c5 调收音机 [2312 images]
    │   ├── c6 喝水 [2325 images]
    │   ├── c7 拿后面的东西 [2002 images]
    │   ├── c8 整理头发和化妆 [1911 images]
    │   └── c9 和乘客交谈 [2129 images]
    │   
    └── test [79726 images]
    ```
- `driver_imgs_list.csv` 所有训练集图像文件名(img)列表，包含对应的驾驶员(subject)和行为分类(classname)。其中有26位驾驶员。
- `sample_submission.csv` 比赛提交样本。

[//]: # (Image References)

[c0]: ./dataset_original/train/c0/img_34.jpg
[c1]: ./dataset_original/train/c1/img_6.jpg
[c2]: ./dataset_original/train/c2/img_94.jpg
[c3]: ./dataset_original/train/c3/img_5.jpg
[c4]: ./dataset_original/train/c4/img_14.jpg
[c5]: ./dataset_original/train/c5/img_56.jpg
[c6]: ./dataset_original/train/c6/img_0.jpg
[c7]: ./dataset_original/train/c7/img_81.jpg
[c8]: ./dataset_original/train/c8/img_26.jpg
[c9]: ./dataset_original/train/c9/img_19.jpg
[gif]: ./gifs/p002_movie.gif

以下是每类驾驶状态的图片样本：

|                   |                   |                     |
|:-----------------:|:-----------------:|:-------------------:|
|   `c0` 安全驾驶   | `c1` 右手手机打字 |   `c2` 右手打电话   |
|  ![alt text][c0]  |  ![alt text][c1]  |   ![alt text][c2]   |
| `c3` 左手手机打字 |  `c4` 左手打电话  |    `c5` 调收音机    |
|  ![alt text][c3]  |  ![alt text][c4]  |   ![alt text][c5]   |
|     `c6` 喝水     | `c7` 拿后面的东西 | `c8` 整理头发和化妆 |
|  ![alt text][c6]  |  ![alt text][c7]  |   ![alt text][c8]   |
|  `c9` 和乘客交谈  |                   |                     |
|  ![alt text][c9]  |                   |                     |

通过观察训练图像，我们可以发现，对于同一个驾驶员，训练图像由监视录像逐帧生成。也就是说，若将同一个驾驶员的图片按照`driver_imgs_list.csv`所给的顺序生成`gif`文件，我们将得到一个连贯的小视频。生成的`gif`文件可在文件夹`gifs/`中找到。

在这一部分，你需要探索你将要使用的数据。数据可以是若干个数据集，或者输入数据/文件，甚至可以是一个设定环境。你需要详尽地描述数据的类型。如果可以的话，你需要展示数据的一些统计量和基本信息（例如输入的特征（features)，输入里与定义相关的特性，或者环境的描述）。你还要说明数据中的任何需要被关注的异常或有趣的性质（例如需要做变换的特征，离群值等等）。你需要考虑：
- _如果你使用了数据集，你要详尽地讨论了你所使用数据集的某些特征，并且为阅读者呈现一个直观的样本_
- _如果你使用了数据集，你要计算并描述了它们的统计量，并对其中与你问题相关的地方进行讨论_
- _如果你**没有**使用数据集，你需要对你所使用的输入空间（input space)或输入数据进行讨论？_
- _数据集或输入中存在的异常，缺陷或其他特性是否得到了处理？(例如分类变数，缺失数据，离群值等）_

### 探索性可视化
在这一部分，你需要对数据的特征或特性进行概括性或提取性的可视化。这个可视化的过程应该要适应你所使用的数据。就你为何使用这个形式的可视化，以及这个可视化过程为什么是有意义的，进行一定的讨论。你需要考虑的问题：
- _你是否对数据中与问题有关的特性进行了可视化？_
- _你对可视化结果进行详尽的分析和讨论了吗？_
- _绘图的坐标轴，标题，基准面是不是清晰定义了？_

### 算法和技术
在这一部分，你需要讨论你解决问题时用到的算法和技术。你需要根据问题的特性和所属领域来论述使用这些方法的合理性。你需要考虑：
- _你所使用的算法，包括用到的变量/参数都清晰地说明了吗？_
- _你是否已经详尽地描述并讨论了使用这些技术的合理性？_
- _你是否清晰地描述了这些算法和技术具体会如何处理这些数据？_

### 基准模型
在这一部分，你需要提供一个可以用于衡量解决方案性能的基准结果/阈值。这个基准模型要能够和你的解决方案的性能进行比较。你也应该讨论你为什么使用这个基准模型。一些需要考虑的问题：
- _你是否提供了作为基准的结果或数值，它们能够衡量模型的性能吗？_
- _该基准是如何得到的（是靠数据还是假设）？_


## III. 方法
_(大概 3-5 页）_

### 数据预处理
在这一部分， 你需要清晰记录你所有必要的数据预处理步骤。在前一个部分所描述的数据的异常或特性在这一部分需要被更正和处理。需要考虑的问题有：
- _如果你选择的算法需要进行特征选取或特征变换，你对此进行记录和描述了吗？_
- _**数据的探索**这一部分中提及的异常和特性是否被更正了，对此进行记录和描述了吗？_
- _如果你认为不需要进行预处理，你解释个中原因了吗？_

### 执行过程
在这一部分， 你需要描述你所建立的模型在给定数据上执行过程。模型的执行过程，以及过程中遇到的困难的描述应该清晰明了地记录和描述。需要考虑的问题：
- _你所用到的算法和技术执行的方式是否清晰记录了？_
- _在运用上面所提及的技术及指标的执行过程中是否遇到了困难，是否需要作出改动来得到想要的结果？_
- _是否有需要记录解释的代码片段(例如复杂的函数）？_

### 完善
在这一部分，你需要描述你对原有的算法和技术完善的过程。例如调整模型的参数以达到更好的结果的过程应该有所记录。你需要记录最初和最终的模型，以及过程中有代表性意义的结果。你需要考虑的问题：
- _初始结果是否清晰记录了？_
- _完善的过程是否清晰记录了，其中使用了什么技术？_
- _完善过程中的结果以及最终结果是否清晰记录了？_


## IV. 结果
_（大概 2-3 页）_

### 模型的评价与验证
在这一部分，你需要对你得出的最终模型的各种技术质量进行详尽的评价。最终模型是怎么得出来的，为什么它会被选为最佳需要清晰地描述。你也需要对模型和结果可靠性作出验证分析，譬如对输入数据或环境的一些操控是否会对结果产生影响（敏感性分析sensitivity analysis）。一些需要考虑的问题：
- _最终的模型是否合理，跟期待的结果是否一致？最后的各种参数是否合理？_
- _模型是否对于这个问题是否足够稳健可靠？训练数据或输入的一些微小的改变是否会极大影响结果？（鲁棒性）_
- _这个模型得出的结果是否可信？_

### 合理性分析
在这个部分，你需要利用一些统计分析，把你的最终模型得到的结果与你的前面设定的基准模型进行对比。你也分析你的最终模型和结果是否确确实实解决了你在这个项目里设定的问题。你需要考虑：
- _最终结果对比你的基准模型表现得更好还是有所逊色？_
- _你是否详尽地分析和讨论了最终结果？_
- _最终结果是不是确确实实解决了问题？_


## V. 项目结论
_（大概 1-2 页）_

### 结果可视化
在这一部分，你需要用可视化的方式展示项目中需要强调的重要技术特性。至于什么形式，你可以自由把握，但需要表达出一个关于这个项目重要的结论和特点，并对此作出讨论。一些需要考虑的：
- _你是否对一个与问题，数据集，输入数据，或结果相关的，重要的技术特性进行了可视化？_
- _可视化结果是否详尽的分析讨论了？_
- _绘图的坐标轴，标题，基准面是不是清晰定义了？_


### 对项目的思考
在这一部分，你需要从头到尾总结一下整个问题的解决方案，讨论其中你认为有趣或困难的地方。从整体来反思一下整个项目，确保自己对整个流程是明确掌握的。需要考虑：
- _你是否详尽总结了项目的整个流程？_
- _项目里有哪些比较有意思的地方？_
- _项目里有哪些比较困难的地方？_
- _最终模型和结果是否符合你对这个问题的期望？它可以在通用的场景下解决这些类型的问题吗？_


### 需要作出的改进
在这一部分，你需要讨论你可以怎么样去完善你执行流程中的某一方面。例如考虑一下你的操作的方法是否可以进一步推广，泛化，有没有需要作出变更的地方。你并不需要确实作出这些改进，不过你应能够讨论这些改进可能对结果的影响，并与现有结果进行比较。一些需要考虑的问题：
- _是否可以有算法和技术层面的进一步的完善？_
- _是否有一些你了解到，但是你还没能够实践的算法和技术？_
- _如果将你最终模型作为新的基准，你认为还能有更好的解决方案吗？_

----------
** 在提交之前， 问一下自己... **

- 你所写的项目报告结构对比于这个模板而言足够清晰了没有？
- 每一个部分（尤其**分析**和**方法**）是否清晰，简洁，明了？有没有存在歧义的术语和用语需要进一步说明的？
- 你的目标读者是不是能够明白你的分析，方法和结果？
- 报告里面是否有语法错误或拼写错误？
- 报告里提到的一些外部资料及来源是不是都正确引述或引用了？
- 代码可读性是否良好？必要的注释是否加上了？
- 代码是否可以顺利运行并重现跟报告相似的结果？
