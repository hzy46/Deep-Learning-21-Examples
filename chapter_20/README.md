### 20. 深度强化学习：Deep Q Learning

本节的程序来源于项目 https://github.com/carpedm20/deep-rl-tensorflow。

**20.2.1 安装依赖库**

```
pip install gym[all] scipy tqdm
```

**20.2.2 训练**

使用GPU训练：
```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=True
```

使用CPU训练：
```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=False
```

打开TensorBoard：
```
tensorboard --logdir logs/
```

**20.2.3 测试**

测试在GPU上训练的模型：

```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=True --is_train=False
```

测试在CPU上训练的模型：
```
python main.py --network_header_type=nips --env_name=Breakout-v0 --use_gpu=True --is_train=True
```

在上述命令中加入--display=True选项，可以实时显示游戏进程。

#### 拓展阅读

- 本章主要介绍了深度强化学习算法DQN，关于该算法的更多细节，可以参考论文Playing Atari with Deep Reinforcement Learning。

- 本章还介绍了OpenAI 的gym 库，它可以为我们提供常用的强化学 习环境。读者可以参考它的文档https://gym.openai.com/docs/ 了解 gym 库的使用细节，此外还可以在https://gym.openai.com/envs/ 看到当前Gym 库支持的所有环境。
