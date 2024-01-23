# SimpleAlphaZero_Gomoku

### Training

请进入文件夹`alphazero`下，运行：

```
python train.py
```

若想更改训练参数，请前往`config.py`中进行设置。

### Evaluation

你可以运行下面的命令实现Minimax和AlphaZero之间的对弈：

```
python eval.py --p1 alphazero --p2 minimax --render --games 10
```