# Qwen-NeuronSteering

面向 **Qwen3.5-4B** 的神经元转向工具。通过在 MLP 的 `down_proj` 输入处调整指定神经元的激活，研究中间表示对模型生成行为的影响。

工具采用两阶段流程：先用当前提示词完成一次无干预校准，再根据校准结果计算剂量，在生成过程中施加固定干预。干预通过 PyTorch hook 实现，退出上下文后自动移除，无需修改模型权重。

## 快速开始

需要 Python 3.10 或更高版本和 uv。依赖通过 `uv sync` 安装；模型默认通过国内镜像 `https://hf-mirror.com` 下载 Qwen 官方仓库 `Qwen/Qwen3.5-4B` 的完整快照，包含权重、配置、tokenizer 和聊天模板。镜像下载使用普通 HTTP 传输，禁用 Xet 加速。

```powershell
git clone https://github.com/Seagull-W/Qwen-NeuronSteering.git
cd Qwen-NeuronSteering
uv sync
uv run python -m examples.download_model
uv run python -m examples.local_generation --model "models/Qwen3.5-4B" --prompt "请解释什么是彩虹" --device cpu
```

下载需要网络与足够磁盘空间。重复执行下载命令会复用已有文件；中断后可重新运行。通过 `--output` 指定其他目录，生成时的 `--model` 应指向同一目录；通过 `--revision` 指定提交哈希可固定模型版本。`models/` 已加入 Git 忽略规则。下载完成后，生成阶段仅从本地读取文件。

可以通过 `--endpoint` 更换下载地址，例如切回官方源：

```powershell
uv run python -m examples.download_model --endpoint https://huggingface.co
```

如果默认缓存目录不可写，可在运行命令前设置：

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:HF_HOME = Join-Path $PWD 'models/.hf-cache'
```

使用 GPU 时，将 `--device cpu` 改为 `--device cuda`。示例以 float32 加载模型，请确保设备有足够内存。模型文件均从本地读取。

程序会先输出每个目标神经元的实际干预剂量，再输出生成文本。使用 `--max-new-tokens` 调整生成长度，使用 `--config` 指定实验配置文件。

## 干预配置

在 `steering_config.json` 中设置目标层、神经元编号和有符号转向系数：

```json
{
  "skip_first_token": true,
  "targets": [
    {"layer": 14, "neuron": 8949, "strength": 1.0},
    {"layer": 13, "neuron": 7956, "strength": -1.0}
  ]
}
```

层号和神经元号均采用 **从 0 开始的索引**。例如，`layer: 14` 对应第 15 个 Transformer block。Qwen3.5-4B 的文本部分包含 32 层，每层 MLP 中间维度为 9216。

`strength` 控制干预方向和相对强度：正值沿对应输出方向增加，负值沿相反方向调整，0 表示零增量。实际激活增量由校准结果决定。

`skip_first_token` 控制校准时是否排除第一个有效 token，默认开启。校准排除 padding，并包含聊天模板中的有效 token。配置中的参数可用于实验起点，具体行为效果需通过对照实验评估。

## 工作原理

对送入 dense SwiGLU MLP 的隐藏状态 $x$，中间激活与投影输出为：

$$
a=\operatorname{SiLU}(W_{gate}x)\odot(W_{up}x),\qquad y=W_{down}a.
$$

在第 $n$ 个神经元上加入激活增量 $\Delta_n$，对应的局部输出变化为：

$$
a'_n=a_n+\Delta_n,\qquad y'-y=\Delta_n W_{down}[:,n].
$$

校准阶段收集目标 block 输出残差 $r_L^{(t)}$，计算有效 token 范数的中位数，并按对应权重列的范数归一化：

$$
m_L=\operatorname{median}_{t\in T}\|r_L^{(t)}\|_2,\qquad
\Delta_n=s_n\frac{m_L}{\|W_{down}[:,n]\|_2}.
$$

其中，$T$ 为校准 token 集合，$s_n$ 为配置中的 `strength`。偶数个样本使用 `torch.median` 的较小中间值。

每个提示词单独校准。生成时重新建立 KV cache，将固定剂量用于提示词预填充和后续逐 token 解码。上述输出变化公式描述单层固定输入下的关系，跨层干预的最终效果由模型后续计算共同决定。

## 代码结构

```text
neuron_steering/
  __init__.py              # 公共接口
  core.py                  # 层发现、参数检查、校准与干预
examples/
  download_model.py        # 下载完整模型快照，可重复执行
  local_generation.py      # 本地模型生成入口
tests/
  test_core.py             # 数学关系、校准范围与 hook 清理测试
steering_config.json       # 干预参数
architecture_config.json   # Qwen3.5-4B 架构配置参考
architecture.png           # 架构参考图
pyproject.toml             # 项目依赖与打包配置
```

核心接口包括 `Target`、`discover_layers`、`calibrate` 和 `steer`。`steer` 使用上下文管理器注册和清理 hook，生成过程发生异常时也会清理。

## 测试与实验

无需加载模型权重即可运行小型模型单元测试：

```powershell
uv run python -m unittest discover -s tests
```

评估转向效果时，建议对相同提示词分别运行无干预基线与干预版本，保持聊天模板和生成参数一致，并记录模型版本、依赖版本、转向系数及实际剂量。目前尚未完成真实模型运行验证。

## 支持范围

当前实现面向单条文本输入、处于 `eval()` 模式的模型，以及普通浮点 `torch.nn.Linear` 投影层。校准排除 padding；干预作用于投影当前处理的全部位置。

多模态输入、批量生成、量化、MoE、张量并行、融合 MLP 和编译图需要额外适配。同一模型实例应顺序使用干预上下文，校准时保持无干预状态。
