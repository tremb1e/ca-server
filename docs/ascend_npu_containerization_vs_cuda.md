# 本项目容器化接入宿主机 Ascend NPU 与 CUDA 版本的差异分析

## 1. 结论先行

本项目当前的容器化方案，本质上不是“做一个带 NPU 依赖的镜像”这么简单，而是实现了一套**依赖宿主机 Ascend 驱动和 CANN toolkit 的手工接线方案**。

和常见 CUDA 版容器相比，最大的区别有三点：

1. CUDA 容器通常依赖 `nvidia-container-toolkit`/`--gpus` 自动把 GPU 设备和驱动能力注入容器；本项目的 Ascend 容器则显式挂载宿主机 `/usr/local/Ascend/*`、`/etc/ascend_install.info`、`/dev/davinci*` 等资源，并由入口脚本手工补齐环境变量。
2. CUDA 容器通常主要关心 `CUDA_VISIBLE_DEVICES`；本项目为了让 `torch_npu` 正常初始化，还额外维护了大量 `ASCEND_*` 环境变量、动态库路径、Python 路径、日志目录和 `kernel_meta` 目录。
3. CUDA 版打包通常只要保证 `torch`/CUDA 对应关系即可；本项目为了 Ascend 运行，还专门处理了 `torch_npu` 的安装、冻结发布时的动态库收集、运行时代码中的 NPU 探测与多进程设备重映射。

一句话概括：**CUDA 容器更像“把 GPU 交给容器”；本项目的 Ascend 容器更像“把宿主机整套 Ascend 用户态运行时借给容器”。**

## 2. 说明比较基线

仓库里没有单独提供一个“CUDA 专用 Dockerfile/compose 文件”。因此，下面的“CUDA 版本”比较对象，是两部分的组合：

1. 本项目代码里已经保留的 CUDA 兼容路径，例如 `cuda`/`npu`/`cpu` 三种后端自动探测与切换。
2. PyTorch 项目里最常见的 CUDA 容器化方式，即：
   - 镜像内安装与 CUDA 对应的 `torch`
   - 运行时使用 `--gpus all` 或 Docker Compose 的 `gpus`/NVIDIA runtime
   - 主要通过 `CUDA_VISIBLE_DEVICES` 控制显卡可见性

因此，文中的“和 CUDA 版相比”是**基于仓库现状加上常规 CUDA 容器做法的工程对比**，不是说仓库里存在另一套已提交的 CUDA Compose 文件。

## 3. 当前 NPU 容器化方案的核心设计

### 3.1 镜像不内置 Ascend 驱动，运行时吃宿主机环境

无论源码运行镜像还是预编译镜像，都明确写了同一条原则：运行时直接消费宿主机 Ascend 驱动和 toolkit，不在镜像里安装驱动包。

- `Dockerfile:80-81` 明确写明 runtime 通过 bind mount 消费宿主机 Ascend driver/toolkit。
- `Dockerfile:97` 先创建 `/usr/local/Ascend/driver` 与 `/usr/local/Ascend/ascend-toolkit` 目录，目的是给宿主机挂载点预留路径。
- `Dockerfile.prebuilt:7-8`、`Dockerfile.prebuilt:32` 也沿用了同样策略。

这说明本项目的镜像只负责应用本身，真正的 NPU 用户态运行时来自宿主机。

### 3.2 Compose 显式透传宿主机 NPU 设备、工具和元数据

`docker-compose.yml` 不是只挂一个数据目录，而是把 Ascend 运行所需的关键资源都显式透传进容器：

- 宿主机安装信息：`/etc/ascend_install.info`，见 `docker-compose.yml:54`
- 驱动目录：`/usr/local/Ascend/driver`，见 `docker-compose.yml:55`
- toolkit 目录：`/usr/local/Ascend/ascend-toolkit`，见 `docker-compose.yml:56`
- 调试工具：`npu-smi`、`msnpureport`，见 `docker-compose.yml:57-58`
- 设备节点：`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`、`/dev/davinci0-7`，见 `docker-compose.yml:87-100`
- 运行日志与编译缓存目录：`deploy/data/ascend/kernel_meta` 和 `deploy/data/ascend/log`，见 `docker-compose.yml:47-48`

同时，容器还启用了：

- `platform: linux/arm64`，见 `docker-compose.yml:65`
- `privileged: true`，见 `docker-compose.yml:101`

这说明当前方案对宿主机形态有较强依赖，不是完全可移植的“到处跑”的通用镜像。

### 3.3 入口脚本负责把 Ascend 运行时“接活”

宿主机资源挂进来以后，容器还必须把这些资源组装成 `torch_npu` 可以消费的运行环境。这个工作由 `deploy/entrypoint.sh` 完成：

- 优先 `source` 宿主机 CANN 的 `set_env.sh`，见 `deploy/entrypoint.sh:17-27`
- 追加 Ascend Python 组件路径，见 `deploy/entrypoint.sh:43-48`
- 追加 Ascend 编译器/工具链路径，见 `deploy/entrypoint.sh:45`、`deploy/entrypoint.sh:50-56`
- 追加 Ascend 动态库路径，见 `deploy/entrypoint.sh:46-47`、`deploy/entrypoint.sh:58-64`

这一层非常关键。因为宿主机把目录挂进容器，并不等于 `torch_npu` 自动就能找到库、插件、Python 模块和 OPP/TBE 资源，仍然需要显式配置环境。

## 4. 与 CUDA 版本相比，额外做了哪些工作

下面按工程层次拆开说明。

| 对比项 | 常见 CUDA 版容器 | 本项目 Ascend NPU 容器 | 额外工作 |
| --- | --- | --- | --- |
| 驱动接入方式 | 主要依赖 NVIDIA runtime 自动注入 | 手工挂载宿主机 Ascend 驱动、toolkit、安装信息和工具文件 | 需要在 Compose 中显式维护挂载清单 |
| 设备透传 | 常见是 `--gpus` 或 `gpus: all` | 手工枚举 `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`、`/dev/davinci0-7` | 需要按机器实际卡数维护设备节点 |
| 权限模型 | 常见可不使用 `privileged` | 当前使用 `privileged: true` 保守保证兼容性 | 需要为 Ascend 设备访问放宽权限 |
| 环境变量 | 主要是 `CUDA_VISIBLE_DEVICES` | 需要一组 `ASCEND_*` 变量外加 `PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH` | 需要入口脚本动态拼接运行环境 |
| Python 依赖 | `torch` + CUDA 对应 wheel | `torch` + `torch-npu` 严格配套，且受 CANN Python 组件影响 | 需要版本锁定和兼容性约束 |
| 运行日志/缓存 | 一般业务日志即可 | 还要保留 `kernel_meta`、Ascend process log | 需要额外挂载和目录初始化 |
| 冻结发布 | 多数只处理 `torch` 动态库 | 还要额外收集 `torch_npu` 动态库/子模块/数据文件 | 需要补 PyInstaller spec 和构建脚本 |
| 应用代码 | 往往直接 `torch.cuda.*` | 需要统一 `npu/cuda/cpu` 探测、AMP、设备设置、worker 设备映射 | 需要增加专门的加速器抽象层 |

### 4.1 手工挂载宿主机 Ascend 运行时资源

这是最明显的一项额外工作。

在 CUDA 容器里，常见模式是：

- 容器内提前带好 CUDA 兼容环境
- 运行时通过 NVIDIA runtime 注入驱动能力
- 容器侧一般不需要手工挂完整 toolkit 目录

而本项目为了接入宿主机 NPU，必须在 Compose 中显式挂入：

- `ascend_install.info`
- `driver`
- `ascend-toolkit`
- `npu-smi`
- `msnpureport`
- `davinci` 系列设备节点

证据见 `docker-compose.yml:51-59`、`docker-compose.yml:87-100`。

这说明项目不是依赖 Docker vendor runtime 自动发现 NPU，而是由项目自己把宿主机 Ascend 运行环境“搬进容器”。

### 4.2 手工建立 Ascend 环境变量和库搜索路径

相比 CUDA 版主要依赖 `CUDA_VISIBLE_DEVICES` 与镜像内置 CUDA 库，本项目对环境变量的处理明显更重：

- `Dockerfile:131-144` 预置了 `ASCEND_DRIVER_HOME`、`ASCEND_INSTALL_INFO`、`ASCEND_TOOLKIT_HOME`、`ASCEND_OPP_PATH`、`ASCEND_RT_VISIBLE_DEVICES`、`LD_LIBRARY_PATH`、`PYTHONPATH`
- `docker-compose.yml:21-33` 在运行层再次把 `ASCEND_*` 与路径类变量补齐
- `deploy/entrypoint.sh:17-64` 在容器启动时再次根据宿主机实际挂载结果做动态修正

这类工作在 CUDA 容器中一般不会这么重。因为 CUDA 版通常把用户态库放在镜像里，或者由 NVIDIA runtime 自动完成关键库映射，应用层不需要显式关心 OPP、TBE、AICPU、CANN Python site-packages 这些路径。

### 4.3 额外处理 Ascend 日志和编译缓存目录

本项目不仅挂设备和驱动，还专门保留了 Ascend 特有运行痕迹目录：

- `/app/kernel_meta`
- `/app/ascend_logs`

见 `Dockerfile:107-110`、`docker-compose.yml:47-48`、`deploy/entrypoint.sh:29-31`。

这些目录通常用于算子编译产物、运行时日志或调试信息。CUDA 版容器往往只关心业务日志，不需要专门为框架后端准备这类目录。

### 4.4 额外处理 `torch-npu` 安装和版本兼容

源码运行镜像里，构建阶段除了安装 `torch`，还额外支持安装 `torch-npu`：

- `Dockerfile:17-20` 定义了 `INSTALL_TORCH_NPU` 和 `TORCH_NPU_VERSION`
- `Dockerfile:55-63` 只有在 `INSTALL_TORCH_NPU=1` 时才安装 `torch-npu`
- `docker-compose.yml:71-74` 默认把 `INSTALL_TORCH_NPU=1` 和 `TORCH_NPU_VERSION=2.6.0.post5` 传入构建

此外，`requirements.txt` 还专门说明了 Ascend CANN Python runtime 与 NumPy 2.x 的兼容性问题，因此把 NumPy 锁在 1.26.x 区间：

- `requirements.txt:23-25`

这属于典型的 Ascend 额外工作。CUDA 版一般只需要保证 `torch` 和 CUDA wheel 匹配，很少需要因为 CANN Python 运行时再额外压住 NumPy 主版本。

### 4.5 冻结发布时额外收集 `torch_npu` 动态库和元数据

本项目显然已经遇到过一个现实问题：即使业务代码打成单二进制/单目录发布，Ascend 相关 Python 扩展和动态库仍然不能丢，否则容器里无法初始化 NPU。

为此项目补了两层工作：

1. PyInstaller spec 专门收集 `torch_npu`
2. 构建脚本在检测到 `torch_npu` 可用时，把它的子模块、数据文件和动态库都打进去

具体证据：

- `ca-server.spec:10` 收集 `torch_npu` 数据文件
- `ca-server.spec:14` 收集 `torch_npu` 动态库
- `ca-server.spec:17` 收集 `torch_npu` 子模块
- `scripts/build_frozen_bundle.sh:84-96` 运行时检测 `torch_npu`，存在则追加 `--collect-submodules torch_npu --collect-data torch_npu --collect-binaries torch_npu`
- `Dockerfile.prebuilt:75-77` 的 `LD_LIBRARY_PATH` 里，还专门把 `/app/bin/_internal/torch_npu` 和 `/app/bin/_internal/torch_npu/lib` 放进去了

这也是和 CUDA 版相比的重要区别。CUDA 容器即使做冻结发布，重点通常是 `torch` 本身；而 Ascend 路径下，`torch_npu` 是额外插件，必须显式照顾。

### 4.6 针对冻结运行时，额外修补 `torch_npu` 初始化行为

项目不只是在打包层处理了 `torch_npu`，还在运行时代码里专门做了补丁，避免冻结运行时下 `torch`/`torch_npu` 因访问源码失败而初始化异常。

证据：

- `src/utils/accelerator.py:16-41`
- `ca_train/accelerator.py:16-41`

这里的 `_patch_torch_config_module_for_frozen_runtime()` 会在 frozen 模式下包装 `torch.utils._config_module` 的源码扫描逻辑，避免 `OSError` 影响启动。

这说明项目为了让 Ascend/NPU 路径在二进制发布模式下可用，已经做了专门兼容；这不是一个典型 CUDA-only 项目常见的工作量。

### 4.7 应用层新增统一加速器抽象，而不是直接写死 `torch.cuda`

项目的代码层也为了容器内 NPU 可用做了额外适配。

核心文件有两份：

- `src/utils/accelerator.py`
- `ca_train/accelerator.py`

它们完成了这些事情：

- 尝试显式导入 `torch_npu`，注册 `torch.npu` 后端，见 `src/utils/accelerator.py:40-45`、`ca_train/accelerator.py:44-52`
- 统一探测 NPU/CUDA/CPU 并做降级，见 `src/utils/accelerator.py:82-123`、`ca_train/accelerator.py:82-128`
- NPU 下显式 `torch.npu.set_device()`，见 `ca_train/accelerator.py:138-145`
- NPU 下使用专门的 AMP/GradScaler，见 `ca_train/accelerator.py:188-214`

如果是纯 CUDA 容器，很多地方完全可以直接写成 `torch.cuda.is_available()`、`torch.cuda.set_device()`、`torch.cuda.amp.autocast()`。现在之所以没有这么写，正是因为项目要在同一容器化产物里兼容 Ascend NPU。

### 4.8 多进程训练时，额外处理 Ascend 设备重映射

这部分是和 CUDA 版最容易混淆、但也最关键的一点。

训练 worker 不是直接拿宿主机物理卡号当容器内逻辑卡号使用，而是先设置可见设备，再在 worker 内统一使用 `npu:0`。

证据：

- `ca_train/accelerator.py:161-173`
- `ca_train/hmog_vqgan_experiment.py:504-513`

NPU 路径下，`set_visible_device()` 会：

- 设置 `ASCEND_RT_VISIBLE_DEVICES=<实际物理卡号>`
- 设置 `ASCEND_DEVICE_ID=0`
- 清掉 `CUDA_VISIBLE_DEVICES`

然后 worker 再把设备字符串设置成 `npu:0`。

这和 CUDA 版的思路相似，但 NPU 这里需要同时照顾 Ascend 自己的设备环境变量语义，额外复杂了一层。

### 4.9 禁止 PyTorch 在环境未就绪前自动乱导入后端

`src/__init__.py:5-7` 设置了：

```python
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
```

注释里已经写明原因：防止 PyTorch 自动导入不完整的可选后端，例如宿主机 Ascend 环境尚未完全挂进容器或环境变量尚未补齐时，`torch_npu` 被提前拉起导致异常。

这也是 Ascend/NPU 路径比较典型的防御性处理。CUDA-only 场景下，通常不需要在应用入口显式关掉 PyTorch 的可选后端自动加载。

## 5. 哪些工作不是为了 NPU，不能混为一谈

分析容器化时，需要把 NPU 专项工作和通用容器工程工作分开。

以下内容虽然也在容器文件里，但不属于“为了能使用宿主机 NPU而额外做的工作”：

- `network_mode: host`，这是为了线上网络路径兼容，见 `README.md:58-59`
- `grpc_health_probe` 健康检查，这是为了服务探活，见 `Dockerfile:65-73`、`docker-compose.yml:109-118`
- 业务数据目录、模型目录、证书目录挂载，这是普通持久化需求，见 `docker-compose.yml:36-50`

真正和 NPU 强相关的是：

- 宿主机 Ascend 资源挂载
- Ascend 环境变量和库路径组装
- `torch_npu` 安装与打包
- NPU 设备节点和权限
- 应用层 NPU 兼容逻辑

## 6. 这套设计为什么比 CUDA 版“更重”

根本原因有三个：

1. **Ascend 运行时更依赖宿主机完整用户态布局。**  
   本项目显式依赖 `/usr/local/Ascend/driver`、`/usr/local/Ascend/ascend-toolkit`、`/etc/ascend_install.info`、设备节点和工具文件，说明单靠镜像内部 Python 依赖并不足以驱动 `torch_npu`。

2. **`torch_npu` 不是“只装一个 wheel 就完事”。**  
   它还依赖 CANN 的 Python 组件、动态库、OPP/TBE 目录、运行日志目录等，因此容器启动阶段必须额外补环境。

3. **项目需要同时兼容 `npu/cuda/cpu`，不能把代码写死成单一后端。**  
   因此除了容器层，还得在应用层增加设备探测、降级策略、AMP 兼容和 worker 设备重映射。

## 7. 最终判断

如果把本项目视为一个“CUDA 版改造成 Ascend 版”的容器化工程，那么额外增加的工作可以概括为四层：

1. **基础设施层**：挂载宿主机 Ascend driver/toolkit/install info/tools/devices，并放宽容器权限。
2. **运行时层**：入口脚本 `source set_env.sh`，补齐 `ASCEND_*`、`PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH`，准备 `kernel_meta` 和 Ascend 日志目录。
3. **依赖与打包层**：安装 `torch-npu`，锁定版本组合，冻结发布时额外收集 `torch_npu` 的库、数据和子模块。
4. **应用代码层**：新增统一 accelerator 抽象、NPU 后端初始化、AMP/GradScaler 兼容、多进程设备重映射和 frozen 运行时补丁。

因此，本项目为了“让容器真正用上宿主机 NPU”所做的工作，明显比一个普通 CUDA 容器更多，而且这些工作大部分都已经直接体现在 `Dockerfile`、`Dockerfile.prebuilt`、`docker-compose.yml`、`deploy/entrypoint.sh`、`requirements.txt`、`ca-server.spec`、`scripts/build_frozen_bundle.sh` 和 `ca_train/src` 的运行时代码里。
