# HMR (Human Mesh Recovery) - TensorFlow 2 & Python 3.8 Version

> **注意**: 本项目是基于 [akanazawa/hmr](https://github.com/akanazawa/hmr) 的改造版本,已升级支持 **TensorFlow 2.10** 和 **Python 3.8**,并针对 **Windows 平台**进行了优化。

## 📋 项目概述

从单张图像中恢复人体的 3D 形状和姿态。本改造版本在保留原项目核心功能的基础上,进行了以下重大改进:

### ✨ 主要改进

- ✅ **TensorFlow 2.10 支持**: 从 TF 1.3 升级到 TF 2.10,使用 `tf_slim` 替代已废弃的 `tf.contrib`
- ✅ **Python 3.8 兼容**: 完全支持 Python 3.8,不再依赖 Python 2.7
- ✅ **Windows 平台优化**: 解决了 `opendr` 在 Windows 上的编译问题,使用 `pyrender` 作为替代渲染引擎
- ✅ **Mesh 导出功能**: 支持将预测结果保存为标准的 OBJ 格式,可直接在 Blender、MeshLab 等软件中打开
- ✅ **坐标系校正**: 自动应用坐标变换,使导出的模型正立且正面向前
- ✅ **非交互式可视化**: 自动生成可视化结果图片,无需手动关闭窗口

---

## 🚀 快速开始

### 环境要求

- **Python**: 3.8 (推荐)
- **TensorFlow**: 2.10.0 (CPU/GPU)
- **操作系统**: Windows 10/11 (也支持 Linux/macOS)

### 安装步骤

#### 1. 创建 Conda 环境

```bash
conda create -n hmr python=3.8
conda activate hmr
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**国内用户推荐使用阿里云镜像加速:**

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

#### 3. 下载预训练模型

**国内用户推荐使用百度网盘下载(速度更快):**

- **百度网盘链接**: https://pan.baidu.com/s/1MBCxKO2XEw4NG0M_PVnflg
- **提取码**: `chuc`
- **文件名**: `models.tar.gz`

下载后解压到项目根目录:
```bash
tar -xf models.tar.gz
# Windows 用户可以使用 7-Zip 或 WinRAR 解压
```

**或者使用官方链接:**

```bash
# 方法1: 使用 wget (需要安装 wget)
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz
tar -xf models.tar.gz

# 方法2: 手动下载
# 访问 https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz
# 下载后解压到项目根目录
```

解压后应该看到 `models/` 目录包含以下文件:
- `model.ckpt-667589.data-00000-of-00001`
- `model.ckpt-667589.index`
- `model.ckpt-667589.meta`
- `neutral_smpl_with_cocoplus_reg.pkl`


---

## 💻 使用示例

### 基本用法

处理单张图片:

```bash
python -m demo --img_path data/im1954.jpg
```

运行后会生成:
- `output_result.png` - 可视化结果(包含输入图像、骨架投影、3D mesh 等)
- `output_mesh.obj` - 3D mesh 网格文件(OBJ 格式)

### 使用 OpenPose 辅助裁剪

对于未紧密裁剪的图片,可以使用 OpenPose 的输出 JSON 来自动计算边界框:

```bash
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
```

**注意**: 需要先运行 [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 并使用 `--write_json` 选项生成关键点数据。

### 图片要求

- 人物应该相对居中
- 人物高度约为 150px 时效果最佳
- 背景尽量简洁

---

## 📦 输出说明

### OBJ 文件格式

导出的 `output_mesh.obj` 文件包含:
- **顶点数**: 6890 个
- **面片数**: 13776 个三角面
- **坐标系**: Y轴向上,Z轴向前(标准 3D 软件坐标系)
- **兼容性**: 可直接在 Blender、Maya、3ds Max、MeshLab 等软件中打开

### 坐标系变换

为了适配主流 3D 软件,程序会自动应用坐标变换:
- X 轴: 保持不变(左右方向)
- Y 轴: 翻转(使模型正立)
- Z 轴: 翻转(使模型正面向前)

如需保存原始坐标系,可修改 `demo.py` 中的调用:
```python
save_mesh_to_obj(vert_shifted, faces, 'output_mesh.obj', apply_transform=False)
```

---

## 🔧 技术细节

### 主要代码变更

1. **TensorFlow 2 迁移**
   - 使用 `tf.compat.v1` API 保持兼容性
   - 使用 `tf_slim` 替代 `tf.contrib.slim`
   - 更新 `name_scope` 参数用法
   - 移除 `.value` 属性访问(shape 直接可用)

2. **渲染引擎替换**
   - 移除: `opendr`(Windows 编译困难)
   - 新增: `pyrender` + `trimesh` + `PyOpenGL`

3. **依赖管理**
   - 所有依赖版本明确指定,确保可重现性
   - 添加详细的注释说明各包用途

### 已知限制

- ⚠️ 仅支持 CPU 推理(GPU 需要额外配置 CUDA)
- ⚠️ 批量处理多张图片需要自行编写脚本
- ⚠️ 训练代码尚未完全测试(TF2 兼容性)

---

## 📚 训练与数据集

有关训练代码和数据集准备的详细说明,请参考原文档:
- [doc/train.md](doc/train.md)

**注意**: 训练代码可能需要额外的 TF2 适配工作。

---

## 🙏 致谢与引用

### 原始论文

如果您在研究中使用此代码,请引用原始论文:

```bibtex
@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa and Michael J. Black and David W. Jacobs and Jitendra Malik},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```

### 相关项目

以下是一些优秀的衍生项目:

- **[russoale/hmr2.0](https://github.com/russoale/hmr2.0)**: Python 3 + TF 2.0 版本
- **[Dawars/hmr](https://hub.docker.com/r/dawars/hmr/)**: Docker 镜像
- **[MandyMo/pytorch_HMR](https://github.com/MandyMo/pytorch_HMR.git)**: PyTorch 实现
- **[Dene33/video_to_bvh](https://github.com/Dene33/video_to_bvh)**: 视频转 BVH 动画
- **[layumi/hmr](https://github.com/layumi/hmr)**: 添加 2D-3D 颜色映射

---

## 📝 版本历史

### v2.0 (当前版本)
- ✨ 升级到 TensorFlow 2.10
- ✨ 支持 Python 3.8
- ✨ Windows 平台完整支持
- ✨ 添加 OBJ mesh 导出功能
- ✨ 自动坐标系校正
- ✨ 非交互式可视化

### v1.0 (原始版本)
- 基于 akanazawa/hmr
- TensorFlow 1.3 + Python 2.7

---

## 📄 许可证

本项目继承原始 HMR 项目的许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

如有问题或建议,请通过 GitHub Issues 联系。

---

**项目主页**: [akanazawa/hmr](https://github.com/akanazawa/hmr) | [项目页面](https://akanazawa.github.io/hmr/)


