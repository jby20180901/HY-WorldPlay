# 无限长视频生成使用指南

## 概述

`generate_infinite_video.py` 是一个用于生成无限长视频的脚本，它使用滑动窗口拼接技术将多个视频片段无缝连接起来。这个脚本基于原有的 `generate.py` 脚本扩展，保持了所有原始功能和参数，同时新增了一些关键功能以支持无限视频生成。

## 新增特性

1. **滑动窗口拼接**：通过生成多个连续的视频片段并平滑拼接来制作长视频
2. **循环运动支持**：运动序列可以循环使用，创建无限循环的动画效果
3. **重叠帧处理**：片段之间重叠一定数量的帧，提高拼接自然度
4. **可配置参数**：支持自定义片段数量、窗口大小和重叠帧数量

## 主要新增参数

### 核心参数

- `--num_segments`: 要生成的视频片段总数（默认：5）
  - 例如：`--num_segments 10` 生成10个片段
- `--video_length`: 每个视频片段的帧数（默认：127）
  - 这是单个片段的长度，与原脚本一致
- `--overlap`: 片段之间的重叠帧数（默认：16）
  - 重叠帧用于平滑连接相邻片段
  - 建议设置为 16-32 以获得最佳效果

## 使用方法

### 基本用法

```bash
python generate_infinite_video.py \
  --pose "w-3, right-0.5, d-4" \
  --prompt "A beautiful landscape with mountains and lakes" \
  --model_path "/path/to/model" \
  --action_ckpt "/path/to/action_checkpoint" \
  --image_path "/path/to/reference_image.jpg" \
  --resolution 480p \
  --model_type bi \
  --num_segments 10 \
  --overlap 16
```

### 详细参数说明

| 参数名 | 用途 | 示例 |
|--------|------|------|
| `--pose` | 相机运动指令字符串 | `"w-3, right-0.5, d-4"` |
| `--prompt` | 文本提示词 | `"A beautiful landscape"` |
| `--model_path` | 模型权重路径 | `"./models/hunyuanvideo-1.5"` |
| `--action_ckpt` | 动作模型检查点 | `"./models/action_ckpt.pth"` |
| `--image_path` | 参考图片路径 | `"./input/reference.jpg"` |
| `--resolution` | 视频分辨率 | `480p` 或 `720p` |
| `--model_type` | 模型类型 | `bi`（双向）或 `ar`（自回归） |
| `--num_segments` | 片段数量 | `5`（生成5个片段） |
| `--video_length` | 单个片段帧数 | `127`（与原脚本一致） |
| `--overlap` | 重叠帧数 | `16`（片段间重叠16帧） |

## 实现原理

### 1. 滑动窗口拼接

脚本将视频生成分解为多个小片段，每个片段使用相同的模型和参数生成。片段之间通过重叠帧平滑连接：
- 第一个片段生成完整的 `video_length` 帧
- 后续片段生成完整帧后，去掉前 `overlap` 帧，再与之前的片段连接

### 2. 循环运动系统

运动指令序列会被循环使用，这样即使原始运动序列有限，也能创造出持续的运动效果：
- 当当前片段用完所有运动指令后，会自动回到运动序列的开头

### 3. 平滑过渡

通过设置适当的重叠帧数，相邻片段之间的过渡会更加自然。重叠帧的数量可以通过 `--overlap` 参数调整。

## 注意事项

1. **内存管理**：生成多个片段可能会占用更多内存
   - 建议使用 `--offloading true` 启用模型卸载
2. **运动连贯性**：确保您的运动指令是连贯的，这样循环使用时不会出现跳跃
3. **重叠帧设置**：重叠帧过多可能增加计算量，过少可能导致拼接不自然
4. **参考图片质量**：高质量的参考图片对最终生成效果至关重要
5. **片段数量**：虽然支持生成无限个片段，但受限于磁盘空间和生成时间

## 输出

脚本会将生成的无限长视频保存到 `--output_path` 所指定的目录，默认输出路径为 `./outputs_infinite`，最终文件名为 `infinite_video.mp4`。

## 示例

### 生成一个持续前进的长视频

```bash
python generate_infinite_video.py \
  --pose "w-100" \
  --prompt "A desert landscape with sand dunes and a clear blue sky" \
  --model_path "./models/hunyuanvideo-1.5" \
  --action_ckpt "./models/action_model.pth" \
  --image_path "./input/desert.jpg" \
  --resolution 480p \
  --model_type bi \
  --num_segments 10 \
  --overlap 32
```

### 生成一个循环旋转的视频

```bash
python generate_infinite_video.py \
  --pose "left-16, w-5, right-16, s-5" \
  --prompt "A modern cityscape at night with bright lights" \
  --model_path "./models/hunyuanvideo-1.5" \
  --action_ckpt "./models/action_model.pth" \
  --image_path "./input/city.jpg" \
  --resolution 480p \
  --model_type bi \
  --num_segments 20
```