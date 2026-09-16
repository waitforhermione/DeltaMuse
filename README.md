# piano_to_harmonica_score — Delta Harmonica Score

纯钢琴 `.mp3` / `.wav` → 自动转录为音符 → 提取单声部主旋律 → 适配《三角洲行动》口琴键位 → 渲染成一张可直接照着弹的 PNG 曲谱。

<!-- TODO: 截图（GUI 主窗口 + 生成的曲谱 PNG）放在这里 -->

## Download（Windows）

1. 到 [Releases](../../releases) 下载最新的 `DeltaHarmonicaScore-v1.0.0-windows-x64.zip`
2. 解压到任意文件夹
3. 双击 `DeltaHarmonicaScore.exe` — **无需安装 Python 或任何依赖**

## How to use the GUI

1. 选择（或拖入）钢琴 `.mp3` / `.wav`，或 `.mid` / `.midi`
2. 选择 Entire song 或 Segment（填 `00:48` / `01:22` 这类时间）
3. 选择 Style：Practice（视奏友好，默认）或 Compact（总览）
4. 点 **Generate Score**，完成后 **Open Score** 直接打开 PNG

高级参数（移调、range mode、section 时长、context 秒数）折叠在 Advanced settings 里，普通用户无需关心。

## Supported formats

输入：`.mp3` / `.wav`（钢琴独奏录音）、`.mid` / `.midi`。输出：PNG 曲谱（另存 score.json 可选）。

## CLI usage

```bash
# 整首
python main.py convert song.mp3

# 只转换一段（副歌），只解码/转录该段附近，谱面时间从 0 开始
python main.py convert song.mp3 --start 00:48 --end 01:22

# 更适合视奏的排版（大字号、宽行距；--style 默认已是 practice）
python main.py convert song.mp3 --start 00:48 --end 01:22 --style practice
```

输出自动保存在输入文件旁边，命名为 `song_score.png` / `song_00m48s-01m22s_score.png`；
同名文件不会被覆盖（自动加 `_2`、`_3`）。用 `--verbose` 查看耗时与中间统计，
用 `--debug` 查看完整 traceback。个人偏好（默认 style、区间上下文秒数等）可保存：

```bash
python main.py config --set default_style compact   # 总览模式；默认 practice
python main.py config                                # 查看当前配置
```

**当前进度：Milestone 1–5 + 可用性/性能优化已完成。**

```text
piano audio ──► TransKun V2 ──► polyphonic NoteEvent[]
                                     │
                                     ▼
                        MelodyExtractor (note 或 SKIP)
                                     │
                                     ▼
                       strictly monophonic melody
                                     │
                     transpose ──► range fit ──► map
                                     │
                                     ▼
        PlayableNote[] ──► StaticScoreLayout ──► LayoutScore ──► PNG
```

尚未实现：SVG / PDF / GUI（均为明确非目标）。

---

## 安全边界

本项目只做**音频分析、音乐转录、曲谱转换、静态谱面生成**。

程序永远不会替代用户演奏：不包含 `SendInput`、`keybd_event`、`mouse_event`、`PostMessage`、AutoHotkey、键盘/鼠标宏、DLL 注入、DirectX Hook、游戏进程或内存读取等任何自动输入能力。程序的产出永远是**一份谱**，实际演奏由用户本人完成。

---

## Development

```bash
# 构建 Windows 独立包（输出 release/DeltaHarmonicaScore-v1.0.0-windows-x64.zip）
./scripts/build_windows.ps1
```

GUI 入口：`python main.py gui`（或直接 `python main.py`）；GUI 与 CLI 共用 Qt-free 的
`app/application/conversion_service.py`，桌面端使用 PySide6，后台转换运行在 QThread 中。

## 环境要求

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.11+ | 本项目在 3.11.9 上验证 |
| FFmpeg | 可选 | **不是必需**。libsndfile ≥ 1.1 已内置 MP3 解码；仅当解码 `.m4a/.aac/.wma` 等容器时才需要 FFmpeg |
| 磁盘 | ~3 GB | PyTorch + TransKun 模型 |
| 网络 | 仅安装时需要 | 测试与推理完全离线，不会下载任何模型 |

## 安装

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # Windows
.venv/bin/python   -m pip install -r requirements.txt        # macOS / Linux
```

Windows 上如果 `ncls` 报 `Microsoft Visual C++ 14.0 or greater is required`，先装预编译 wheel 再重试：

```bash
.venv\Scripts\python -m pip install ncls==0.0.68
.venv\Scripts\python -m pip install -r requirements.txt
```

CPU 推理即可：10.8 秒钢琴音频约 12 秒完成转录。有 CUDA 时加 `--device cuda`。

---

## 快速开始

```bash
# 生成示例音频（合成钢琴乐句，无需网络）
python samples/generate_sample_audio.py

# 分析：打印时长、音符数、音域、复音度
python main.py analyze samples/sample.wav

# 转录：音频 -> MIDI（复音）
python main.py transcribe samples/sample.wav --output samples/sample.mid

# 提取主旋律：复音音符 -> 严格单声部 MIDI（可含 rest）
python main.py extract-melody samples/sample.mid --strategy continuity \
  --output samples/melody_continuity.mid

# 旋律 -> 口琴键位谱（score.json）
python main.py map-melody samples/melody_continuity.mid \
  --transpose auto --range octave_fold \
  --output samples/melody_continuity.score.json

# 计算静态谱布局（调试 JSON，不生成图片）
python main.py layout-score samples/melody_continuity.score.json \
  --section-duration 6 \
  --output samples/melody_continuity.layout.json

# 渲染成 PNG 曲谱
python main.py render-score samples/melody_continuity.score.json \
  --output samples/melody_continuity.png

# 或者从音频一步到位：TransKun -> 旋律 -> 口琴键位 -> 布局 -> PNG
python main.py convert samples/sample.wav --output samples/sample_score.png
```

`render-score` 实际输出：

```text
Input             : samples\melody_continuity.score.json
Notes             : 7
Sections          : 2
Fragments         : 8
Canvas            : 2000 x 592
Output            : samples\melody_continuity.png
```

`convert` / `map-melody` 的输出格式由扩展名决定：`.json`（默认）输出 `score.json`，`.png` 输出渲染好的曲谱图片。

```bash
# 只转换一段（如副歌），谱面时间轴从片段起点重新归零
python main.py convert song.mp3 --start 00:48 --end 01:22 --style practice --output chorus.png
```

### 时间区间（--start / --end）

- 格式：纯秒（`48`、`82.5`）、`MM:SS`（`00:48`）、`MM:SS.sss`、`HH:MM:SS`（`00:01:22`）。
- 语义：只给 `--start` 转到结尾；只给 `--end` 从开头转；都不给=整首（旧行为不变）。
- 规则：区间外音符删除；跨边界音符截断；输出时间轴重新归零；`score.json` 写入
  `"source_segment": {"start": 48.0, "end": 82.0}`（未裁剪时为 `null`）；
  PNG header 显示 `Segment: 00:48–01:22`，section 时间仍从 `00:00` 开始。
- `--end` 超出媒体时长会被 clamp 并提示；`--start` 越界、`end <= start`、非法格式均为 `ERROR:` + 非零退出码。
- 支持命令：`convert`、`map-melody`、`extract-melody`。

### 视觉 preset（--style）

| preset | 定位 | 关键差异 |
| --- | --- | --- |
| `compact`（默认） | 总览 / 整曲导出，M5 原始观感 | 2000px 宽，section 220px，note 字号 13 |
| `practice` | 视奏 / 练习 | 2400px 宽，section 330px，note 字号 18，lane 标签加粗，网格/分隔更清晰 |

音乐语义在两种 preset 下完全一致（time→x、lane→y、duration→width 的比例关系不变），只改变比例与视觉权重。`render-score` / `convert` / `map-melody` 输出 PNG 时均支持 `--style`；`--section-duration` / `--canvas-width` 可在 preset 基础上继续覆盖。

### 视觉层级（M5.5）

- Header 两行元数据：来源行（Source 显示 basename · Profile · Duration）/ 生成行（Segment · Transpose · Notes · Playable），正数移调带 `+` 号。
- 每个 section 顶部一条通栏分隔线 + 左侧时间标签，长图换行一目了然。
- continuation 碎片使用浅蓝底/浅蓝边框（同音延续，不重复 label 的语义不变）；普通音符为深蓝。
- lane 与 note 标签使用粗体字体（小字号更清晰），字体回退逻辑不变。

### CLI 参数

```bash
python main.py analyze INPUT [--json REPORT.json] [--backend NAME] [--device auto|cpu|cuda|mps] [--debug]

python main.py transcribe INPUT [-o OUT.mid] [--backend NAME] [--device auto|cpu|cuda|mps]
                                [--segment-hop-size SEC] [--segment-size SEC] [--debug]

python main.py extract-melody INPUT [--strategy highest|lowest|continuity]
                                [--min-note-duration SEC] [--onset-epsilon SEC]
                                [-o OUT.mid] [--json REPORT.json] [--backend ...]

python main.py map-melody INPUT [--transpose auto|N] [--range octave_fold|nearest_note|drop]
                                [--profile PATH] [-o OUT.score.json] [--strategy ...] [--backend ...]

python main.py convert INPUT [--transpose ...] [--range ...] [--profile PATH]
                             [-o OUT.score.json] [--strategy ...] [--backend ...]

python main.py layout-score INPUT.score.json [--section-duration SEC] [--canvas-width PX]
                                [--title TEXT] [-o OUT.layout.json]

python main.py render-score INPUT.score.json [--section-duration SEC] [--canvas-width PX]
                                [--title TEXT] [-o OUT.png]
```

- `map-melody` 接受 MIDI 或音频；它会先跑旋律阶段（对已经是单声部的旋律等价于恒等变换），所以也能直接吃复音 MIDI。
- `convert` 是音频专用入口（MIDI 输入会报错并提示改用 `map-melody`）。
- `layout-score` 读取 `score.json`，输出**布局调试 JSON**（Milestone 4 没有图片）。
- `--profile` 允许替换口琴配置；默认使用随包的 `app/config/delta_harmonica.json`。
- `--debug` 显示完整 traceback 与第三方库警告；默认只输出 `ERROR: ...` 并返回非零退出码。

### 试听预览（可选）

```bash
python samples/synthesize_preview.py samples/melody_continuity.mid
```

用 pretty_midi 自带的简易合成器把任意 MIDI 渲染成 WAV，仅用于人工听检。它不是库的一部分，测试也不依赖它，且不需要音色库。

---

## 项目结构

```text
piano_to_harmonica_score/
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── app/
│   ├── audio/
│   │   ├── loader.py           # 解码 / mono / 归一化 / 重采样
│   │   ├── preprocessing.py
│   │   ├── transcriber.py      # PianoTranscriber 协议 + 后端工厂
│   │   └── backends/transkun_backend.py
│   ├── midi/
│   │   ├── midi_writer.py      # NoteEvent[] -> SMF
│   │   └── midi_loader.py      # SMF -> NoteEvent[]
│   ├── core/
│   │   ├── models.py           # NoteEvent, AudioBuffer, MelodyNote, PlayableNote
│   │   ├── errors.py
│   │   ├── polyphony.py        # onset 分组、重叠处理、复音统计
│   │   ├── melody_extractor.py # highest / lowest / continuity + SKIP
│   │   ├── instrument_profile.py  # InstrumentProfile + KeyBinding
│   │   ├── transposer.py       # manual / auto 移调
│   │   ├── range_fitter.py     # octave_fold / nearest_note / drop
│   │   ├── mapper.py           # canonical key mapping
│   │   └── converter.py        # 转换编排 + ConversionReport
│   ├── score/
│   │   ├── labels.py           # format_note_label（Z / Z# / Z↑ / Z↓2 ...）
│   │   ├── layout.py           # 静态谱布局（纯几何，禁止 Pillow）
│   │   ├── styles.py           # ScoreStyle：全部颜色/字号集中管理
│   │   └── renderer_png.py     # PNG 渲染（DrawSink 抽象 + 字体回退）
│   ├── export/
│   │   ├── json_exporter.py    # score.json
│   │   ├── score_loader.py     # score.json -> ScoreDocument
│   │   ├── layout_exporter.py  # 布局调试 JSON
│   │   └── image_exporter.py   # LayoutScore -> PNG 文件
│   ├── config/delta_harmonica.json
│   └── cli.py
├── tests/
└── samples/
    ├── generate_sample_audio.py
    └── synthesize_preview.py
```

## 分层架构

```text
Audio ──► Transcription ──► NoteEvent ──► Melody Processing
      ──► MelodyNote ──► PlayableNote ──► Static Score Layout ──► Renderer ──► PNG
```

各层禁止越界，这一点由代码结构强制保证：

- `MelodyExtractor` / `InstrumentProfile` / `Transposer` / `RangeFitter` / `Mapper` / `Converter` 只依赖标准库与 `NoteEvent`，完全不 import torch / TransKun / librosa / CLI；
- **`app/score/layout.py` 禁止 import PIL / matplotlib / PySide6**——它只做几何与数据；
- **`Renderer` 只接受 `LayoutScore`**：不读 MP3/MIDI、不接触 PlayableNote / TransKun / mapper / converter，也**不重新计算几何**（每个矩形/标签/网格线都直接来自布局）；
- 所有绘制走 `DrawSink` 抽象，因此渲染测试可以记录 draw call 而不需要 OCR；
- **mapper 只做 mapping**，range fitting 必须发生在 mapping 之前；
- 后端必须先由 `app.audio.loader` 解码，拿到 `AudioBuffer` 后再推理，**不重复解码**。

---

## 数据模型

```python
@dataclass(frozen=True)
class NoteEvent:      # 转录 / MIDI 输入
    pitch, start, duration, velocity, confidence

@dataclass(frozen=True)
class MelodyNote:     # M3 内部载体
    source_pitch, pitch, start, duration, velocity, confidence

@dataclass(frozen=True)
class PlayableNote:   # 已映射到具体按键
    source_pitch, pitch, start, duration, lane, key_label, semitone, octave_offset, confidence
```

`MelodyNote` 的 `source_pitch` 在 transpose / fold / nearest 之后**从不被改写**；`PlayableNote` 原样带回。`confidence` 一路透传（TransKun 2.0.1 不提供，因此为 `None` 是正常状态）。

---

## 主旋律提取（Milestone 2 + 2.5）

- **onset 分组**：组 anchor = 本组第一颗音的 onset；只有 `start - anchor <= onset_epsilon`(0.03 s) 才入组（与 anchor 比较，不会 chain merge）。
- **highest / lowest**：baseline，每组一棵音，**永不 SKIP**。
- **continuity**：带 SKIP 的 DAG 最短路径，边 `(i0,j0) → (i,j)` 表示中间组全部跳过，边成本 `(i-i0-1)*skip_cost + transition_cost`；因此重连时连续性基于最近一颗真实旋律音。复杂度 `O(组数 × 窗口 × 候选²)`。
- **SKIP 不免费**（`skip_cost = 2.5` 按组累加），且有 `max_skip_groups = 8` / `max_skip_seconds = 2.0` 双重护栏；**绝不补音**。
- 两个语义独立的阈值：`min_note_duration`(0.05 s, 原始短音) 与 `overlap_fragment_threshold`(0.03 s, 截断碎片)。

---

## 口琴键位转换（Milestone 3）

### Playable pitches（推导，不硬编码）

```text
pitch(lane, octave_offset, semitone)
  = base_pitch + natural_intervals[lane] + 12 * octave_offset + (1 if semitone else 0)
```

默认 profile（base 60，intervals 0/2/4/5/7/9/11/12，offsets -1/0/1）→ **playable = 48..85，38 个**，binding 48 个（测试确认）。可演奏性用 `playable_pitch_set` 成员判断，不是 min/max 区间。

### canonical 选择

`(|octave_offset|, semitone==False, lane)` 全序取最小（60 的三种按法中选 `Z/无修饰`）。

### auto transpose 排名

`(MIDI 越界数, 不可演奏数, fold 数, modifier_cost, |transpose|, transpose)`：modifier 权重 semitone=1 / octave=2，排在可演奏性之后。

### range fitting

| 模式 | 行为 |
| --- | --- |
| `octave_fold`（默认） | `pitch ± 12n` 中同 pitch class 且可演奏者，八度距离最小优先，平局取较低；无候选 → drop |
| `nearest_note` | 距离最小者，平局取较低 |
| `drop` | 删除并计数 |

---

## 静态谱布局（Milestone 4）

### 阅读方向

横轴 = 时间，纵轴 = 8 个 lane；一行是一个固定时长 section（默认 **6 s**），多个 section 从上到下排列。**不是纵向下落谱。**

### 核心数据结构

```python
ScoreLayoutConfig   # 全部几何参数集中管理（canvas/section/边距/lane/网格）
GridLine            # x, time, kind("major"|"minor")
LayoutNote          # 一段可绘制碎片（见下）
LayoutSection       # index, start_time, end_time, y, height, notes, grid
LayoutHeader        # title/source/profile/duration/transpose/playable_ratio/...
LayoutScore         # width, height, header, duration, lanes, sections
```

```python
@dataclass(frozen=True)
class LayoutNote:
    note_id: int              # 碎片自身 id（全曲顺序递增）
    source_note_id: int       # 来自哪颗 PlayableNote
    section_index: int
    x, y, width, height
    label                     # "Z" / "Z#" / "Z↑" / ",↑"
    lane
    start, duration           # 原始音符时间
    fragment_start, fragment_duration
    is_continuation: bool     # 跨 section 的后续碎片
    confidence: float | None
```

### 几何公式（均可独立测试）

| 项 | 公式 |
| --- | --- |
| section | `index = floor(start / section_duration)`，带 1e-9 相对容差；`start == section_end` 归下一节（6.0 s → 6–12 s 这一节） |
| usable_width | `canvas_width - left_label_width - right_padding`（默认 2000-80-24 = 1896） |
| time → x | `left_label_width + (time - section_start) / section_duration * usable_width` |
| lane → y | `section_y + (section_height - lane_count*lane_height)/2 + lane*lane_height + (lane_height - note_height)/2` |
| duration → width | `max(duration / section_duration * usable_width, min_note_width)` |
| 总高度 | `top_padding + header_height + section_count*section_height + bottom_padding` |

- section 的 y 由 `lane_count` 与配置决定，8 个 lane **互不重叠、顺序正确、都在 section 内**（穷举测试）。
- lane 顺序完全由 profile lanes 数据驱动（`Z X C V B N M ,`），布局阶段**不会从 pitch 重新推导**。

### 长音跨 section

音符是半开区间 `[start, end)`。一个跨边界的长音会拆成多个 fragment：

```text
note 5.5 → 7.5（section 6 s）
  section 0: 5.5 → 6.0   fragment_duration 0.5   is_continuation False
  section 1: 6.0 → 7.5   fragment_duration 1.5   is_continuation True
```

- 同一 source note 的所有 fragment `fragment_duration` 之和 == 原 `duration`（测试验证）；
- **结束时恰好落在边界上的音符不会产生 0 宽碎片**（`section_index_for_end` 会退回上一节）；
- `fragment_start / fragment_duration` 忠实保留，不会把所有碎片都写成原始 duration。

### modifier label

由 `format_note_label(note, symbols)` 生成，顺序固定 `key → # → octave`：

| semitone | octave | label |
| --- | --- | --- |
| 否 | 0 | `Z` |
| 是 | 0 | `Z#` |
| 否 | +1 / -1 | `Z↑` / `Z↓` |
| 是 | +1 / -1 | `Z#↑` / `Z#↓` |
| 否 | +2 / -2 | `Z↑2` / `Z↓2` |

渲染器（M5）只画 `label`，不再理解 semitone / octave_offset。

### 时间网格

布局阶段就生成 `GridLine(x, time, kind)`：默认 major 每 1.0 s、minor 每 0.5 s，major/minor 不重复、按时间排序。6 s section → major `0..6`（7 条）、minor `0.5..5.5`（6 条），共 13 条。

### 最后一节

最后一节可以短于完整 6 s（例如 14.2 s → 0–6 / 6–12 / 12–14.2），但 **x 比例固定用 section_duration**，不会把最后 2.2 s 拉伸到整行宽，保证所有行时间比例一致。

### duration

`layout_duration = max(score_duration, max(note.start + note.duration))` —— 不是简单取最后一个音符的 start。

---

## PNG 渲染（Milestone 5）

### 架构

| 模块 | 职责 |
| --- | --- |
| `app/score/styles.py` | `ScoreStyle`：全部颜色、字号、线宽、圆角集中管理 |
| `app/score/renderer_png.py` | `render_score(layout, …) -> PIL.Image`：纯绘制，可注入 `DrawSink` |
| `app/export/image_exporter.py` | `export_score_png(layout, path)`：写文件，PIL 异常包装成 `ScoreRenderError` |

`DrawSink` 是渲染器唯一的绘制出口（`rectangle` / `rounded_rectangle` / `line` / `text` / `text_length`），默认实现包装 `PIL.ImageDraw`；测试注入 `RecordingSink` 记录全部 draw call。

### 绘制顺序（固定）

```text
background -> header -> lane rows -> grid lines -> note blocks -> note labels
```

- **Header**：标题 + 一行元数据（Source / Profile / Duration / Transpose / Notes / Playable），空字段优雅省略，不显示 Python 字段名。
- **Section 标题**：每节左侧 `00:00–00:06`（最后一节 `00:06–00:08.97`）。
- **Lane**：左侧每节重复 `Z X C V B N M ,`（来自 `layout.lanes`，不硬编码）；奇数行浅色交替背景 + 每行底边一条淡线。
- **Grid**：直接使用 M4 的 `GridLine`，minor 先画、major 后画（major 更明显但不抢音符）；渲染器**不重新计算时间**。
- **Note**：按 `LayoutNote.x/y/width/height` 画圆角矩形（宽度不足以圆角时退化为矩形）。
- **Label**：使用 `LayoutNote.label`（`Z` / `Z#` / `Z↑` / `B#↑` / `M↑2`），白色置于音符内部；`is_continuation=True` 的碎片只画条、**不重复 label**；碎片太窄放不下标签时跳过。
- **Clipping**：`min_note_width` 可能让碎片略越过绘制区右缘，渲染器裁剪到 `canvas_width - right_padding`（必测项）。

### 字体

按 `Segoe UI → Arial → 微软雅黑 → DejaVu Sans → Calibri → Ubuntu` 顺序尝试，并实际检查 `↑ ↓ – #` 四个字形能否绘制（`getmask().getbbox()`），全部失败回落 Pillow 内置字体。渲染因此永不因字体崩溃。

### 确定性

相同 layout + style 的绘制调用序列完全一致（测试对比两次渲染的全部 draw call）。PNG 字节不要求逐位一致（Pillow 压缩可能变化），但图像尺寸与内容确定。

### 演示文件

- `samples/melody_continuity.png` — 真实 sample（7 notes / 2 sections / 8 fragments）
- `samples/renderer_demo.score.json` + `samples/renderer_demo.png` — 人工构造，覆盖 `Z` `X#` `C↑` `V↓` `B#↑` `M↑2` `V#` `Z↓` 与一个跨 section 长音

---

## score.json

```json
{
  "version": 1,
  "profile": "delta_harmonica",
  "lanes": ["Z", "X", "C", "V", "B", "N", "M", ","],
  "symbols": { "semitone": "#", "octave_up": "↑", "octave_down": "↓" },
  "duration": 8.971354166666666,
  "conversion": { "transpose": -7, "auto_transpose": true, "range_mode": "octave_fold" },
  "report": { "input_note_count": 7, "output_note_count": 7, "playable_ratio": 1.0, "...": "..." },
  "notes": [
    {
      "id": 0, "start": 0.0, "duration": 0.5640625,
      "source_pitch": 72, "pitch": 65, "lane": 3, "key": "V",
      "semitone": false, "octave": 0, "confidence": null
    }
  ]
}
```

- 每颗 note 完整表达：**什么时候弹、按哪个键、是否半音、是否升降八度**，并保留 `source_pitch`。
- `confidence` 为 M4 起新增的**可选**字段（null 表示后端不提供），旧文件缺省时按 `null` 读取。
- **确定性**：同样 melody + config 连续导出两次，字节完全一致（SHA256 验证）。无时间戳、无随机 id、字段顺序固定、float 不截断、写文件强制 `\n`。

---

## 转录后端

| 项 | 值 |
| --- | --- |
| 后端 | TransKun V2（no pedal extension，官方 pip 包内置权重） |
| 包版本 | `transkun==2.0.1` |
| 权重 / 配置 | `transkun/pretrained/2.0.pt`、`2.0.conf`（随包发布，安装即得） |
| 模型类 | `transkun.ModelTransformer.TransKun`（由 `2.0.conf` 指定） |
| 采样率 | 44 100 Hz |
| 分段 | 默认使用 checkpoint 自身配置（hop 8 s / segment 16 s） |
| 设备 | `--device auto`（CUDA → MPS → CPU） |
| 置信度 | 不提供逐音符置信度 → `confidence = None` |

架构上保留了 backend abstraction：`BACKENDS` 注册表 + `PianoTranscriber` 协议，未来接入 Aria-AMT / ByteDance Piano 只需新增一个 adapter 模块。

---

## 测试

```bash
.venv\Scripts\python -m pytest -q
```

| 测试文件 | 用例数 | 覆盖 |
| --- | --- | --- |
| `tests/test_audio_loader.py` | 33 | WAV/MP3、mono、归一化、重采样、异常输入 |
| `tests/test_models.py` | 34 | 四个数据模型的全部不变量 |
| `tests/test_transcriber.py` | 28 | 协议、后端工厂、事件转换、真实模型集成（可选） |
| `tests/test_midi_writer.py` | 10 | MIDI 写出、pretty_midi/mido 双库校验 |
| `tests/test_midi_loader.py` | 12 | MIDI 读入、轨道元数据、损坏文件 |
| `tests/test_polyphony.py` | 37 | onset 分组（anti-chain-merge）、重叠处理、复音统计 |
| `tests/test_melody_extractor.py` | 78 | 代价模型、三种策略、rest/gap、不变量、规模 |
| `tests/test_melody_skip.py` | 39 | SKIP 行为、窗口上限、稀疏转录回归 |
| `tests/test_instrument_profile.py` | 55 | profile 加载/校验、派生 pitches、binding、round-trip |
| `tests/test_transposer.py` | 33 | manual/auto 移调、MIDI 边界、排名、deterministic argmin |
| `tests/test_range_fitter.py` | 28 | fold 上/下/平局/无候选、nearest 平局、drop |
| `tests/test_mapper.py` | 17 | 全 playable pitch 穷举 + round-trip、canonical |
| `tests/test_converter.py` | 31 | 管线顺序、provenance、报告、三种 range mode |
| `tests/test_json_exporter.py` | 17 | schema、utf-8 符号、字节级确定性 |
| `tests/test_score_loader.py` | 24 | score.json 导入、校验、可选字段 |
| `tests/test_score_labels.py` | 24 | Z / Z# / Z↑ / Z↓ / Z#↑ / Z#↓ / Z↑2 / Z↓2 |
| `tests/test_layout.py` | 71 | section 分配、x/y/width、跨节碎片、边界、网格、确定性 |
| `tests/test_layout_exporter.py` | 9 | 布局调试 JSON schema 与确定性 |
| `tests/test_renderer.py` | 43 | draw call 记录、lane/section 标题、几何、裁剪、continuation、header/Segment、绘制顺序、字体回退 |
| `tests/test_time_range.py` | 40 | 时间解析、区间规范化、裁剪规则 |
| `tests/test_style_presets.py` | 10 | compact / practice preset 与音乐语义不变性 |
| `tests/test_cli.py` | 69 | 七个子命令、`--start/--end`、`--style`、报告、错误处理 |
| **合计** | **743** | 全部通过（约 4 秒） |

测试完全离线：音频夹具由 numpy + soundfile 现场合成，后端用桩替换，不下载任何模型。

真实模型集成测试为**可选**（需要约 10 秒 CPU 推理）：

```bash
set PIANO_SCORE_RUN_MODEL_TESTS=1
.venv\Scripts\python -m pytest tests/test_transcriber.py -q
```

---

## 已知限制

1. **只支持纯钢琴**。无主唱、无鼓、无贝斯、无复杂混音；不做人声分离、不做多乐器转录。
2. **没有逐音符置信度**。TransKun 2.0.1 不暴露该信息，`confidence` 恒为 `None`；接口与 JSON 字段已保留，M5 可据此画虚线/淡化样式。
3. **纯正弦/电子音色识别不出来**；短音符与接近文件结尾仍在发声的音符易漏。
4. **SKIP 会让旋律出现 rest**（设计目标）；布局阶段 rest 表现为该时间段的空 lane。
5. **极低音区的旋律可能被跳过**（C2–E2）。
6. **auto transpose 会为减少 modifier 而整体移调**（示例中 -7）；想保持原调用 `--transpose 0`。
7. **`min_note_width` 可能让碎片略微越过右侧 padding**：渲染器已裁剪到绘制区右缘，因此极短碎片在画面上可能只有 1–2 px 宽。
8. **布局输出的最后一节按完整 section_duration 计算几何**（时间比例统一），渲染时右侧留白是正确行为。
9. **PNG 字节不保证逐位一致**（Pillow 压缩/元数据可能变化），但图像尺寸与绘制内容确定；布局调试 JSON 与 score.json 仍是字节级确定。
10. **极短碎片的标签会被省略**（放不下时不绘制），跨节 continuation 也不重复标签——这是设计行为，读谱时以第一段标签为准。
11. 第一版只有一套浅色打印友好配色；霓虹/游戏 HUD 风格不在范围内。
12. 不支持 SVG / PDF / GUI / 自动播放 / 编辑器。

---

## 里程碑

## Acknowledgements

This project was built with the help of AI coding assistants:

- **DeepSeek V4.1 Flash**
- **GLM-5.3 Flash**

## 里程碑

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M1 | Audio → NoteEvent → MIDI，CLI `analyze` / `transcribe` | ✅ 已完成 |
| M2 | 复音转单音、`highest`/`lowest`/`continuity`、`melody.mid` | ✅ 已完成 |
| M2.5 | continuity 增加 SKIP 状态，允许 rest | ✅ 已完成 |
| M3 | InstrumentProfile、auto transpose、range fitting、mapping、`score.json` | ✅ 已完成 |
| M4 | 静态谱布局（sections / grid / fragments / 几何 / 调试导出） | ✅ 已完成 |
| M5 | Pillow PNG 渲染、`render-score` / `convert → PNG` 全链路 | ✅ 已完成 |

## Acknowledgements

This project was built with the help of AI coding assistants:

- **DeepSeek V4.1 Flash**
- **GLM-5.3 Flash**

Both models contributed code, test design and documentation across all
milestones. All decisions, review and acceptance were done by the human author.
