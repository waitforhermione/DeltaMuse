# Delta Harmonica Score

把钢琴独奏录音（MP3/WAV）自动转换成《三角洲行动》口琴键位的静态曲谱图片（PNG）。

## 功能说明

- **自动转录**：使用 TransKun V2 模型识别钢琴录音中的每个音符（音高、时间、力度）
- **主旋律提取**：从复音钢琴中提取单声部主旋律（DP 路径搜索，自动跳过伴奏）
- **口琴映射**：自动移调适配键位范围，生成每个音对应的按键（`Z` `X#` `C↑` `B#↑` 等）
- **片段转换**：只转换某一段（如副歌），只解码该段附近音频，谱面时间从 0 开始
- **两种排版**：Practice（视奏友好，默认）/ Compact（整曲总览）
- **纯离线**：模型权重随程序打包，安装后无需联网

## 如何使用

### Windows 桌面版（推荐）

1. 到 [Releases](https://github.com/waitforhermione/DeltaMuse/releases) 下载 `DeltaHarmonicaScore-v1.0.0-windows-x64.zip`
2. 解压到任意文件夹，双击 `DeltaHarmonicaScore.exe`（无需安装 Python）
3. 选择或拖入钢琴录音 → 选整首或某一段 → 点 **Generate Score**
4. 完成后点 **Open Score** 直接打开曲谱

### 命令行

```bash
# 整首
python main.py convert song.mp3

# 只转换一段，自动生成 song_00m48s-01m22s_score.png
python main.py convert song.mp3 --start 00:48 --end 01:22

# 指定排版样式（practice / compact）
python main.py convert song.mp3 --start 00:48 --end 01:22 --style practice

# 从 MIDI 转谱
python main.py map-melody song.mid

# 更多子命令：analyze / transcribe / extract-melody / render-score / config
python main.py --help
```

输入支持 `.mp3` / `.wav` / `.mid` / `.midi`，输出为 PNG 曲谱（可选 score.json）。
