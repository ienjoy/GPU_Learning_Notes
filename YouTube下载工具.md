
# YouTube 等平台音频/视频下载方法（开源 Python 工具篇）

## 一、核心引擎：yt-dlp（最推荐）

这是所有下载工具的基础，功能最强，支持上千个网站（YouTube、B站、Vimeo等）。

### 安装
```bash
pip install yt-dlp


基本使用
1. 下载音频（MP3，最佳质量）
bash
yt-dlp -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 "视频链接"

2. 下载视频（最佳画质）

bash
yt-dlp -f best "视频链接"

3. 下载整个播放列表

bash
yt-dlp -f bestaudio --extract-audio --audio-format mp3 -o "%(playlist_title)s/%(title)s.%(ext)s" "播放列表链接"

常用参数说明
参数	说明
-f bestaudio	选择最佳音频流
--extract-audio	提取音频
--audio-format mp3	转换为 MP3 格式
--audio-quality 0	最佳音频质量（0=最好）
-o "路径/%(title)s.%(ext)s"	自定义输出文件名和路径

二、封装好的开源 Python 工具

这些项目基于 yt-dlp 或类似库，提供更简单的使用方式。

1. ytdlp-simple（适合开发者集成）

特点：零配置，自动处理 FFmpeg 依赖，特别适合 AI 应用（如为 Whisper 准备音频）

安装：pip install ytdlp-simple

代码示例：

python
from ytdlp_simple import download_audio

# 下载音频并返回文件路径

audio_path = download_audio("https://www.youtube.com/watch?v=TzKhHmngWzA")
print(f"音频已下载至: {audio_path}")

2. youtube-music-downloader（适合批量下载播放列表）

特点：专为 YouTube Music 设计，自动嵌入封面图和元数据（歌手、专辑等）

安装：

bash
git clone https://github.com/insaiyancvk/youtube-music-downloader.git
cd youtube-music-downloader
pip install -r requirements.txt

使用：

bash
python main.py --playlist "播放列表URL"

3. VideoDownloader（带图形界面 + AI 人声分离）
特点：Tkinter 图形界面，支持 320kbps MP3 转换，内置 AI 人声分离（Demucs）

安装：

bash
git clone https://github.com/abhinavsingh/VideoDownloader.git
cd VideoDownloader
pip install -r requirements.txt
python main.py
4. youtube-downloader（终端交互界面 TUI）
特点：终端内的交互式界面，需要 YouTube API 密钥

安装：

bash
git clone https://github.com/notandy/youtube-downloader.git
cd youtube-downloader
pip install -r requirements.txt
配置：需要申请 YouTube Data API v3 密钥，放在 config.json 中

5. fetchtube（极简轻量）
特点：代码简单，无复杂依赖，适合学习和简单场景

安装：pip install fetchtube

代码示例：

python
import fetchtube

# 下载视频
fetchtube.download("https://www.youtube.com/watch?v=TzKhHmngWzA", "output.mp4")

# 仅下载音频
fetchtube.download("https://www.youtube.com/watch?v=TzKhHmngWzA", "output.mp3", audio_only=True)
三、必备依赖：FFmpeg
处理音频转换（如下载的 WebM 转 MP3）需要 FFmpeg。

安装方法
Windows：

下载 https://ffmpeg.org/download.html

解压后，将 bin 文件夹路径添加到系统环境变量 PATH

macOS：

bash
brew install ffmpeg
Linux (Ubuntu/Debian)：

bash
sudo apt update
sudo apt install ffmpeg
验证安装：

bash
ffmpeg -version
四、快速对比表
工具	类型	图形界面	下载播放列表	嵌入元数据	人声分离	适合人群
yt-dlp	命令行	❌	✅	❌	❌	进阶用户、开发者
ytdlp-simple	Python库	❌	✅	❌	❌	AI/开发者
youtube-music-downloader	命令行	❌	✅	✅	❌	音乐爱好者
VideoDownloader	图形界面	✅	❌	❌	✅	新手、音频处理
youtube-downloader	终端TUI	✅	✅	❌	❌	终端爱好者
fetchtube	Python库	❌	❌	❌	❌	初学者、极简需求
五、针对你提供的视频的具体命令
你提供的链接：https://www.youtube.com/watch?v=TzKhHmngWzA

方法1：使用 yt-dlp 下载最佳质量 MP3
bash
yt-dlp -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 "https://www.youtube.com/watch?v=TzKhHmngWzA"
方法2：下载并指定输出文件名
bash
yt-dlp -f bestaudio --extract-audio --audio-format mp3 --audio-quality 0 -o "Spring_Waltz.%(ext)s" "https://www.youtube.com/watch?v=TzKhHmngWzA"
方法3：下载视频（最高画质）
bash
yt-dlp -f best "https://www.youtube.com/watch?v=TzKhHmngWzA"
六、注意事项
版权问题：请仅下载自己有权使用的内容，遵守相关平台的服务条款。

网络环境：某些地区可能需要特殊网络配置才能访问 YouTube。

更新工具：YouTube 经常变更接口，建议定期更新：

bash
pip install --upgrade yt-dlp
播放列表下载：下载整个播放列表时请尊重创作者，不要频繁大量请求。

七、延伸：在线替代方案（无需安装）
如果不想安装任何软件，可以使用以下在线服务（适合偶尔下载）：

YTMP3：https://ytmp3.cc

EzMP3：https://ezmp3.cc

Y2Mate：https://y2mate.com

快速技巧：将 YouTube 链接中的 youtube 改为 ssyoutube，例如：
https://www.ssyoutube.com/watch?v=TzKhHmngWzA

参考资料
yt-dlp GitHub：https://github.com/yt-dlp/yt-dlp

FFmpeg 官网：https://ffmpeg.org

文中各项目的 GitHub 地址已分别在对应章节给出

text

**使用方法**：
1. 全选并复制上面的全部内容（从 \`\`\`markdown 到 最后的 \`\`\`）
2. 打开记事本（Windows）或文本编辑（Mac）
3. 粘贴内容
4. 点击「文件」→「另存为」
5. 文件名输入 `YouTube下载工具.md`，编码选择 UTF-8
6. 保存即可

之后就可以用支持 Markdown 的软件（如 Typora、VS Code、Obsidian 等）打开查看了。