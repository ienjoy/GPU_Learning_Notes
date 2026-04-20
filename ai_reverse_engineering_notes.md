# AI 逆向工程笔记

> 原文：[从逆向工程重新认识 AI 的强大](https://blog.huli.tw/2026/04/18/ai-reverse-engineering-op/)  
> 作者：Huli｜日期：2026-04-18

---

## 作者使用的工具与方式

- **AI 工具**：Cursor + Claude Opus 4.6 high thinking
- **操作方式**：给一个简单的指令，其余由 AI 自主完成
  - 典型 prompt：`xxx 资料夹底下有个 apk，把它逆向还原，要还原成原始码`
- **费用**：Cursor 计费下每个 App 约 < $5 美元；Claude Code 较贵，某个 Unity 游戏花了 4000 万 token ≈ $27 美元
- **人工介入**：主要是在 AI 卡住时给方向提示，例如"静态分析不行就试试动态分析"

---

## 精选案例（7 个）

### 案例一：Cocos2d 游戏（纯静态分析）

**流程：**
1. jadx 拆包 → 发现逻辑不在 Java 层
2. 找到 `libcocos2djs.so`，静态分析找到加解密函数
3. 识别出加密算法是 **Blowfish**
4. 追踪设置 key 的函数，发现 key 被逐字拼接（`key.push_back('7')...`，共 32 个字符）
5. 写 Python 脚本解密所有资源，还原出 JavaScript 游戏逻辑

**亮点：** AI 先扫描代码找连续字串失败后，推断出 key 是不连续拼接的，换方法追踪函数调用。

---

### 案例二：另一个 Cocos2d 游戏（静态 → 动态）

**流程：**
1. 静态分析找到疑似 key，但解不了加密的 jsc
2. 尝试 Unicorn Engine 模拟执行 → 卡住
3. 改用动态分析：安装 Android 模拟器 + Frida
4. Hook `xxtea_decrypt`（libcocos2djs.so 的 public 导出函数）
5. 游戏运行时拿到正确 key，还原所有 jsc 文件
6. 顺带用 frida-dexdump 脱了一个弱壳

---

### 案例三：Unity 游戏（密文模式分析 + 神来一笔）

**流程：**
```
XAPK 解包
  ↓
Il2CppDumper + ILSpy + JADX（标准流程）
  ↓
UnityPy 扫描 → 只有 4 个工具脚本
  ↓
找 AppConfig → 拿到 CDN 地址和 AES Key
  ↓
尝试下载 → 全部 404（死胡同）
  ↓
重新检查 AssetBundle → 找到 3000 个加密文件！
  ↓
观察密文 → 发现 6 bytes 重复模式
  ↓
❌ 错误假设：Lua bytecode → 推导出错误密钥流
❌ 尝试 AES-CBC/ECB/OFB/CTR/CFB → 全不匹配
❌ 尝试 RC4 → 不匹配
❌ 尝试 .NET Random（10K seeds）→ 不匹配
  ↓
逆向 libil2cpp.so → 确认是简单重复 XOR
  ↓
❌ 静态找 ENCRYPT_BYTES → 走不通
  ↓
回头看密文特征 → 1500 个文件共享 62 bytes 前缀
  ↓
💡 假设明文是 '-'(0x2d) 重复（Lua 注释 `-----`）→ XOR 得到 key → 解密成功！
```

**亮点：** AI 在写 Unicorn 模拟脚本的过程中，观察到密文规律，用"明文是长横线注释"的假设反推出 key，全部解开。

---

### 案例四：另一个 Unity 游戏（静态分析直接搞定）

**流程：**
1. C# dump 出来后，用 `encrypt` 关键字搜索，直接找到 `LuaEncryption` 的 key
2. 找到自定义 asset bundle offset（跳过前 12 bytes）
3. 写 Python 脚本：跳过头部 → 扫描 `UnityFS\x00` 标记 → 切割 bundle → XOR 解密
4. 得到 7000+ 个 Lua JIT bytecode，用 ljd 还原成可读 Lua 源码（共 1000 万行）

---

### 案例五：混淆过的 App

**流程：**
1. jadx 发现混淆，AI 自写反混淆脚本还原核心 class 名称
2. 基于 capstone + lief 反汇编 arm64-v8a so 文件，还原为 pseudo-C
3. 分析出加解密算法和 key

**亮点：** AI 读混淆代码的能力比人强。

---

### 案例六：银行级 App（商业壳）⭐

**这是难度最高的案例。**

**保护措施：**

| 层 | 保护手段 |
|---|---|
| Native | 扫描 `/proc/self/maps` 检测 Frida、ptrace 反调试、字串比对、.text 段加密 |
| Java | Root 检测、模拟器检测、SSL Pinning、anti-debug |

**流程：**
1. `objdump -h` 看 section headers → 发现非标准 section，.text 加密
2. 推断出商业壳厂商
3. 多种方法失败后，分析壳的运行机制，成功脱壳
4. 知道了所有保护手段后，逐一 hook 绕过
5. 写 1000 行 Python 脚本还原 Java 混淆字符串
6. **最终：App 成功打开，请求内容可以 hook，所有保护全部绕过**

---

### 案例七：双层加密游戏（换方向）

- dex 先用 so A 加密，so A 再用 so B 加密，so B 还加了壳
- AI 尝试一天，so 的壳没有完全脱掉
- **转换策略**：发现主要逻辑在 Lua 层，通过观察加密 hex 模式直接解出 Lua 资源
- 结论：壳没脱，但核心游戏逻辑拿到了，算半成功

---

## AI 的不足之处

- 有时只还原到 interface（方法定义），没有实现逻辑
- AI 会根据命名和结构**猜**运作逻辑，写出头头是道的报告，但实际没有真正还原
- **应对方法**：每次都追问 `"你有拿到原始码了吗？给我看登录系统的实现"`，逼 AI 做得更深

---

## 核心心得

### 1. 我才是 AI 的限制器
> "我明明没有试过，但觉得 AI 办不到。"

不要先假设 AI 做不到，什么都让它去试，做不到再说。

### 2. AI 逆向的本质
就是**做各种尝试并观察输出，再从输出中改善，或换个方法**。和人类逆向的思路完全一样，但 AI 更快、更有耐心、不怕重复。

### 3. 攻防平衡的变化
- 守方花大力气做的商业壳，AI 一小时可能就脱掉了
- 传统"增加难度延长破解时间"的策略，在 AI 面前效果大打折扣
- 未来守方可能需要"用 AI 来加壳"，每次生成不同的保护模式

---

## iOS 逆向补充（自己整理）

### Android vs iOS 工具对比

| 功能 | Android | iOS |
|---|---|---|
| 反编译 Java/OC | jadx | class-dump |
| Native 层分析 | IDA / Ghidra (.so) | IDA / Ghidra (Mach-O) |
| 动态 hook | Frida | Frida（同样支持） |
| 脱壳 | frida-dexdump | frida-ios-dump |
| 系统加密 | 商业壳 | FairPlay DRM |

### iOS 特有难点
1. **FairPlay 加密**：App Store 下载的 ipa 是加密的，需要越狱设备先 dump 明文二进制
2. **需要越狱设备**：比 Android 模拟器配置麻烦
3. **架构**：现代 App 都是 arm64，和 Android arm64-v8a 一样，工具链通用

### 设备选择

| 设备 | 可用性 | 原因 |
|---|---|---|
| iPhone 5 | ❌ 不推荐 | 最高 iOS 10，32位 ARMv7，现代 App 装不进去 |
| iPhone 8 Plus | ✅ 推荐 | A11 arm64，可越狱到 iOS 16，兼容性好 |
| Mac iOS 模拟器 | ✅ 免费首选 | 无需越狱，Frida 可用，arm64 |

### iPhone 8 Plus 越狱方案
- 工具：**palera1n**（基于 checkm8 硬件漏洞，苹果无法修补）
- 支持：iOS 15 / 16
- 类型：半捆绑越狱（每次重启需重新触发，不影响使用）
- 安装 Frida：通过 Sileo/Cydia 安装后，Mac 端 `frida-ps -U` 列出进程即可开始
