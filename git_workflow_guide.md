# Git 推送学习笔记到 GitHub 操作指南

## 仓库信息

| 项目 | 内容 |
|------|------|
| GitHub 仓库 | [ienjoy/GPU_Learning_Notes](https://github.com/ienjoy/GPU_Learning_Notes) |
| 本地目录 | `/Users/xieqihua/GPU365` |
| 分支 | `main` |

---

## 日常操作：新建/修改笔记后推送

```bash
cd /Users/xieqihua/GPU365

git add .
git commit -m "简短描述本次变更"
git push
```

---

## .gitignore 已排除的内容

```
pplx-garden/   # 第三方开源项目，不推送
.venv/         # Python 虚拟环境，不推送
.DS_Store      # macOS 系统文件
```

---

## 首次配置（已完成，仅供参考）

### 1. 在根目录初始化 git repo

```bash
cd /Users/xieqihua/GPU365
git init
git branch -M main
```

### 2. 关联远程仓库

```bash
# 使用 PAT（Personal Access Token）认证
git remote add origin https://<用户名>:<PAT>@github.com/ienjoy/GPU_Learning_Notes.git
```

> ⚠️ **安全提醒**：PAT 明文写在命令里会留在终端历史记录中。
> 建议操作：
> 1. 用完后在 GitHub Settings → Developer settings → Personal access tokens 里 **Regenerate**
> 2. 或者改用 SSH Key 认证（更安全）

### 3. 首次推送

```bash
git add .
git commit -m "Initial commit"
git push -u origin main
```

---

## 处理子目录有独立 .git 的情况

如果某个子目录（如 `notes/`）之前有自己的 `.git`，直接在父目录 `git add` 会把它当成 submodule，导致内容推不上去。

**解决办法**：先删除子目录的 `.git`，再从父目录 push。

```bash
# 先取消子目录的 git 追踪
cd /Users/xieqihua/GPU365/notes
git rm -r --cached .

# 然后在父目录正常 add/commit/push
cd /Users/xieqihua/GPU365
git add .
git commit -m "合并子目录内容"
git push
```

---

## SSH Key 认证配置（更安全的替代方案）

```bash
# 1. 生成 SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 复制公钥
cat ~/.ssh/id_ed25519.pub

# 3. 在 GitHub 粘贴：Settings → SSH and GPG keys → New SSH key

# 4. 测试连接
ssh -T git@github.com

# 5. 修改 remote 为 SSH 地址
git remote set-url origin git@github.com:ienjoy/GPU_Learning_Notes.git
```

---

## 常用 Git 命令速查

```bash
git status          # 查看哪些文件有变更
git log --oneline   # 查看提交历史
git diff            # 查看未暂存的具体改动
git remote -v       # 查看远程仓库地址
```
