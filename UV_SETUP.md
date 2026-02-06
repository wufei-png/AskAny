# UV Python 环境配置指南

## 1. Zsh 中使用 UV

### 1.1 激活虚拟环境

在项目目录下，使用以下方式激活 uv 创建的虚拟环境：

```bash
# 方式1：直接激活（推荐）
source .venv/bin/activate

# 方式2：使用 uv 命令
uv venv  # 如果虚拟环境不存在，先创建
source .venv/bin/activate
```

### 1.2 在 Zsh 中自动激活虚拟环境（可选）

将以下内容添加到 `~/.zshrc` 文件中，实现进入项目目录时自动激活虚拟环境：

```bash
# 自动激活 uv 虚拟环境
function auto_activate_venv() {
    if [[ -f .venv/bin/activate ]]; then
        source .venv/bin/activate
    fi
}

# 进入目录时检查并激活
autoload -U add-zsh-hook
add-zsh-hook chpwd auto_activate_venv

# 进入项目时自动激活
auto_activate_venv
```

### 1.3 常用 UV 命令

```bash
# 安装 Python 版本
uv python install 3.11

# 固定项目使用的 Python 版本
uv python pin 3.11

# 同步依赖（安装/更新所有依赖）
uv sync

# 添加新依赖
uv add package-name

# 运行 Python 脚本
uv run python script.py

# 运行命令（自动使用虚拟环境）
uv run python -m askany.main --serve
```

### 1.4 获取 Python 路径

```bash
# 获取虚拟环境中的 Python 路径
which python  # 激活虚拟环境后
# 或
readlink -f .venv/bin/python

# 当前项目的 Python 路径：
# AskAny/.venv/bin/python
# 实际指向：
# /home/wufei/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11
```

## 2. Python 插件配置（VS Code / Cursor）

### 2.1 创建 VS Code 工作区配置

在项目根目录创建 `.vscode/settings.json` 文件：

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.terminal.activateEnvInCurrentTerminal": true,
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "python.analysis.autoImportCompletions": true,
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "[python]": {
        "editor.defaultFormatter": "black",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

### 2.2 手动选择解释器

1. 按 `Ctrl+Shift+P` (或 `Cmd+Shift+P` on Mac)
2. 输入 `Python: Select Interpreter`
3. 选择 `.venv/bin/python` 或输入路径：
   ```
   AskAny/.venv/bin/python
   ```

### 2.3 验证配置

在 VS Code/Cursor 的终端中运行：

```bash
which python
python --version
```

应该显示：
- `python` 路径：`AskAny/.venv/bin/python`
- `python` 版本：`Python 3.11.14`

## 3. 项目特定配置

### 3.1 当前项目信息

- **项目路径**: `AskAny`
- **虚拟环境路径**: `.venv/`
- **Python 版本**: 3.11.14
- **UV Python 路径**: `/home/wufei/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11`

### 3.2 快速启动命令

```bash
# 激活环境并运行服务
source .venv/bin/activate && python -m askany.main --serve

# 或使用 uv run（无需激活）
uv run python -m askany.main --serve
```

## 4. 故障排除

### 4.1 Python 插件找不到解释器

1. 确保 `.venv` 目录存在
2. 运行 `uv sync` 确保虚拟环境已创建
3. 在 VS Code/Cursor 中重新加载窗口：`Ctrl+Shift+P` -> `Developer: Reload Window`

### 4.2 导入错误

确保已安装所有依赖：
```bash
uv sync
```

### 4.3 路径问题

如果遇到路径问题，使用绝对路径：
```bash
# 在 .vscode/settings.json 中使用绝对路径
"python.defaultInterpreterPath": "AskAny/.venv/bin/python"
```

## 5. 推荐工作流

1. **首次设置**:
   ```bash
   cd AskAny
   uv sync
   source .venv/bin/activate
   ```

2. **日常开发**:
   ```bash
   # 进入项目目录（如果配置了自动激活，会自动激活虚拟环境）
   cd AskAny
   
   # 或手动激活
   source .venv/bin/activate
   
   # 运行项目
   python -m askany.main --serve
   ```

3. **添加新依赖**:
   ```bash
   uv add package-name
   # 或编辑 pyproject.toml 后运行
   uv sync
   ```

