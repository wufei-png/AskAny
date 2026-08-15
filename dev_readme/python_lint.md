uv sync --all-extras                              # 安装锁定的开发与可选依赖
uv run --locked ruff check askany test            # 检查受支持代码
uv run --locked ruff check --fix askany test      # 自动修复
uv run --locked ruff format askany test           # 格式化
uv run --locked --all-extras pyright              # standard 类型检查
