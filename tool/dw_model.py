from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-14B-AWQ",  # 模型仓库 ID，格式正确
    local_dir="../models/Qwen3-14B-AWQ",  # 本地保存路径
    local_dir_use_symlinks=False,  # 避免在不支持 symlink 的文件系统出错
)
