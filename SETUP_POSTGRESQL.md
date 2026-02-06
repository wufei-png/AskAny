# PostgreSQL 安装和配置指南

## 系统要求

- Ubuntu/Debian (WSL2 或其他 Linux 发行版)
- 或 macOS
- 或 Windows (通过 WSL2)

## 安装步骤

### 1. 安装 PostgreSQL

#### Ubuntu/Debian (WSL2)

```bash
# 更新包列表
sudo apt update

# 安装 PostgreSQL 和常用工具
sudo apt install -y postgresql postgresql-contrib

# 安装 PostgreSQL 开发包（用于编译 pgvector）
sudo apt install -y postgresql-server-dev-all

# 启动 PostgreSQL 服务
sudo service postgresql start

# 设置 PostgreSQL 开机自启
sudo systemctl enable postgresql
```

#### macOS (使用 Homebrew)

```bash
# 安装 PostgreSQL
brew install postgresql@15

# 启动 PostgreSQL 服务
brew services start postgresql@15
```

#### Windows (WSL2)

在 WSL2 中按照 Ubuntu/Debian 的步骤安装。

### 2. 配置 PostgreSQL 用户

```bash
# 切换到 postgres 用户
sudo -u postgres psql

# 在 PostgreSQL 命令行中执行：
# 创建用户（如果不存在）
CREATE USER root WITH PASSWORD '123456';

# 授予创建数据库权限
ALTER USER root CREATEDB;

# 授予超级用户权限（可选，用于创建扩展）
ALTER USER root WITH SUPERUSER;

# 退出
\q
```

或者使用命令行：

```bash
# 创建用户
sudo -u postgres createuser -s root

# 设置密码
sudo -u postgres psql -c "ALTER USER root WITH PASSWORD '123456';"
```

### 3. 安装 pgvector 扩展

#### 方法 1: 使用 apt (推荐，如果可用)

```bash
# Ubuntu 22.04+ 可能包含 pgvector 包
sudo apt install -y postgresql-15-pgvector
# 注意：版本号可能不同，根据你的 PostgreSQL 版本调整
```

#### 方法 2: 从源码编译安装

```bash
# 安装 git 和构建工具
sudo apt install -y git build-essential

# 克隆 pgvector 仓库
cd /tmp
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector

# 编译和安装
make
sudo make install

# 清理
cd ..
rm -rf pgvector
```

#### 方法 3: 使用 Docker (如果使用 Docker PostgreSQL)

```bash
# 使用包含 pgvector 的 PostgreSQL 镜像
docker run -d \
  --name postgres \
  -e POSTGRES_USER=root \
  -e POSTGRES_PASSWORD=123456 \
  -e POSTGRES_DB=askany \
  -p 5432:5432 \
  pgvector/pgvector:pg15
```

### 4. 创建数据库和启用扩展

```bash
# 创建数据库
createdb -h localhost -U root askany

# 或者使用 postgres 用户创建
sudo -u postgres createdb askany

# 连接到数据库并启用 pgvector 扩展
psql -h localhost -U root -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 验证扩展是否安装成功
psql -h localhost -U root -d askany -c "\dx"
```

### 5. 配置 PostgreSQL 允许本地连接

编辑 PostgreSQL 配置文件：

```bash
# 找到配置文件位置
sudo -u postgres psql -c "SHOW config_file;"

# 编辑 pg_hba.conf（通常在 /etc/postgresql/15/main/pg_hba.conf）
sudo nano /etc/postgresql/15/main/pg_hba.conf
```

确保有以下行（允许本地连接）：

```
# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
```

重启 PostgreSQL：

```bash
sudo service postgresql restart
```

### 6. 验证安装

```bash
# 测试连接
psql -h localhost -U root -d askany -c "SELECT version();"

# 检查 pgvector 扩展
psql -h localhost -U root -d askany -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

## 故障排除

### 问题 1: 无法连接到数据库

**错误**: `psql: error: connection to server at "localhost" (127.0.0.1), port 5432 failed`

**解决方案**:
```bash
# 检查 PostgreSQL 是否运行
sudo service postgresql status

# 如果未运行，启动它
sudo service postgresql start

# 检查端口是否监听
sudo netstat -tlnp | grep 5432
```

### 问题 2: 认证失败

**错误**: `password authentication failed for user "root"`

**解决方案**:
```bash
# 重置密码
sudo -u postgres psql -c "ALTER USER root WITH PASSWORD '123456';"

# 或者使用 postgres 用户连接
sudo -u postgres psql -d askany
```

### 问题 3: pgvector 扩展未找到

**错误**: `ERROR: could not open extension control file`

**解决方案**:
```bash
# 检查扩展文件是否存在
sudo find /usr -name "vector.control" 2>/dev/null

# 如果不存在，需要安装 pgvector（见步骤 3）
# 安装后重启 PostgreSQL
sudo service postgresql restart
```

### 问题 4: 权限不足

**错误**: `ERROR: permission denied to create extension`

**解决方案**:
```bash
# 授予超级用户权限
sudo -u postgres psql -c "ALTER USER root WITH SUPERUSER;"
```

## 快速安装脚本

如果你使用的是 Ubuntu/Debian，可以运行以下脚本：

```bash
#!/bin/bash
set -e

echo "Installing PostgreSQL..."
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-all

echo "Starting PostgreSQL..."
sudo service postgresql start
sudo systemctl enable postgresql

echo "Creating user and database..."
sudo -u postgres createuser -s root || true
sudo -u postgres psql -c "ALTER USER root WITH PASSWORD '123456';" || true
sudo -u postgres createdb askany || true

echo "Installing pgvector..."
cd /tmp
git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
cd ..
rm -rf pgvector

echo "Enabling pgvector extension..."
sudo -u postgres psql -d askany -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "✅ PostgreSQL setup completed!"
echo "Test connection: psql -h localhost -U root -d askany"
```

保存为 `setup_postgresql.sh`，然后运行：

```bash
chmod +x setup_postgresql.sh
./setup_postgresql.sh
```

## 下一步

安装完成后，运行：

```bash
# 验证连接
psql -h localhost -U root -d askany -c "SELECT version();"

# 运行 ingest
python -m askany.main --ingest
```

