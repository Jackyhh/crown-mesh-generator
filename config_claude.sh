#!/bin/bash
# config_claude.sh
# 设置 Claude Code 所需环境变量

# 把本地可执行路径加入 PATH
# export PATH="$HOME/.local/bin:$PATH"
export PATH="~/.local/bin:$PATH"

# 配置 Anthropic Claude Code 的 API 地址和 Token
export ANTHROPIC_BASE_URL="https://us019adhapepn.imds.ai/api"
export ANTHROPIC_AUTH_TOKEN="cr_98bbacdd6458951b8820aac70c413ee607ed7eb84844a4e755a27b1eddbe796a"

echo "Claude 环境变量已配置完成"