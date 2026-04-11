#!/bin/bash
# 旧版「本机 policy socket 服务 + eval 客户端」已移除（易与环境/后训练边界混淆）。
# 评测与后训练请使用 TCP 服务层 + RLinf，见仓库根目录 run.md。
set -euo pipefail
echo -e "\033[33m[deprecated]\033[0m policy_model_server.py / eval_policy_client.py 已删除。"
echo "替代流程:"
echo "  1) RoboTwin 环境: python script/robotwin_env_server.py --port 8765 --config ... --assets-path ..."
echo "  2) DP 推理:       python script/robotwin_dp_server.py --port 8767 --ckpt /path/to.ckpt"
echo "  3) RLinf:         train_embodied_agent.py + config robotwin_place_empty_cup_dsrl_dp 等"
echo "详见 run.md（Docker 路径示例: /workspace/...）。"
exit 1
