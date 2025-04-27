# 启动 server
python -u /home/users/ntu/yaweizha/scratch/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py --port 30000 --root_path /home/users/ntu/yaweizha/scratch/TravelUAV_env > server.log 2>&1 &

sleep 5

# 检查30000端口是否监听
if netstat -tuln | grep 30000 > /dev/null; then
    echo "✅ Server已监听端口 30000，启动成功"
else
    echo "❌ Server未监听端口，启动失败"
    exit 1
fi
