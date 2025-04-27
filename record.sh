日常快速检查自己当前作业情况
myqstat
需要更详细的信息或排错时
qstat -answ
使用 qdel 命令强制结束作业：
qdel 123456.aspbs

netstat -tulnp | grep 30000
如果看到类似输出，说明端口 30000 被用着：

nginx
Copy
Edit
tcp   0   0 0.0.0.0:30000   0.0.0.0:*   LISTEN   12345/python

如果你担心多开的冲突
ps -p 2845768 -o pid,ppid,cmd

# ~/.bashrc 或 ~/.bash_profile 加一行
module load cuda/11.8.0

检查包的版本
pip list | grep msgpack
pip show msgpack
pip show msgpack-rpc-python

lsof -i:9999
或者
netstat -tulnp | grep 9999