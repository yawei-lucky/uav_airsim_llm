import msgpackrpc
port = 30000
client = msgpackrpc.Client(msgpackrpc.Address("127.0.0.1", port), timeout=3)
print(client.call("ping"))  # 成功返回 'pong' 才说明服务正常

from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool

machines_info = [{
    'MACHINE_IP': '127.0.0.1',
    'SOCKET_PORT': port,
    'MAX_SCENE_NUM': 16,
    'open_scenes': ['NewYorkCity'],
    'gpus': [0]
}]

sim = AirVLNSimulatorClientTool(machines_info)
sim.run_call()
