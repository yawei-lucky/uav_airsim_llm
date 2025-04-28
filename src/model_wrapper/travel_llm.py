import numpy as np
import torch
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor

class DummyProcessor:
    def preprocess(self, images, return_tensors='pt'):
        batch_size = 1
        channels = 3
        height = 224
        width = 224

        # 用随机噪声代替纯0，更接近“正常”输入分布
        dummy_image = torch.rand(batch_size, channels, height, width)

        return {'pixel_values': dummy_image}

class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.model_max_length = 512

    def __call__(self, text, return_tensors=None):
        tokens = text.split()
        ids = [self.bos_token_id] + list(range(2, 2 + len(tokens)))
        if return_tensors == 'pt':
            return type('DummyTokenized', (object,), {'input_ids': torch.tensor(ids).unsqueeze(0)})
        else:
            return type('DummyTokenized', (object,), {'input_ids': ids})  # ✅ 返回 list，不要 Tensor

    def add_tokens(self, special_tokens_list, special_tokens=True):
        return len(special_tokens_list)

    def resize_token_embeddings(self, size):
        pass

class TravelModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        # self.tokenizer, self.model, self.image_processor = load_model(model_args)
        # self.traj_model = load_traj_model(model_args)
        # self.model.to(torch.bfloat16)
        # self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        # self.dino_moinitor = None
        # self.model_args = model_args
        # self.data_args = data_args
        
        self.debug = getattr(model_args, 'debug', True)  # ✅新增：从model_args取debug标志

        if not self.debug:
            self.tokenizer, self.model, self.image_processor = load_model(model_args)
            self.traj_model = load_traj_model(model_args)
            self.model.to(torch.bfloat16)
            self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        else:
            self.tokenizer = DummyTokenizer()
            self.model = None
            self.image_processor = DummyProcessor()
            self.traj_model = None

        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args

    # def prepare_inputs(self, episodes, target_positions, assist_notices=None):
    #     inputs = []
    #     rot_to_targets = []
        
    #     for i in range(len(episodes)):
    #         input_item, rot_to_target = prepare_data_to_inputs(
    #             episodes=episodes[i],
    #             tokenizer=self.tokenizer,
    #             image_processor=self.image_processor,
    #             data_args=self.data_args,
    #             target_point=target_positions[i],
    #             assist_notice=assist_notices[i] if assist_notices is not None else None
    #         )
    #         inputs.append(input_item)
    #         rot_to_targets.append(rot_to_target)
    #     batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)

    #     inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
    #         if 'prompts' not in k and 'images' not in k and 'historys' not in k}
    #     inputs_device['prompts'] = [item for item in batch['prompts']]
    #     inputs_device['images'] = [item.to(self.model.device) for item in batch['images']]
    #     inputs_device['historys'] = [item.to(device=self.model.device, dtype=self.model.dtype) for item in batch['historys']]
    #     inputs_device['orientations'] = inputs_device['orientations'].to(dtype=self.model.dtype)
    #     inputs_device['return_waypoints'] = True
    #     inputs_device['use_cache'] = False
        
    #     return inputs_device, rot_to_targets
    
    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        if self.debug:
            # ✅ Debug模式，直接返回假的inputs
            inputs_device = {'return_waypoints': True, 'use_cache': False}
            rot_to_targets = []
            for i in range(len(episodes)):
                pos = np.array(episodes[i][-1]['sensors']['state']['position'])
                target_pos = np.array(target_positions[i])
                dir_vec = (target_pos - pos)[:3]
                dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
                rot_to_targets.append(dir_vec)
            return inputs_device, rot_to_targets
        else:
            # ✅ 正常流程
            inputs = []
            rot_to_targets = []
            for i in range(len(episodes)):
                input_item, rot_to_target = prepare_data_to_inputs(
                    episodes=episodes[i],
                    tokenizer=self.tokenizer,
                    image_processor=self.image_processor,
                    data_args=self.data_args,
                    target_point=target_positions[i],
                    assist_notice=assist_notices[i] if assist_notices is not None else None
                )
                inputs.append(input_item)
                rot_to_targets.append(rot_to_target)
            batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)

            inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
                if 'prompts' not in k and 'images' not in k and 'historys' not in k}
            inputs_device['prompts'] = [item for item in batch['prompts']]
            inputs_device['images'] = [item.to(self.model.device) for item in batch['images']]
            inputs_device['historys'] = [item.to(device=self.model.device, dtype=self.model.dtype) for item in batch['historys']]
            inputs_device['orientations'] = inputs_device['orientations'].to(dtype=self.model.dtype)
            inputs_device['return_waypoints'] = True
            inputs_device['use_cache'] = False
            
            return inputs_device, rot_to_targets


    def run_llm_model(self, inputs):
        waypoints_llm = self.model(**inputs).cpu().to(dtype=torch.float32).numpy()
        waypoints_llm_new = []
        for waypoint in waypoints_llm:
            waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            waypoints_llm_new.append(waypoint_new)
        return np.array(waypoints_llm_new)

    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
        inputs = prepare_data_to_traj_model(episodes, waypoints_llm_new, self.image_processor, rot_to_targets)
        waypoints_traj = self.traj_model(inputs, None)
        refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(refined_waypoints, episodes)
        return refined_waypoints
    
    def eval(self):
        # self.model.eval()
        # self.traj_model.eval()
        if not self.debug:
            self.model.eval()
            self.traj_model.eval()

        
    def run(self, inputs, episodes, rot_to_targets):
        # waypoints_llm_new = self.run_llm_model(inputs)
        # refined_waypoints = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)
        
        # return refined_waypoints
        if self.debug:
            refined_waypoints = []
            for i, (ep, target_pos) in enumerate(zip(episodes, rot_to_targets)):
                pos = np.array(ep[-1]['sensors']['state']['position'])
                dir_vec = np.array(target_pos) - pos
                dir_vec = dir_vec[:3]
                dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)  # 归一化方向
                waypoints = [dir_vec.tolist() for _ in range(5)]  # ✅ 生成5个一样的点，符合 move_path 要求
                refined_waypoints.append(waypoints)
            return refined_waypoints
        else:
            waypoints_llm_new = self.run_llm_model(inputs)
            refined_waypoints = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)
            return refined_waypoints
    
    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.debug:
            return [False for _ in range(len(episodes))]
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones
        

    