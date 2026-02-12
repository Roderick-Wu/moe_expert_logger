import os
import json
import torch

class MoeLogger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MoeLogger, cls).__new__(cls)
            cls._instance.log_path = os.environ.get("VLLM_LOG_MOE")

            logging_layer = os.environ.get("VLLM_LOG_LAYER")
            if logging_layer is not None:
                cls._instance.log_layer = [int(logging_layer)] # We keep at list, can probably log multiple layers somehow later

            num_layers = os.environ.get("VLLM_NUM_MOE_LAYERS")
            if num_layers is not None:
                cls._instance.num_moe_layers = int(num_layers)

            cls._instance.call_counter = 0
            cls._instance.initialized = False
            
        return cls._instance

    def log(self, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        if not self.log_path or not self.log_layer or not self.num_moe_layers:
            return

        # Determine which MoE layer this call corresponds to
        current_layer = self.call_counter % self.num_moe_layers
        self.call_counter += 1

        # Only log when we hit the target layer
        if current_layer not in self.log_layer:
            return

        if not self.initialized:
            self._write_header()

        # Move to CPU once per batch to avoid multiple synchronization points
        ids_cpu = topk_ids.detach().cpu().tolist()
        weights_cpu = topk_weights.detach().cpu().tolist()

        with open(self.log_path, "a") as f:
            for i, (ids, weights) in enumerate(zip(ids_cpu, weights_cpu)):
                record = {
                    "type": "route",
                    "req_id": "offline", 
                    "token_idx": i,
                    "layer": self.log_layer[0],
                    "topk_ids": ids,
                    "topk_weights": [round(w, 4) for w in weights]
                }
                f.write(json.dumps(record) + "\n")

    def _write_header(self):
        import vllm
        header = {
            "type": "meta",
            "model_id": "Qwen1.5-MoE-A2.7B-Chat",
            "vllm_version": vllm.__version__,
            "torch_version": torch.__version__,
            "device": str(torch.cuda.get_device_name(0)),
            "layers_logged": self.log_layer,
        }
        with open(self.log_path, "w") as f:
            f.write(json.dumps(header) + "\n")
        self.initialized = True

moe_logger = MoeLogger()