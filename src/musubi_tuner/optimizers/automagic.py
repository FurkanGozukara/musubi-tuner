import random
from typing import List

import torch

from .optimizer_utils import Auto8bitTensor, QBytesTensor, copy_stochastic, stochastic_grad_accummulation


class Automagic(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-6, # lr is start lr
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-6, # amount to bump the lr when adjusting
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        beta2=0.999,
        weight_decay=0.0,
        do_paramiter_swapping=False,
        paramiter_swapping_factor=0.1,
        offload_gradients=False,
        fused=False,
        offload_state=False,
    ):
        self.lr = lr
        if self.lr > 1e-3:
            print(f"Warning! Start lr is very high: {self.lr}. Forcing to 1e-6. this does not work like prodigy")
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "beta2": beta2,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [
            lr for group in self.param_groups
        ]

        self.is_stochastic_rounding_accumulation = False
        self.offload_gradients = offload_gradients
        self.fused = fused
        self.offload_state = offload_state
        self._hook_handles = []

        # Set up either fused updates or stochastic low-precision accumulation.
        for group in self.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    continue
                if self.fused:
                    self._hook_handles.append(
                        param.register_post_accumulate_grad_hook(self._make_backward_hook(group))
                    )
                elif param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    self._hook_handles.append(
                        param.register_post_accumulate_grad_hook(
                            lambda current_param: stochastic_grad_accummulation(
                                current_param, offload_to_cpu=self.offload_gradients
                            )
                        )
                    )

        self.do_paramiter_swapping = do_paramiter_swapping
        self.paramiter_swapping_factor = paramiter_swapping_factor
        self._total_paramiter_size = 0
        # count total paramiters
        for group in self.param_groups:
            for param in group['params']:
                self._total_paramiter_size += torch.numel(param)
        # pretty print total paramiters with comma seperation
        print(f"Total training paramiters: {self._total_paramiter_size:,}")

        # needs to be enabled to count paramiters
        if self.do_paramiter_swapping:
            self.enable_paramiter_swapping(self.paramiter_swapping_factor)

    def enable_paramiter_swapping(self, paramiter_swapping_factor=0.1):
        self.do_paramiter_swapping = True
        self.paramiter_swapping_factor = paramiter_swapping_factor
        # call it an initial time
        self.swap_paramiters()

    def swap_paramiters(self):
        all_params = []
        # deactivate all paramiters
        for group in self.param_groups:
            for param in group['params']:
                param.requires_grad_(False)
                # remove any grad
                param.grad = None
                all_params.append(param)
        # shuffle all paramiters
        random.shuffle(all_params)

        # keep activating paramiters until we are going to go over the target paramiters
        target_paramiters = int(
            self._total_paramiter_size * self.paramiter_swapping_factor)
        total_paramiters = 0
        for param in all_params:
            total_paramiters += torch.numel(param)
            if total_paramiters >= target_paramiters:
                break
            else:
                param.requires_grad_(True)

    @staticmethod
    def _get_lr(param_group, param_state):
        if 'avg_lr' in param_state:
            lr = param_state["avg_lr"]
        else:
            lr = 0.0
        return lr

    def _get_group_lr(self, group):
        group_lrs = []
        for p in group["params"]:
            lr = self._get_lr(group, self.state[p])
            if isinstance(lr, torch.Tensor):
                lr = float(lr.detach())
            group_lrs.append(lr)
        # return avg
        if len(group_lrs) == 0:
            return self.lr
        return sum(group_lrs) / len(group_lrs)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-
                    1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    # automagic manages its own lr
    def get_learning_rates(self):

        lrs = [
            self._get_group_lr(group)
            for group in self.param_groups
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    @staticmethod
    def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
        flat = bits.reshape(-1)
        packed = torch.empty((flat.numel() + 7) // 8, dtype=torch.uint8, device=bits.device)
        weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=bits.device)
        chunk_bits = 4 * 1024 * 1024
        for start in range(0, flat.numel(), chunk_bits):
            end = min(start + chunk_bits, flat.numel())
            chunk = flat[start:end]
            pad = (-chunk.numel()) % 8
            if pad:
                chunk = torch.cat((chunk, chunk.new_zeros(pad)))
            packed[start // 8 : (end + 7) // 8].copy_(
                (chunk.view(-1, 8).to(torch.uint8) * weights).sum(dim=-1, dtype=torch.uint8)
            )
        return packed

    @staticmethod
    def _unpack_bits(packed: torch.Tensor, numel: int) -> torch.Tensor:
        result = torch.empty(numel, dtype=torch.bool, device=packed.device)
        shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
        chunk_bytes = 512 * 1024
        for start in range(0, packed.numel(), chunk_bytes):
            chunk = packed[start : start + chunk_bytes]
            unpacked = ((chunk.unsqueeze(-1) >> shifts) & 1).reshape(-1).to(torch.bool)
            output_start = start * 8
            output_end = min(output_start + unpacked.numel(), numel)
            result[output_start:output_end].copy_(unpacked[: output_end - output_start])
        return result

    def _restore_offloaded_state(self, p):
        state = self.state[p]
        packed = state.pop("last_polarity_packed", None)
        if packed is not None:
            # torch.optim casts state tensors to the parameter dtype while loading.
            # Packed polarity bytes must remain integers before bit operations.
            packed = packed.to(device=p.device, dtype=torch.uint8)
            state["last_polarity"] = self._unpack_bits(packed, p.numel()).reshape(p.shape)
            state.pop("last_polarity_numel", None)
        lr_mask = state.get("lr_mask")
        if isinstance(lr_mask, Auto8bitTensor) and lr_mask.quantized.device != p.device:
            lr_mask.quantized = lr_mask.quantized.to(p.device)

    def _offload_optimizer_state(self, p):
        state = self.state[p]
        last_polarity = state.pop("last_polarity", None)
        if last_polarity is not None:
            state["last_polarity_packed"] = self._pack_bits(last_polarity).cpu()
            state["last_polarity_numel"] = p.numel()
        lr_mask = state.get("lr_mask")
        if isinstance(lr_mask, Auto8bitTensor):
            lr_mask.quantized = lr_mask.quantized.cpu()
        for key in ("exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq", "avg_lr", "RMS"):
            value = state.get(key)
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()

    def _make_backward_hook(self, group):
        def hook(p):
            if p.grad is None:
                return
            with torch.no_grad():
                self._restore_offloaded_state(p)
                self._update_param(p, group)
                if self.offload_state:
                    self._offload_optimizer_state(p)

        return hook

    def _update_param(self, p, group):
        if p.grad is None or not p.requires_grad:
            return

        grad = p.grad
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)
        if grad.is_sparse:
            raise RuntimeError("Automagic does not support sparse gradients.")

        state = self.state[p]
        factored = len(grad.shape) >= 2
        if len(state) == 0:
            self.initialize_state(p)
        elif factored:
            if "exp_avg_sq_row" not in state or "exp_avg_sq_col" not in state:
                state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1]).to(grad)
                state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:]).to(grad)
            else:
                state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
        elif "exp_avg_sq" not in state:
            state["exp_avg_sq"] = torch.zeros_like(grad)
        else:
            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

        p_data_fp32 = p
        if isinstance(p_data_fp32, QBytesTensor):
            p_data_fp32 = p_data_fp32.dequantize()
        if p.dtype != torch.float32:
            p_data_fp32 = p_data_fp32.clone().float()

        state["step"] = state.get("step", 0) + 1
        state["RMS"] = self._rms(p_data_fp32)

        beta2 = group["beta2"]
        eps = group["eps"]
        if isinstance(eps, (tuple, list)):
            eps = eps[0]
        update = grad.square().add_(eps)
        if factored:
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]
            exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
            exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)
            del update
            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad)
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
            update = exp_avg_sq.rsqrt().mul_(grad)

        p.grad = None
        del grad
        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        if "last_polarity" not in state or "lr_mask" not in state:
            self.initialize_state(p)
        last_polarity = state.pop("last_polarity")
        current_polarity = update > 0
        sign_agreement = last_polarity.eq_(current_polarity)
        state["last_polarity"] = current_polarity

        old_lr_mask = state.pop("lr_mask")
        lr_mask = old_lr_mask.to(torch.float32)
        del old_lr_mask
        lr_mask.add_(sign_agreement, alpha=2.0 * self.lr_bump).sub_(self.lr_bump)
        lr_mask.clamp_(min=self.min_lr, max=self.max_lr)
        del sign_agreement

        update.mul_(lr_mask)
        state["lr_mask"] = Auto8bitTensor(lr_mask)
        state["avg_lr"] = torch.mean(lr_mask)
        if group["weight_decay"] != 0:
            p_data_fp32.addcmul_(p_data_fp32, lr_mask, value=-group["weight_decay"])
        del lr_mask

        p_data_fp32.add_(-update)
        if p.dtype != torch.float32:
            copy_stochastic(p, p_data_fp32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.fused:
            self.step_hook()
            return loss

        self.step_hook()
        for group in self.param_groups:
            for p in group["params"]:
                self._restore_offloaded_state(p)
                self._update_param(p, group)
                if self.offload_state:
                    self._offload_optimizer_state(p)
        return loss
    
    def initialize_state(self, p):
        state = self.state[p]
        state["step"] = 0

        # store the lr mask
        if 'lr_mask' not in state:
            initial_lr = torch.full(p.shape, self.lr, device=p.device, dtype=torch.float32)
            state['lr_mask'] = Auto8bitTensor(initial_lr)
            del initial_lr
        state['avg_lr'] = torch.tensor(self.lr, device=p.device, dtype=torch.float32)
        if 'last_polarity' not in state:
            state['last_polarity'] = torch.zeros(
                p.shape, dtype=torch.bool, device=p.device)
        
        factored = len(p.shape) >= 2
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(
                p.shape[:-1]).to(p)
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:]).to(p)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["RMS"] = 0
    
    # override the state_dict to save the lr_mask
    def state_dict(self, *args, **kwargs):
        orig_state_dict = super().state_dict(*args, **kwargs)
        # convert the state to quantized tensor to scale and quantized
        new_sace_state = {}
        for p, state in orig_state_dict['state'].items():
            save_state = {k: v for k, v in state.items() if k != 'lr_mask'}
            
            # Check if lr_mask exists in the state before trying to access it
            if 'lr_mask' in state:
                save_state['lr_mask'] = state['lr_mask'].state_dict()
            
            new_sace_state[p] = save_state
            
        orig_state_dict['state'] = new_sace_state
        
        return orig_state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        # Validate that the state_dict is from an Automagic optimizer
        is_valid_automagic_state = False
        
        # Check if state_dict has the expected structure
        if 'state' in state_dict and isinstance(state_dict['state'], dict):
            # Check if at least one state entry has an lr_mask, which is specific to Automagic
            for param_id, param_state in state_dict['state'].items():
                if isinstance(param_state, dict) and 'lr_mask' in param_state:
                    is_valid_automagic_state = True
                    break
        
        if not is_valid_automagic_state:
            return
        
        # First, call the parent class's load_state_dict to load the basic optimizer state
        # We'll handle the lr_mask separately
        state_dict_copy = {
            'state': {},
            'param_groups': state_dict['param_groups']
        }
        
        # Copy all state entries except lr_mask
        for param_id, param_state in state_dict['state'].items():
            state_dict_copy['state'][param_id] = {
                k: v for k, v in param_state.items() if k != 'lr_mask'
            }
        
        # Call parent class load_state_dict with the modified state dict
        super().load_state_dict(state_dict_copy)
        
        # Now handle the lr_mask separately
        # We need to map the saved parameters to the current parameters
        # This is tricky because the parameter IDs might be different
        
        # Get all current parameters that require gradients
        current_params = []
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    current_params.append(p)
        
        # If the number of parameters doesn't match, we can't reliably map them
        if len(current_params) != len(state_dict['param_groups'][0]['params']):
            print(f"WARNING: Number of parameters doesn't match between saved state ({len(state_dict['param_groups'][0]['params'])}) "
                  f"and current model ({len(current_params)}). Learning rate masks may not be correctly loaded.")
        
        # Map parameters by their position in the param_groups
        # This assumes the order of parameters is preserved between saving and loading
        saved_param_ids = list(state_dict['state'].keys())
        
        for i, current_param in enumerate(current_params):
            if i >= len(saved_param_ids):
                break
                
            saved_param_id = saved_param_ids[i]
            saved_state = state_dict['state'][saved_param_id]
            
            # Skip if this saved state doesn't have an lr_mask
            if 'lr_mask' not in saved_state:
                continue
                
            # Initialize the state for this parameter if it doesn't exist
            if current_param not in self.state:
                self.initialize_state(current_param)
                
            # Get the current state for this parameter
            current_state = self.state[current_param]
            
            # Load the lr_mask from the saved state
            saved_lr_mask = saved_state['lr_mask']
            
            # Reconstruct the Auto8bitTensor from its state dict
            try:
                # Make sure the shapes match
                if 'quantized' in saved_lr_mask and saved_lr_mask['quantized'].shape == current_param.shape:
                    current_state['lr_mask'] = Auto8bitTensor(saved_lr_mask)
                else:
                    print(f"WARNING: Shape mismatch for parameter {i}. "
                          f"Expected {current_param.shape}, got {saved_lr_mask['quantized'].shape if 'quantized' in saved_lr_mask else 'unknown'}. "
                          f"Initializing new lr_mask.")
                    # Initialize a new lr_mask
                    current_state['lr_mask'] = Auto8bitTensor(torch.ones(
                        current_param.shape).to(current_param.device, dtype=torch.float32) * self.lr
                    )
            except Exception as e:
                print(f"ERROR: Failed to load lr_mask for parameter {i}: {e}")
                # Initialize a new lr_mask
                current_state['lr_mask'] = Auto8bitTensor(torch.ones(
                    current_param.shape).to(current_param.device, dtype=torch.float32) * self.lr
                )
