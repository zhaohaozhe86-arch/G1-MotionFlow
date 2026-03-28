import torch
import torch.nn as nn
from .base import BaseLosses


class CommitLoss(nn.Module):
    """
    Useless Wrapper
    """
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, commit, commit2, **kwargs):
        return commit


class GPTLosses(BaseLosses):
    
    def __init__(self, cfg, stage, num_joints, **kwargs):
        # Save parameters
        self.stage = stage
        recons_loss = cfg.LOSS.ABLATION.RECONS_LOSS

        # Define losses
        losses = []
        params = {}
        if stage == "vae":
            losses.append("recons_feature")
            params['recons_feature'] = cfg.LOSS.LAMBDA_FEATURE

            losses.append("recons_velocity")
            params['recons_velocity'] = cfg.LOSS.LAMBDA_VELOCITY

            losses.append("vq_commit")
            params['vq_commit'] = cfg.LOSS.LAMBDA_COMMIT
        elif stage in ["lm_pretrain", "lm_instruct"]:
            losses.append("gpt_loss")
            params['gpt_loss'] = cfg.LOSS.LAMBDA_CLS

        # Define loss functions & weights
        losses_func = {}
        for loss in losses:
            if loss.split('_')[0] == 'recons':
                if recons_loss == "l1":
                    losses_func[loss] = nn.L1Loss
                elif recons_loss == "l2":
                    losses_func[loss] = nn.MSELoss
                elif recons_loss == "l1_smooth":
                    losses_func[loss] = nn.SmoothL1Loss
            elif loss.split('_')[1] in [
                    'commit', 'loss', 'gpt', 'm2t2m', 't2m2t'
            ]:
                losses_func[loss] = CommitLoss
            elif loss.split('_')[1] in ['cls', 'lm']:
                losses_func[loss] = nn.CrossEntropyLoss
            else:
                raise NotImplementedError(f"Loss {loss} not implemented.")

        super().__init__(cfg, losses, params, losses_func, num_joints,
                         **kwargs)

    def update(self, rs_set):
        '''Update the losses'''
        total: float = 0.0

        if self.stage in ["vae"]:
            nfeats = rs_set['m_rst'].shape[-1]
            
            # Key-Joints Weighting
            weight_mask = torch.ones_like(rs_set['m_rst'])
            
            if nfeats == 363:
                WAIST_WEIGHT = 3.0  
                FOOT_WEIGHT = 3.0   
                
                # 1. Root, Joint 0
                weight_mask[..., 0:4] = WAIST_WEIGHT       
                weight_mask[..., 91:97] = WAIST_WEIGHT     
                weight_mask[..., 271:274] = WAIST_WEIGHT   
                
                # 2. L_Foot, Joint 6
                weight_mask[..., 19:22] = FOOT_WEIGHT     
                weight_mask[..., 127:133] = FOOT_WEIGHT    
                weight_mask[..., 289:292] = FOOT_WEIGHT    
                
                # 3. R_Foot, Joint 12
                weight_mask[..., 37:40] = FOOT_WEIGHT      
                weight_mask[..., 163:169] = FOOT_WEIGHT    
                weight_mask[..., 307:310] = FOOT_WEIGHT    
                
                # 4. Connect
                weight_mask[..., 361:363] = FOOT_WEIGHT    

            m_rst_weighted = rs_set['m_rst'] * weight_mask
            m_ref_weighted = rs_set['m_ref'] * weight_mask
            
            total += self._update_loss("recons_feature", m_rst_weighted, m_ref_weighted)
            
            if nfeats in [263, 135 + 263, 345, 357, 363]:
                if nfeats == 135 + 263:
                    vel_start = 135 + 4
                else:
                    vel_start = 4 
                    
                vel_end = (self.num_joints - 1) * 3 + vel_start
                
                ric_rst = rs_set['m_rst'][..., vel_start:vel_end] * weight_mask[..., vel_start:vel_end]
                ric_ref = rs_set['m_ref'][..., vel_start:vel_end] * weight_mask[..., vel_start:vel_end]
                
                total += self._update_loss("recons_velocity", ric_rst, ric_ref)
            else:
                if self._params['recons_velocity'] != 0.0:
                    raise NotImplementedError(
                        "Velocity not implemented for nfeats = {}".format(nfeats))
                        
            total += self._update_loss("vq_commit", rs_set['loss_commit'],
                                       rs_set['loss_commit'])

        if self.stage in ["lm_pretrain", "lm_instruct"]:
            total += self._update_loss("gpt_loss", rs_set['outputs'].loss,
                                       rs_set['outputs'].loss)

        # Update the total loss
        self.total += total.detach()
        self.count += 1

        return total