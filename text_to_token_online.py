import torch
import numpy as np
import os
import time
from omegaconf import OmegaConf, open_dict
from mGPT.models.build_model import build_model

CONFIG_PATH = "configs/config_ug1_stage2.yaml" 
LM_CONFIG_PATH = "configs/lm/default.yaml" 
VQ_CONFIG_PATH = "configs/vq/default.yaml"
CKPT_PATH = "experiments/mgpt/Pretrain_UnitreeG1_363/checkpoints/epoch=119.ckpt"
OUTPUT_DIR = "results/generated_tokens_npy"

def apply_patches(cfg, lm_cfg, vq_cfg):
    from omegaconf import OmegaConf, open_dict
    import torch.serialization
    
    _original_load = torch.load
    def _unsafe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _unsafe_load
    torch.serialization.load = _unsafe_load

    with open_dict(cfg):
        if 'model' not in cfg: 
            cfg.model = OmegaConf.create()
        if 'params' not in cfg.model: 
            cfg.model.params = OmegaConf.create()
            
        cfg.model.params.codebook_size = 512 
        
        # LM
        cfg.model.params.lm = lm_cfg  
        cfg.lm = OmegaConf.create({"default": lm_cfg})
        
        # VQ
        cfg.model.params.vq = vq_cfg
        cfg.vq = OmegaConf.create({"default": vq_cfg})
        
        # 1. DATASET
        if 'DATASET' not in cfg:
            cfg.DATASET = OmegaConf.create()
        cfg.DATASET.NFEATS = 363
        cfg.DATASET.JOINT_TYPE = 'unitreeg1' 
        
        # 2. METRIC 
        if 'METRIC' not in cfg: 
            cfg.METRIC = OmegaConf.create()
        cfg.METRIC.DIST_SYNC_ON_STEP = True  
        
        # 3. ABLATION
        if 'ABLATION' not in cfg:
            cfg.ABLATION = OmegaConf.create()
        cfg.ABLATION.VAE_TYPE = 'actor' 
        cfg.ABLATION.VAE_ARCH = 'encoder_decoder' 
        cfg.ABLATION.PE_TYPE = 'actor' 
        cfg.ABLATION.DIFF_PE_TYPE = 'actor'

    OmegaConf.resolve(cfg)
    return cfg

def load_inference_model():
    print(f"🔄 Main config: {CONFIG_PATH}")
    cfg = OmegaConf.load(CONFIG_PATH)
    
    print(f"🔄 LM config: {LM_CONFIG_PATH}")
    lm_cfg = OmegaConf.load(LM_CONFIG_PATH)

    print(f"🔄 VQ config: {VQ_CONFIG_PATH}")
    vq_cfg = OmegaConf.load(VQ_CONFIG_PATH)
    
    cfg = apply_patches(cfg, lm_cfg, vq_cfg)
    cfg.TEST.BATCH_SIZE = 1 
    
    class DummyDatamodule:
        def __init__(self):
            self.njoints = 363   
            self.name = "UnitreeG1_363"
            self.hparams = type('obj', (object,), {'njoints': 363})
            self.feats2joints = lambda x: x 
            self.renorm4t2m = lambda x: x   
            self.mm_mode = lambda x: None 
            self.t2m_dataset = [] 
            
    print(f"🔄 Configuring LLM architecture (Feature Dim: 363)...")
    model = build_model(cfg, DummyDatamodule())
    model.eval()
    model.cuda()

    print(f"📥 Loading checkpoints: {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, map_location="cuda")["state_dict"]
    new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("metrics.")}
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Model Ready！")
    return model

def clean_tokens(token_tensor):
    """cleaning T5 special tokens"""
    token_np = token_tensor.detach().cpu().numpy()
    clean_tokens = [t for t in token_np if t >= 3]

    return np.array(clean_tokens, dtype=np.int32)

def main():
    try:
        model = load_inference_model()
    except Exception as e:
        print(f"\n❌ Failed Loading: {e}")
        import traceback
        traceback.print_exc()
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*50)
    print("🚀 LLM-to-Tokens")
    print("="*50 + "\n")

    while True:
        text_prompt = input(">>> Inputs (eg. 'A robot walks forward', q): ").strip()
        
        if text_prompt.lower() in ['q', 'exit']:
            break
        if not text_prompt:
            continue
            
        try:
            with torch.no_grad():
                print(f"   ⏳ LLM thinking...")
                
                output_tokens, _ = model.lm.generate_direct(
                    [text_prompt], 
                    do_sample=False,     
                    max_length=196       
                )
                
                raw_tokens = output_tokens[0]
                final_tokens = clean_tokens(raw_tokens)
                
                safe_text = text_prompt.replace(" ", "_")[:20]
                filename = f"{OUTPUT_DIR}/{safe_text}.npy"
                
                np.save(filename, final_tokens)
                
                print(f"✅ Finished! Saved: {filename}")
                print(f"   📊 Length of motion: {len(final_tokens)} frames")
                print(f"   🔢 Token preview: {final_tokens[:15]} ...\n")

        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()