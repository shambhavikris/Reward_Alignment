from trainers import RewardModelTrainer
from configs import get_configs
from gpt import GPTRewardModel
from dataset import DahoasRMStaticDataset_Filtered
import torch._dynamo
torch._dynamo.config.suppress_errors = True



device = 'cuda'
cfg = get_configs("gpt2-medium/dropout")
cfg.batch_size = 1
cfg.pretrain = './sft_sft_0_202404201349/sft_sft_0_202404201349_step20000.pt'
cfg.total_epochs = 1
cfg.exp_name = 'rm_frft_trial_to_be_deleted'
generationmodel = './sft_sft_0_202404201349/sft_sft_0_202404201349_step20000.pt'

if cfg.pretrain == "huggingface":
    rm = GPTRewardModel.from_pretrained(cfg)
else:
    rm = GPTRewardModel.from_backbone_checkpoint(cfg, cfg.pretrain)

train_ds = DahoasRMStaticDataset_Filtered(block_size=1024,
                                    split='train',
                                    current_model_path=generationmodel,
                                    max_examples=2,
                                    tokenizer_name="tiktoken/gpt2")
test_ds = DahoasRMStaticDataset_Filtered(block_size=1024,
                                split='test',
                                current_model_path=generationmodel,
                                max_examples=2,
                                tokenizer_name="tiktoken/gpt2")
trainer = RewardModelTrainer(cfg, device, rm, train_ds, test_ds)
trainer.fit()



