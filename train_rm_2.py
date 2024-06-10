from trainers import RewardModelTrainer
from configs import get_configs
from gpt import GPTRewardModel
from dataset import Read_RM_Filtered_Data
import torch._dynamo
torch._dynamo.config.suppress_errors = True

device = 'cuda'
cfg = get_configs("gpt2-medium/dropout")
cfg.batch_size = 1
cfg.pretrain = './sft_sft_0_202404201349/sft_sft_0_202404201349_step20000.pt'
cfg.total_epochs = 1
cfg.exp_name = 'rm_frft_may8'

path_train = 'path/to_train_subset_for_that_epoch'
path_test = 'path/to_test_subset_for_that_epoch'

if cfg.pretrain == "huggingface":
    rm = GPTRewardModel.from_pretrained(cfg)
else:
    rm = GPTRewardModel.from_backbone_checkpoint(cfg, cfg.pretrain)

train_ds = Read_RM_Filtered_Data(path_train)
test_ds = Read_RM_Filtered_Data(path_test)

trainer = RewardModelTrainer(cfg, device, rm, train_ds, test_ds)
trainer.fit()
