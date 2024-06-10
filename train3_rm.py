from trainers import RewardModelTrainer
from configs import get_configs
from gpt import GPTRewardModel
from dataset import Read_RM_Filtered_Data
import torch._dynamo
torch._dynamo.config.suppress_errors = True


def training_rm(reward_model, exp_name, path_train, path_test, batch_size=1):

    device = 'cuda'
    cfg = get_configs("gpt2-medium/dropout")
    cfg.batch_size = batch_size
    cfg.pretrain = reward_model
    cfg.total_epochs = 1
    cfg.exp_name = exp_name

    path_train = path_train
    path_test = path_test

    rm = GPTRewardModel.from_backbone_checkpoint(cfg, cfg.pretrain)

    train_ds = Read_RM_Filtered_Data(path_train)
    test_ds = Read_RM_Filtered_Data(path_test)
    print("Loaded training and testing datasets")

    trainer = RewardModelTrainer(cfg, device, rm, train_ds, test_ds)
    trainer.fit()

    print(f"Training complete. Model saved as {exp_name}")
