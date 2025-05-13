from src.neural_net.LightWeightModel import LightWeightModel as Model
from src.dataset import SpeechModule
import torch
import torch.nn as nn
import torch.optim as optim
import config
import winsound
import random
import torch.nn.functional as F
from tools.utils import plot_spectrogram, ctc_decoder
from src.preprocess import LogMelSpectrogram
import torchaudio
from tools import language_corpus as lc
import os
import json 
from tqdm import tqdm

class SpeechTrainer:
    def __init__(self, model, loaders, criterion, optimizer, scheduler, device):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.check_sample = False
        self.log_file = config.LOG_DIR / 'train_log.json'
        self.step_losses = {
            'train': [],
            'val': []
        }
        self.epoch_losses = {
            'train': [],
            'val': []
        }

        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write(model)

        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)

    def print_memory_stats():
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            print(torch.cuda.memory_summary())

    def start(self, num_epochs=10, resume=False):
        print(f"Training on {self.device}")
        start_epoch = 0

        if resume:
            print("Resuming from checkpoint...")
            start_epoch = self.load_checkpoint(config.CHECKPOINT_DIR / 'checkpoint.pth')

        for epoch in range(start_epoch, num_epochs):
            epoch += 1
            buckets = list(self.loaders.keys())
            random.shuffle(buckets)
            train_loaders, val_loaders = [], []
            random_bucket = random.choice(buckets)
            print(f"Random Bucket: {random_bucket}")
            for bucket in buckets:
                train_loaders.append(self.loaders[bucket]['train'])
                val_loaders.append(self.loaders[bucket]['val'])

                if self.check_sample:
                    print(self.loaders[random_bucket])
                    self.sanity_check(self.loaders[random_bucket]['train'])
                    self.sanity_check(self.loaders[random_bucket]['val'])
                    self.check_sample = False

            train_loss = self.train(train_loaders, epoch)
            val_loss = self.validate(val_loaders, epoch)

            if val_loss <= 0.5:
                self.save_checkpoint(epoch, id=f"target_reached_{val_loss:.2f}")

            self.epoch_losses['train'].append(train_loss)
            self.epoch_losses['val'].append(val_loss)

            self.scheduler.step(val_loss)
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            tqdm.write(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.save_checkpoint(epoch, id=f"val_loss_{val_loss:.4f}")

    def step(self, mode='train', batch=None, step_count=0):
        inputs, labels, inputs_len, labels_len, file_name = batch
        inputs, labels = inputs.to(self.device).transpose(1, 2).contiguous(), labels.to(self.device)
        inputs_len, labels_len = inputs_len.to(self.device), labels_len.to(self.device)

        bs = inputs.shape[0]
        hidden = self.model._init_hidden(batch_size=bs, device=self.device)

        output, _ = self.model(inputs, hidden)
        _log_softmax = F.log_softmax(output, dim=2)

        loss = self.criterion(_log_softmax, labels, inputs_len // 2, labels_len)

        if mode == 'val' or (step_count % 100 == 0 and step_count > 0):
            sample = output.transpose(0, 1).contiguous()
            prediction = torch.argmax(sample[0], dim=1)
            
            with open(self.log_file, 'a') as f:
                f.write(f"Step {step_count} | Loss: {loss.item():.4f}\nPrediction: {prediction.tolist()} | Labels: {labels[0].tolist()}\n")
            tqdm.write(f"prediction: {ctc_decoder(prediction.tolist())} \nLabels: {labels[0].tolist()} ")

        return loss

    def train(self, loaders, epoch):
        self.model.train()
        total_step = sum([len(loader) for loader in loaders])
        total_loss = 1e-9
        current_step = 0

        progress_bar = tqdm(total=total_step, desc=f"Epoch {epoch}/{config.H_PARAMS['TOTAL_EPOCH']}", dynamic_ncols=True, leave=True)

        for loader_idx, loader in enumerate(loaders):
            for batch_idx, batch in enumerate(loader):
                current_step += 1

                self.optimizer.zero_grad()
                loss = self.step(mode='train', batch=batch, step_count=current_step)            
                loss.backward()
                loss = loss.item()
                total_loss += loss

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                progress_bar.set_postfix(loss=loss, avg_loss=total_loss / current_step)
                progress_bar.update(1)

                if current_step % 100 == 0:
                    self.print_grad_stats(self.model)
                self.optimizer.step()
                self.step_losses['train'].append(f"{loss:.4f}")

                if current_step >= total_step:
                    self.save_checkpoint(epoch, id=f"train_last_step_{current_step}_{loss:.4f}")

        progress_bar.close()
        return total_loss / total_step

    def validate(self, loaders, epoch):
        self.model.eval()
        total_loss = 1e-9
        total_step = sum([len(loader) for loader in loaders])
        current_step = 0

        progress_bar = tqdm(total=total_step, desc=f"Epoch {epoch} [Validation]", dynamic_ncols=True)

        for loader in loaders:
            with torch.no_grad():
                for idx, batch in enumerate(loader):
                    current_step += 1
                    loss = self.step(mode='val', batch=batch)
                    total_loss += loss.item()
                    self.step_losses['val'].append(f"{loss.item():.4f}")
                    progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / current_step)
                    progress_bar.update(1)

                    if loss.item() <= 0.5 and not os.path.exists(config.CHECKPOINT_DIR / f"val_target_reached.pth"):
                        self.save_checkpoint(epoch, id=f"val_target_reached")

        progress_bar.close()
        return total_loss / total_step
    
    def print_grad_stats(self, model):
        with open(self.log_file, 'a') as f:
            f.write(f"Learning Rate: {self.scheduler.get_last_lr()}\n")
            f.write(f"Gradients:\n")
           
            print(f"Learning Rate: {self.scheduler.get_last_lr()}")
            for name, param in model.named_parameters():
                if param.requires_grad is not None:
                    f.write(f"{name}: {param.grad.norm():.4f}\n")
                    print(f"{name}: {param.grad.norm():.4f}")

    def sanity_check(self, loaders):
        for batch in loaders:
            inputs, labels, input_len, labels_len, file_name = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            input_len, labels_len = input_len.to(self.device), labels_len.to(self.device)
            random_idx = random.randint(0, inputs.shape[0] - 1)

            if inputs is None or labels is None:
                raise ValueError("Inputs or labels are None.")
            if len(inputs) == 0 or len(labels) == 0:
                raise ValueError("Inputs or labels are empty.")
            if inputs.shape[0] != labels.shape[0]:
                raise ValueError("Batch size mismatch between inputs and labels.")
            if input_len.shape[0] != labels_len.shape[0]:
                raise ValueError("Batch size mismatch between input lengths and label lengths.")

            print(f"Inputs shape: {inputs.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Input lengths shape: {input_len.shape}")
            print(f"Label lengths shape: {labels_len.shape}")
            print(f"Input lengths: {input_len}")
            print(f"Label lengths: {labels_len}")

            print(f"Sample: {random_idx}")
            print(f"File name: {file_name[random_idx]}")
            print(f"Input Shape: {inputs[random_idx].shape}")
            print(f"Label Shape: {labels[random_idx].shape}")
            print(f"Input length: {input_len[random_idx]}")
            print(f"Label length: {labels_len[random_idx]}")
            print(f"Input: {inputs[random_idx]}")
            print(f"Label: {labels[random_idx]}")
            print(f"Decoded Label: {lc.decode(labels[random_idx].tolist())}")
            audio_path = config.WAVS_PATH / f'{file_name[random_idx]}.wav'
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file {audio_path} does not exist.")
            audio, sr = torchaudio.load(audio_path)       

            spec = LogMelSpectrogram()(audio)
            winsound.PlaySound(audio_path, winsound.SND_FILENAME)
            print(f"Spec Stats: {spec.shape} | Min: {spec.min()} | Max: {spec.max()} | Mean: {spec.mean()} | Std: {spec.std()}")
            print(f"Loaded Specs Stats: {inputs[random_idx].shape} | Min: {inputs[random_idx].min()} | Max: {inputs[random_idx].max()} | Mean: {inputs[random_idx].mean()} | Std: {inputs[random_idx].std()}")

            plot_spectrogram( inputs[random_idx].transpose(0, 1), spec, sample_rate=sr)
            return

    def save_checkpoint(self,epoch, id = random.randint(0, 10000)):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch_losses': self.epoch_losses,
            'step_losses': self.step_losses
        }
        torch.save(checkpoint, config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}_{id}.pth")
        print(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch_losses = checkpoint.get('epoch_losses', {'train': [], 'val': []})
        self.step_losses = checkpoint.get('step_losses', {'train': [], 'val': []})
        
        start_epoch = checkpoint['epoch']  # Resume from next epoch
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
    
def main(resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    speech_module = SpeechModule()
    loaders = speech_module.loaders

    total_steps = sum([len(loader) for loader in loaders.values()])

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=config.H_PARAMS["TOTAL_EPOCH"] * len(config.BUCKETS), cycle_momentum=False)

    
    
    trainer = SpeechTrainer(model=model, loaders=loaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device)

    trainer.start(num_epochs=config.H_PARAMS["TOTAL_EPOCH"], resume=False)
    
if __name__ == "__main__":
    main()