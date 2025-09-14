from unittest import loader
from src.neural_net.LightWeightModel import LightWeightModel as Model
from src.dataset import SpeechModule
import torch
import torch.nn as nn
import torch.optim as optim
import config
import random
import torch.nn.functional as F
from tools.utils import plot_spectrogram, ctc_decoder, play_sound, audio_sanity_check
import torchaudio
from tools import language_corpus as lc
from src.preprocess import WhisperLogMelSpectrogram
import os
import json 
from tqdm import tqdm
import logging

# Set up logging for training shape tracking
logger = logging.getLogger(__name__)

class SpeechTrainer:
    def __init__(self, model, loaders, criterion, optimizer, scheduler, device, total_steps, speech_module=None):
        self.model = model
        self.loaders = loaders
        self.speech_module = speech_module  # Add speech_module for epoch progress updates
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.check_sample = True
        self.log_file = config.LOG_DIR / 'train_log.json'
        self.overall_step_count = 0
        self.total_steps = total_steps
        self.step_losses = {
            'train': [],
            'val': []
        }
        self.epoch_losses = {
            'train': [],
            'val': []
        }
        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file))


        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)

        print(f"Training on {self.device}")
        print(f"Model: {self.model}")
        print(f"Total Parameters: {sum(p.numel() for p in self.model.parameters())}")

    def start(self, num_epochs=15, resume=False, sort=False, checkpoint_name=None):
        start_epoch = 0
        if resume:
            if checkpoint_name is None:
                raise ValueError("Checkpoint name must be provided for resuming training.")
            start_epoch = self.load_checkpoint(config.CHECKPOINT_DIR / checkpoint_name)
            print(f"Resuming training from epoch {start_epoch}")

        initial_bias =  -0.2
        final_bias = 0.0
        decay_epochs = 3
        new_bias = self.model.final_fc.bias.data[0]
        for epoch in range(start_epoch, num_epochs):
            epoch += 1
            
            if epoch <= decay_epochs:
                new_bias = initial_bias + (final_bias - initial_bias) * (epoch - 1) / (decay_epochs - 1)
                self.model.final_fc.bias.data[0] = new_bias
                
            # Use simple train/val split from LibriSpeech
            if self.check_sample:
                audio_sanity_check(self.loaders['train'], self.speech_module, self.device)
                audio_sanity_check(self.loaders['val'], self.speech_module, self.device)
                self.check_sample = False

            train_loss = self.train(self.loaders, epoch, new_bias)
            # self.save_checkpoint(epoch, id=f"train_{train_loss:.4f}")
            # val_loss = self.validate(self.loaders, epoch)
            val_loss = 0.6
            # self.save_checkpoint(epoch, id=f"val_{val_loss:.4f}")

            if val_loss <= 0.5:
                self.save_checkpoint(epoch, id=f"target_reached_{val_loss:.2f}")

            self.epoch_losses['train'].append(train_loss)
            self.epoch_losses['val'].append(val_loss)

            tqdm.write(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def step(self, mode='train', batch=None, step_count=0):
        if mode == 'train':
            self.overall_step_count += 1
        # Input Shape: (batch_size, n_feats, seq_len)
        # Unpack the 6 elements from our new dataset (including unpadded_specs)
        inputs, labels, inputs_len, labels_len, file_name, unpadded_specs = batch
        log_debug = self.overall_step_count % 100 == 0 or self.overall_step_count == 1

        inputs_len = inputs_len // 2
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        # inputs_len, labels_len = inputs_len.to(self.device), labels_len.to(self.device)

        bs = inputs.shape[0]
        hidden = self.model._init_hidden(batch_size=bs, device=self.device)

        if log_debug:
            logger.debug(f"Step {step_count} ({mode}) - Before model inputs shape: {inputs.shape}")
        output, _ = self.model(inputs, inputs_len, hidden)
        if log_debug:
            logger.debug(f"Step {step_count} ({mode}) - Model output shape: {output.shape}")

        # Test: Check actual dimensions to determine if scaling is needed
        if step_count == 1:
            tqdm.write(f"{'=' * 10} Debug Info {'=' * 10}")
            tqdm.write(f"Step {step_count} ({mode}) - Raw inputs shape: {inputs.shape}")
            tqdm.write(f"Step {step_count} ({mode}) - Labels shape: {labels.shape}")
            tqdm.write(f"Step {step_count} ({mode}) - Input lengths: {inputs_len.shape}")
            tqdm.write(f"Step {step_count} ({mode}) - Label lengths: {labels_len.shape}")
            # nputs are (batch, n_mels, time), output is (time, batch, n_class)
            tqdm.write(f"Step 1 shape verification - Input: {inputs.shape}, Output: {output.shape}")
            tqdm.write(f"Step 1 time dimension check - Output time: {output.shape[0]}, Input time: {inputs.shape[2]}")

        # if mode == 'val':
        #     output = output / temperature

        _log_softmax = F.log_softmax(output, dim=2)

        # Whisper preserves temporal dimension, so output lengths = input lengths
        loss = self.criterion(_log_softmax, labels, inputs_len, labels_len)
        if (mode == 'val' and step_count % 100 == 0) or (step_count % 100 == 0) or True:
            sample = output.transpose(0, 1).contiguous()
            prediction = torch.argmax(sample[0], dim=1)
            tqdm.write(f"Decoded Label: {lc.decode(labels[0].tolist())}")
            tqdm.write(f"T1: {sample[0, 0, :10].tolist()}")
            # with open(self.log_file, 'a') as f:
            #     f.write(f"Step {step_count} | Loss: {loss.item():.4f}\nPrediction: {prediction.tolist()} | Labels: {labels[0].tolist()}\n")
            tqdm.write(f"Prediction: {ctc_decoder(prediction.tolist())} \nLabels: {labels[0].tolist()} ")
            log_file_path = config.LOG_DIR / "predictions.log"
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"Step {step_count} ({mode}) - {loss.item():.4f}\n")
                log_file.write(f"Prediction: {ctc_decoder(prediction.tolist())}\n")
                log_file.write(f"Ground Truth: {labels[0].tolist()}\n")
                log_file.write("=" * 50 + "\n")
        return loss, 0

    def train(self, loaders, epoch, new_bias):
        """Train method that works with LibriSpeech train/val structure"""
        self.model.train()

        # Get the train loader from the loaders dict
        train_loader = loaders['train']
        total_step = len(train_loader)
        total_loss = 0
        current_step = 0

        progress_bar = tqdm(total=total_step, desc=f"Epoch {epoch}/{config.H_PARAMS['TOTAL_EPOCH']} [Train]", dynamic_ncols=True, leave=True)
        for batch_idx, batch in enumerate(train_loader):
            current_step += 1

            self.optimizer.zero_grad()
            loss, penalty = self.step(mode='train', batch=batch, step_count=current_step)            
            loss.backward()
            loss = loss.item()
            total_loss += loss

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            lr = f"{self.scheduler.get_last_lr()[0]:.7f}".rstrip('0')
            progress_bar.set_postfix({
                "LR": lr,
                "Blank Bias": self.model.final_fc.bias.data[0].item(),
                "Loss": loss,
                "Avg Loss": total_loss / current_step
            })
            progress_bar.update(1)

            self.optimizer.step()
            self.scheduler.step()
            self.step_losses['train'].append(f"{loss:.4f}")

        progress_bar.close()
        return total_loss / total_step

    def get_blank_token_penalty(self, current_step):
        max_penalty = 0.5
        max_steps = 0.4 * self.total_steps
        if current_step < max_steps:
            return 0.0
        else:
            return min(max_penalty, (current_step - max_steps) / (self.total_steps - max_steps) * max_penalty)

        
    def validate(self, loaders, epoch):
        """Validation method that works with LibriSpeech train/val structure"""
        self.model.eval()
        
        # Get the val loader from the loaders dict
        val_loader = loaders['val']
        total_loss = 0
        total_step = len(val_loader)
        current_step = 0

        progress_bar = tqdm(total=total_step, desc=f"Epoch {epoch} [Validation]", dynamic_ncols=True)

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                current_step += 1
                loss, _ = self.step(mode='val', batch=batch, step_count=current_step)
                total_loss += loss.item()
                self.step_losses['val'].append(f"{loss.item():.4f}")
                progress_bar.set_postfix({
                    "Batch": f"{idx+1}/{len(val_loader)}",
                    "Loss": loss.item(),
                    "Avg Loss": total_loss / current_step,
                })
                progress_bar.update(1)

                if loss.item() <= 0.5 and not os.path.exists(config.CHECKPOINT_DIR / f"val_target_reached.pth"):
                    self.save_checkpoint(epoch, id=f"val_target_reached")

        progress_bar.close()
        return total_loss / total_step
    
    def print_grad_stats(self, model):
        # with open(self.log_file, 'a') as f:
        #     f.write(f"Learning Rate: {self.scheduler.get_last_lr()[0]}\n")
        #     f.write(f"Gradients:\n")
           
        for name, param in model.named_parameters():
            if param.requires_grad is not None:
                # f.write(f"{name}: {param.grad.norm():.4f}\n")
                tqdm.write(f"{name}: {param.grad.norm():.4f}")

    def save_checkpoint(self,epoch, id = random.randint(0, 10000)):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch_losses': self.epoch_losses,
            'step_losses': self.step_losses,
            'overall_step_count': self.overall_step_count
        }
        torch.save(checkpoint, config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}_{id}.pth")
        print(f"Checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch_losses = checkpoint.get('epoch_losses', {'train': [], 'val': []})
        self.step_losses = checkpoint.get('step_losses', {'train': [], 'val': []})
        start_epoch = checkpoint['epoch'] 
        self.overall_step_count = checkpoint.get('overall_step_count', 0)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
    
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Model().to(device)
    
    # Create speech module and load LibriSpeech data
    speech_module = SpeechModule()
    loaders = speech_module.load_and_create_dataloaders(
        subsets=config.LIBRISPEECH_SUBSETS,
        batch_size=config.H_PARAMS["BATCH_SIZE"]
    )
 
    total_steps = len(loaders['train']) * config.H_PARAMS["TOTAL_EPOCH"]

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config.H_PARAMS["BASE_LR"], weight_decay=0.0)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.H_PARAMS["BASE_LR"], total_steps=total_steps, div_factor=10, final_div_factor=100, pct_start=0.3, cycle_momentum=False)
    trainer = SpeechTrainer(model=model, loaders=loaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device, total_steps=total_steps, speech_module=speech_module)
    
    trainer.start(num_epochs=config.H_PARAMS["TOTAL_EPOCH"], resume=False, sort=True, checkpoint_name="checkpoint_epoch_5_train_48.9481.pth")
    
if __name__ == "__main__":
    main()