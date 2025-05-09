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
class SpeechTrainer:
    def __init__(self, model, loaders, criterion, optimizer, scheduler, device):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.check_sample = True
        self.step_losses = {
            'train': [],
            'val': []
        }
        self.epoch_losses = {
            'train': [],
            'val': []
        }

        if not os.path.exists(config.CHECKPOINT_DIR):
            os.makedirs(config.CHECKPOINT_DIR)

    def start(self, num_epochs=10):
        print(f"Training on {self.device}")

        for epoch in range(num_epochs):
            buckets = list(self.loaders.keys())
            random.shuffle(buckets)
            train_loaders, val_loaders = [], []
            for bucket in buckets:
                random_bucket = random.choice(buckets)
                print(f"Random Bucket: {random_bucket}")
                train_loaders.append(self.loaders[random_bucket]['train'])
                val_loaders.append(self.loaders[random_bucket]['val'])

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
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            self.save_checkpoint(epoch, id=f"loss_{val_loss:.2f}")

    def step(self, mode='train', batch=None):
        inputs, labels, inputs_len, labels_len, file_name = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        inputs_len, labels_len = inputs_len.to(self.device), labels_len.to(self.device)

        inputs = inputs.transpose(1, 2).contiguous()
        
        bs = inputs.shape[0]
        hidden = self.model._init_hidden(bs)
        hn, c0 = hidden[0].to(self.device), hidden[1].to(self.device)
        output, _ = self.model(inputs, (hn, c0))

        print(f"Output Shape: {output.shape} | Contiguous: {output.is_contiguous()}")

        _log_softmax = F.log_softmax(output, dim=2)

        print(f"Log Softmax Shape: {_log_softmax.shape} | Contiguous: {_log_softmax.is_contiguous()}")
        loss = self.criterion(_log_softmax, labels, inputs_len // 2, labels_len)

        sample = output.transpose(0, 1).contiguous()
        prediction = torch.argmax(sample[0], dim=1)
        print(f"prediction: {ctc_decoder(prediction.tolist())} \nLabels: {labels[0].tolist()} ")
        return loss

    def train(self, loaders, epoch):
        self.model.train()
        total_loss = 0.0
        total_step = sum([len(loader) for loader in loaders])
        current_step = 0
        print(f"Total Steps: {total_step}")
        for loader_idx, loader in enumerate(loaders):
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            for batch_idx, batch in enumerate(loader):
                print(f"Epoch {epoch + 1} | Loader {loader_idx + 1}/{len(loaders)} | Batch {batch_idx + 1}/{len(loader)}")
                print(f"Step {current_step + 1}/{total_step} | Loss: {total_loss / (current_step + 1):.4f}")
                self.optimizer.zero_grad()
                loss = self.step(mode='train', batch=batch)            
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                print(f"LR: {self.scheduler.get_last_lr()}")
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.norm():.4f}")

                self.optimizer.step()
                total_loss += loss.item()
                self.step_losses['train'].append(f"{loss.item():.4f}")
                print(f"{list(self.step_losses['train'][:5])} ... {list(self.step_losses['train'][-5:])}")
                current_step += 1

                if current_step == total_step // 2 or current_step == 1:
                    self.save_checkpoint(epoch, id=f"step_{current_step}")
                
                if current_step + 1 >= total_step:
                    self.save_checkpoint(epoch, id="last_step")
                    break
                print("=" * 100)

        return total_loss / total_step

    def validate(self, loaders, epoch):
        self.model.eval()
        total_loss = 0.0
        total_step = sum([len(loader) for loader in loaders])
        current_step = 0
        for loader in loaders:
            with torch.no_grad():
                for idx, batch in enumerate(loader):
                    current_step += 1
                    print(f"Validation Step {current_step + 1}/{total_step} | Loss: {total_loss / current_step:.4f}")
                    loss = self.step(mode='val', batch=batch)
                    total_loss += loss.item()
                    self.step_losses['val'].append(f"{loss.item():.4f}")

                    print(f"Validation Step {current_step}/{total_step} | Loss: {loss.item():.4f}")
                    print("=" * 100)
                
        return total_loss / total_step
    
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
            print(f"Input: {inputs[random_idx].shape}")
            print(f"Label: {labels[random_idx].shape}")
            print(f"Input length: {input_len[random_idx]}")
            print(f"Label length: {labels_len[random_idx]}")
            print(f"Input: {inputs[random_idx]}")
            print(f"Label: {labels[random_idx]}")
            
            audio_path = config.WAVS_PATH / f'{file_name[random_idx]}.wav'
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file {audio_path} does not exist.")
            audio, sr = torchaudio.load(audio_path)       

            spec = LogMelSpectrogram()(audio)
            print(f"Transcription: {lc.decode(labels[random_idx].tolist())}")
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
            'train_loss': self.epoch_losses['train'],
            'val_loss': self.epoch_losses['val'],
            'step_losses': self.step_losses,
        }
        torch.save(checkpoint, config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}_{id}.pth")
        print(f"Checkpoint saved at epoch {epoch}")
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    speech_module = SpeechModule()
    loaders = speech_module.loaders

    trainer = SpeechTrainer(model=model, loaders=loaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device)

    trainer.start(num_epochs=config.H_PARAMS["TOTAL_EPOCH"])
    
if __name__ == "__main__":
    main()