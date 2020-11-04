'''
  the difference between valid() and test() is that valid compute losses.
'''
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.nn as nn


def train(model, loader_train, criterion, optimizer, scheduler,
          epochs_num, log_interval_train=10, log_interval_valid=None,
          save_model_dir=None,
          loader_valid=None, loader_test=None, accumulate_grads_times=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(epochs_num):
        print(f'Epoch {epoch}/{epochs_num - 1}')
        model.train()
        epoch_loss, epoch_sample_num = 0.0, 0
        running_loss, running_sample_num = 0.0, 0

        for batch_idx, (images, masks) in enumerate(loader_train):
            images = images.to(device)
            masks = masks.to(device, dtype=torch.float32)
            outputs = model(images)
            loss = criterion(outputs, masks)

            if accumulate_grads_times is None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if batch_idx % accumulate_grads_times == 0:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            # statistics
            running_loss += loss.item()
            running_sample_num += len(images)

            epoch_loss += loss.item()
            epoch_sample_num += len(images)

            if log_interval_train is not None:
                if batch_idx % log_interval_train == log_interval_train - 1:
                    print(f'[epoch:{epoch}, batch:{batch_idx}, losses:{running_loss / log_interval_train:8.2f}')
                    running_loss, running_sample_num = 0, 0

        print(f'epoch{epoch} losses:{epoch_loss / (batch_idx+1):8.2f}')

        scheduler.step()

        if loader_valid:
            print('compute validation dataset...')
            validate(model, loader_valid, criterion, epoch, log_interval_valid)
        if loader_test:
            print('compute test dataset...')
            validate(model, loader_test, criterion, epoch, log_interval_valid)

        if save_model_dir:
            save_model_file = os.path.join(save_model_dir, f'epoch{epoch}.pth')
            print('save model:', save_model_file)
            os.makedirs(os.path.dirname(save_model_file), exist_ok=True)
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()

            torch.save(state_dict, save_model_file)



def validate(model, dataloader, criterion, epoch, log_interval=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    epoch_loss, epoch_sample_num = 0, 0
    running_loss, running_sample_num = 0, 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            running_sample_num += len(images)

            epoch_loss += loss.item()
            epoch_sample_num += len(images)

            if log_interval is not None:
                if batch_idx % log_interval == log_interval - 1:
                    print(f'[epoch:{epoch}, batch:{batch_idx}, losses:{running_loss / log_interval:8.2f}')
                    running_loss, running_sample_num = 0, 0

        print(f'epoch{epoch} losses:{epoch_loss / (batch_idx + 1):8.2f}')



