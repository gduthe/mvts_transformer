import torch
import torch.optim as optim
from torch.utils.data import random_split, ConcatDataset
from model import MVTSTransformer
from dataset import NHCPPDataset
from box import Box
import yaml
import os
from shutil import copyfile
from tqdm import tqdm
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', help="path to the config yaml file", type=str, default='config.yml')

if __name__ == "__main__":
    # load the config file
    args = parser.parse_args()
    config = Box.from_yaml(filename=args.config, Loader=yaml.FullLoader)

    # init dataset, which can combine multiple datasets into one
    if isinstance(config.io_settings.dataset_path, list) or os.path.isdir(config.io_settings.dataset_path):
        if isinstance(config.io_settings.dataset_path, list):
            dset_list = config.io_settings.dataset_path
        else: # if dataset path is a directory, create a list with all hdf5 files inside
            dset_list =  [os.path.join(config.io_settings.dataset_path, h5file) for h5file
                                            in os.listdir(config.io_settings.dataset_path) if h5file.endswith('.hdf5')]

        datasets = [NHCPPDataset(path, mvts_length=config.model_settings.mvts_length,
                                            normalize_mvts=config.hyperparameters.normalize_mvts) for path in dset_list]
        nhcpp_dataset = ConcatDataset(datasets)
        nhcpp_dataset.num_channels = datasets[0].num_channels
        nhcpp_dataset.num_classes = datasets[0].num_classes
    else:
        nhcpp_dataset = NHCPPDataset(config.io_settings.dataset_path, mvts_length=config.model_settings.mvts_length,
                                                                    normalize_mvts=config.hyperparameters.normalize_mvts)

    # split the dataset and define dataloaders
    if config.run_settings.validate:
        data_lens = [int(config.run_settings.datasplit_train_ratio*len(nhcpp_dataset)),
                     len(nhcpp_dataset)-int(config.run_settings.datasplit_train_ratio*len(nhcpp_dataset))]
        train_dataset, validate_dataset = random_split(nhcpp_dataset, data_lens,
                                    generator=torch.Generator().manual_seed(config.run_settings.datasplit_random_state))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size,
                                           shuffle=True, num_workers=config.run_settings.num_t_workers, pin_memory=True,
                                           persistent_workers=False if config.run_settings.num_t_workers == 0 else True)
        validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=config.hyperparameters.batch_size,
                                           shuffle=True, num_workers=config.run_settings.num_v_workers, pin_memory=True,
                                           persistent_workers=False if config.run_settings.num_v_workers == 0 else True)
    else:
        train_loader = torch.utils.data.DataLoader(nhcpp_dataset, batch_size=config.hyperparameters.batch_size,
                                           shuffle=True, num_workers=config.run_settings.num_t_workers, pin_memory=True,
                                           persistent_workers=False if config.run_settings.num_t_workers == 0 else True)

    # create model saving dir and copy config file to run dir
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")[2:]
    current_run_dir = os.path.join(config.io_settings.run_dir, run_name)
    os.makedirs(os.path.join(current_run_dir, 'trained_models'))
    copyfile(args.config, os.path.join(current_run_dir, 'config.yml'))

    # use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else  'cpu')

    # initialize the model
    model = MVTSTransformer(num_channels=nhcpp_dataset.num_channels, num_classes=nhcpp_dataset.num_classes,
                             **config.model_settings)

    # if using a pretrained model, load it here
    if config.io_settings.pretrained_model:
        model.load_state_dict(torch.load(config.io_settings.pretrained_model, map_location=torch.device('cpu')))

    # send the models to the gpu if available
    model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(config.hyperparameters.learning_rate))

    # define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.hyperparameters.lr_decay_step, gamma=config.hyperparameters.lr_decay_gamma)

    # training loop
    print('Starting run {} on {}'.format(run_name, next(model.parameters()).device))
    pbar = tqdm(total=config.hyperparameters.epochs)
    pbar.set_description('Training')
    for epoch in range(config.hyperparameters.epochs):
        train_loss = 0
        train_acc = 0
        model.train()

        # mini-batch loop
        for i_batch, batch_data in enumerate(train_loader):
            # get batch data and send to the right device
            mvts, label = batch_data
            mvts = mvts.to(device)
            label = label.to(device)

            # reset the gradients back to zero and run forward pass
            optimizer.zero_grad()
            model_output = model(mvts)

            # compute the batch training loss and accuracy
            batch_loss = model.compute_loss(model_output, label)
            train_loss += batch_loss.item()
            train_acc += (model_output.argmax(dim=1) == label).sum() / torch.numel(label)

            # perform SGD parameter update
            batch_loss.backward()
            optimizer.step()

        # compute the epoch training loss and accuracy
        train_loss = train_loss / len(train_loader)
        train_acc = train_acc / len(train_loader)

        # step the scheduler
        scheduler.step()

        # save the trained model every n epochs
        if (epoch+1) % config.io_settings.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(current_run_dir, 'trained_models', 'e{}.pt'.format(epoch+1)))

        if config.run_settings.validate:
            # compute validation loss
            validation_loss = 0
            validation_acc = 0
            model.eval()
            with torch.no_grad():
                for i_batch, batch_data in enumerate(validate_loader):
                    # get data and send to the right device
                    mvts, label = batch_data
                    mvts = mvts.to(device)
                    label = label.to(device)

                    # compute the batch validation loss and accuracy
                    model_output = model(mvts)
                    validation_loss += model.compute_loss(model_output, label).item()
                    validation_acc += (model_output.argmax(dim=1) == label).sum() / torch.numel(label)

            # get the full dataset validation loss and accuracy for this epoch
            validation_loss =  validation_loss / len(validate_loader)
            validation_acc = validation_acc / len(validate_loader)

            # display losses and progress bar
            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}','Validation Loss': f'{validation_loss:.8f}',
                              'Train Acc': f'{train_acc:.8f}','Validation Acc': f'{validation_acc:.8f}'})
            pbar.update(1)

            # save the model with the best validation loss
            if epoch == 0:
                best_validation_loss = validation_loss
            else:
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    torch.save(model.state_dict(), os.path.join(current_run_dir, 'trained_models', 'best.pt'))

        else:
            # display losses and progress bar
            pbar.set_postfix({'Train Loss': f'{train_loss:.8f}', 'Train Acc': f'{train_acc:.8f}'})
            pbar.update(1)