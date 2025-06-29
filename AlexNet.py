import argparse
import torch
import utils
import AlexNet_Module
from utils import get_breastmnist

# Wrap utils.train to auto-squeeze any 2D targets to 1D
_original_train = utils.train

def _train_squeeze(model, train_loader, test_loader, args):
    def gen(loader):
        for imgs, targets in loader:
            targets = targets.squeeze()
            yield imgs, targets
    class WrappedLoader:
        def __init__(self, loader):
            self.dataset = loader.dataset
            self.loader = loader
        def __iter__(self):
            return gen(self.loader)
    return _original_train(model,
                           WrappedLoader(train_loader),
                           WrappedLoader(test_loader),
                           args)

# Override train with squeeze wrapper
utils.train = _train_squeeze

# Initialize dataloaders and device
_dataloaders = utils.dataloaders()
_dataloaders_recon = utils.dataloaders_recon()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Argument parser helper
def _arparse(epoch, batch_size, lr, lr_decay, r, num_save_epoch,
             train_save_dir, train_eval, pretrained, pretrained_weight,
             num_class, save_all, reconstruction, recon_alpha):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--epochs', default=epoch, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--lr', default=lr, type=float)
    parser.add_argument('--lr_decay', default=lr_decay, type=float)
    parser.add_argument('-r', '--routings', default=r, type=int)
    parser.add_argument('-num_save_epoch', default=num_save_epoch, type=int)
    parser.add_argument('--num_class', default=num_class, type=int)
    parser.add_argument('--reconstruction_alpha', default=recon_alpha, type=float)
    parser.add_argument('--train_save_dir', default=train_save_dir, type=str)
    parser.add_argument('--train_eval', default=train_eval, type=bool)
    parser.add_argument('--save_all', default=save_all, type=bool)
    parser.add_argument('--pretrained', default=pretrained, type=bool)
    parser.add_argument('--pretrained_weight', default=pretrained_weight, type=str)
    parser.add_argument('--reconstruction', default=reconstruction, type=bool)
    return parser.parse_args()

# Main entry point
def main(DATASET, NETWORK):

    # CIFAR10, CIFAR100, MNIST, FashionMNIST, FOOD101, FLOWER102, SVHN
    # (your original dataset blocks remain unchanged)

    # BreastMNIST: DATASET=0 and AlexNet (NETWORK=2)
    # if DATASET == 0 and NETWORK == 2:
    #     args = _arparse(
    #         epoch=100,
    #         batch_size=32,
    #         lr=0.001,
    #         lr_decay=0.995,
    #         r=3,
    #         num_save_epoch=1,
    #         train_save_dir='./Result/BreastMNIST/Alexnet/train',
    #         train_eval=True,
    #         save_all=True,
    #         pretrained=False,
    #         pretrained_weight=None,
    #         num_class=2,
    #         reconstruction=False,
    #         recon_alpha=0.0005
    #     )
    #     train_loader, test_loader = get_breastmnist(args.batch_size)
    #     model = AlexNet_Module.AlexNet_MNIST(device).to(device)
    #     print(f'The AlexNet for BreastMNIST architecture is shown:\n {model}')
    #     utils.train(model, train_loader, test_loader, args)

    if DATASET == 0:
        # BreastMNIST dataset
        args = _arparse(
            epoch=100,
            batch_size=32,
            lr=0.001,
            lr_decay=0.995,
            r=3,
            num_save_epoch=1,
            train_save_dir=f'./Result/BreastMNIST/Alexnet/{NETWORK}/train',
            train_eval=True,
            save_all=True,
            pretrained=False,
            pretrained_weight=None,
            num_class=2,
            reconstruction=(NETWORK in [4,5]),  # reconstruction only for 4 ve 5
            recon_alpha=0.0005
        )
        train_loader, test_loader = get_breastmnist(args.batch_size)

        # Switch-case benzeri model seçimi
        if NETWORK == 1:
            model = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
        elif NETWORK == 2:
            model = AlexNet_Module.AlexNet_MNIST(device).to(device)
        elif NETWORK == 3:
            model = AlexNet_Module.CapsNet_MNIST(device).to(device)
        elif NETWORK == 4:
            model = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
        elif NETWORK == 5:
            model = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
        elif NETWORK == 6:
            model = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
        else:
            raise ValueError(f"Unsupported NETWORK for BreastMNIST: {NETWORK}")

        print(f'The model for BreastMNIST (network {NETWORK}) is:\n{model}')
        utils.train(model, train_loader, test_loader, args)

if __name__ == '__main__':
    # 0: BreastMNIST, 1: CIFAR10, 2: CIFAR100, 3: MNIST, 4: FashionMNIST, 5: FOOD101, 6: FLOWER102, 7: SVHN
    DATASET = 0
    # 1: AlexCapsNet, 2: AlexNet, 3: CapsNet, 4: CapsNet_Recon, 5: AlexCapsNet_Recon, 6: S_AlexCapsNet
    NETWORK = 1

    main(DATASET, NETWORK)
