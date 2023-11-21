import argparse

from src.train_all import train


def main(root_dir, batch_size, model_name, device, max_epoch):
    train(root_dir, batch_size, model_name, device, max_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True,
                        help="root directory of dataset")
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=100)
    args = parser.parse_args()

    main(args.root_dir,
         args.batch_size,
         args.model_name,
         args.device,
         args.max_epoch)
