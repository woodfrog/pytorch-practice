import argparse
import os
import sys
import numpy as np
import time

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from models.transfom_net import TransformNet
from models.pretrained_vgg import VGG16

import utils


def check_model_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    transform_net = TransformNet()
    if args.pretrained_weight is not None:  # load pre-trained-weight
        transform_net.load_state_dict(torch.load(args.pretrained_weight))
        print('load weight for transfrom net from {}'.format(args.pretrained_weight))


    optimizer = Adam(transform_net.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = VGG16(requires_grad=False)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1)

    if args.cuda:
        transform_net.cuda()
        vgg.cuda()
        style = style.cuda()

    style_v = Variable(style)
    style_v = utils.normalize_batch(style_v)
    features_style = vgg(style_v)
    gram_style = [utils.gram_matrix(y) for y in features_style]  # prevent repeated computing

    for e in range(args.epochs):
        transform_net.train()  # turn on train mode
        sum_content_loss = 0.
        sum_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            if args.cuda:
                x = x.cuda()

            output = transform_net(x)

            # since we use Imagenet pre-trained vgg, we should normalize the images
            # using the mean of Imagenet
            output = utils.normalize_batch(output)
            x = utils.normalize_batch(x)

            # content loss
            features_output = vgg(output)  # a tuple of len 4, extracted from 4 different layers
            features_x = vgg(x)  # a tuple of len 4

            for idx in range(len(features_x)):
                features_x[idx] = features_x[idx].detach()
            content_loss = args.content_weight * mse_loss(features_output[2], features_x[2])

            style_loss = 0.
            for ft_y, gm_s in zip(features_output, gram_style):  # run over each layer
                gm_y = utils.gram_matrix(ft_y)
                gm_s = gm_s.detach()
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            sum_content_loss += content_loss.data[0]
            sum_style_loss += style_loss.data[0]

            if (batch_id + 1) % args.log_interval == 0:
                msg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  sum_content_loss / (batch_id + 1),
                                  sum_style_loss / (batch_id + 1),
                                  (sum_content_loss + sum_style_loss) / (batch_id + 1)
                )
                print(msg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transform_net.eval()  # set to eval mode
                if args.cuda:
                    transform_net.cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transform_net.state_dict(), ckpt_model_path)
                if args.cuda:
                    transform_net.cuda()
                transform_net.train()

    # save model
    transform_net.eval()
    if args.cuda:
        transform_net.cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transform_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(args):
    '''
        Stylize the given content image
    '''
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    if args.cuda:
        content_image = content_image.cuda()
    content_image = Variable(content_image, volatile=True)

    style_model = TransformNet()
    style_model.load_state_dict(torch.load(args.model))
    if args.cuda:
        style_model.cuda()
    output = style_model(content_image)
    if args.cuda:
        output = output.cpu()
    output_data = output.data[0]
    utils.save_image(args.output_image, output_data)


if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(description="parser for fast neural style transform")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    # The parser for training
    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=5,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=None,
                                  help="size of style-image, default is the original size of style image")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-4,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=50,
                                  help="number of batches after which the training loss is logged, default is 50")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=5000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument("--pretrained-weight", type=str, default=None,
                                  help="the path to the pre-trained weight of the transform net")


    # The parser for inference
    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image")
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_model_paths(args)
        train(args)
    else:
        stylize(args)

