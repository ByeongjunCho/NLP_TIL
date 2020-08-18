import pytorch_lightning as pl
import pytorch_lightning.logging
import argparse

from module import ImageGPT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = ImageGPT.add_model_specific_args(parser)

    parser.add_argument("--train_x", default="./data/train_x.npy")
    parser.add_argument("--train_y", default="./data/train_y.npy")
    parser.add_argument("--test_x", default="./data/test_x.npy")
    parser.add_argument("--test_y", default="./data/test_y.npy")

    parser.add_argument("--gpus", default="0")
    subparsers = parser.add_subparsers()
    parser.add_argument("--pretrained", type=str, default=None)
    # parser_train.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, default='fmnist_gen')

    parser_test = subparsers.add_parser("test")
    parser_test.add_argument("checkpoint", type=str)

    args = parser.parse_args()

    logger = pl.logging.TensorBoardLogger("logs", name=args.name)

    model = ImageGPT(args)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        max_steps=args.steps,
        gpus=args.gpus,
        checkpoint_callback=checkpoint,
        logger=logger,
    )

    trainer.fit(model)