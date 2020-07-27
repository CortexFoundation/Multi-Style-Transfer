import os
import logging
import mxnet as mx
from models import Net
from utils import Options, Configs, preprocess_batch, tensor_load_rgbimage, tensor_save_bgrimage, tensor_save_rgbimage


def evaluate():
    if mx.context.num_gpus() > 0:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu(0)

    # loading configs
    args = Options().parse()
    cfg = Configs(args.config_path)
    # set logging level
    logging.basicConfig(level=logging.INFO)

    # images
    content_image = tensor_load_rgbimage(cfg.content_image, ctx, size=cfg.val_img_size, keep_asp=True)
    style_image = tensor_load_rgbimage(cfg.style_image, ctx, size=cfg.val_style_size)
    style_image = preprocess_batch(style_image)
    # model
    style_model = Net(ngf=cfg.ngf)
    style_model.collect_params().load(cfg.val_model, ctx=ctx)
    # forward
    output = style_model(content_image, style_image)
    # save img
    tensor_save_bgrimage(output[0], cfg.output_img)
    logging.info("Save img to {}".format(cfg.output_img))


def main():
    print("Mode: eval")
    evaluate()

if __name__ == "__main__":
    main()
