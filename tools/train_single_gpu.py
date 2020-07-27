import os
import time
import random
import logging
import datetime
import mxnet as mx
import numpy as np
from PIL import Image
from mxnet import autograd, gluon
from mxnet.gluon import nn, HybridBlock, Parameter, ParameterDict
from dataset import ImageFolder, StyleLoader
from models import Vgg16, Net
from utils import Options, Configs, subtract_imagenet_mean_preprocess_batch, preprocess_batch, subtract_imagenet_mean_batch


def train():
    if mx.context.num_gpus() > 0:
        ctx = mx.gpu()
    else:
        raise RuntimeError('There is no GPU device!')

    # loading configs
    args = Options().parse()
    cfg = Configs(args.config_path)
    # set logging level
    logging.basicConfig(level=logging.INFO)
    # set random seed
    np.random.seed(cfg.seed)

    # build dataset and loader
    content_dataset = ImageFolder(cfg.content_dataset, cfg.img_size, ctx=ctx)
    style_dataset = StyleLoader(cfg.style_dataset, cfg.style_size, ctx=ctx)
    content_loader = gluon.data.DataLoader(content_dataset, batch_size=cfg.batch_size, \
                                            last_batch='discard')
    
    vgg = Vgg16()
    vgg._init_weights(fixed=True, pretrain_path=cfg.vgg_check_point, ctx=ctx)

    style_model = Net(ngf=cfg.ngf)
    if cfg.resume is not None:
        print("Resuming from {} ...".format(cfg.resume))
        style_model.collect_params().load(cfg.resume, ctx=ctx)
    else:
        style_model.initialize(mx.initializer.MSRAPrelu(), ctx=ctx)
    print("Style model:")
    print(style_model)

    # build trainer
    lr_sche = mx.lr_scheduler.FactorScheduler(
        step=170000,
        factor=0.1,
        base_lr=cfg.base_lr
        #warmup_begin_lr=cfg.base_lr/3.0,
        #warmup_steps=300,
    )
    opt = mx.optimizer.Optimizer.create_optimizer('adam', lr_scheduler=lr_sche)
    trainer = gluon.Trainer(style_model.collect_params(), optimizer=opt)

    loss_fn = gluon.loss.L2Loss()

    logging.info("Start training with total {} epoch".format(cfg.total_epoch))
    iteration = 0
    total_time = 0.0
    num_batch = content_loader.__len__() * cfg.total_epoch
    for epoch in range(cfg.total_epoch):
        sum_content_loss = 0.0
        sum_style_loss = 0.0
        for batch_id, content_imgs in enumerate(content_loader):
            iteration += 1
            s = time.time()
            style_image = style_dataset.get(batch_id)

            style_vgg_input = subtract_imagenet_mean_preprocess_batch(style_image.copy())
            style_image = preprocess_batch(style_image)
            style_features = vgg(style_vgg_input)
            style_features = [style_model.gram.gram_matrix(mx.nd, f) for f in style_features]

            content_vgg_input = subtract_imagenet_mean_preprocess_batch(content_imgs.copy())
            content_features = vgg(content_vgg_input)[1]

            with autograd.record():
                y = style_model(content_imgs, style_image)
                y = subtract_imagenet_mean_batch(y)
                y_features = vgg(y)

                content_loss = 2 * cfg.content_weight * loss_fn(y_features[1], content_features)
                style_loss = 0.0
                for m in range(len(y_features)):
                    gram_y = style_model.gram.gram_matrix(mx.nd, y_features[m])
                    _, C, _ = style_features[m].shape
                    gram_s = mx.nd.expand_dims(style_features[m], 0).broadcast_to((gram_y.shape[0], 1, C, C,))
                    style_loss = style_loss + 2 * cfg.style_weight * loss_fn(gram_y, gram_s)
                total_loss = content_loss + style_loss
                total_loss.backward()

            trainer.step(cfg.batch_size)
            mx.nd.waitall()
            e = time.time()
            total_time += e - s
            sum_content_loss += content_loss[0]
            sum_style_loss += style_loss[0]
            if iteration % cfg.log_interval == 0:
                itera_sec = total_time / iteration
                eta_str = str(datetime.timedelta(seconds=int((num_batch-iteration)*itera_sec)))
                mesg = "{} Epoch [{}]:\t[{}/{}]\tTime:{:.2f}s\tETA:{}\tlr:{:.4f}\tcontent: {:.3f}\tstyle: {:.3f}\ttotal: {:.3f}".format(
                    time.strftime("%H:%M:%S", time.localtime()), epoch + 1, batch_id+1, 
                                content_loader.__len__(), itera_sec, eta_str,
                                trainer.optimizer.learning_rate,
                                sum_content_loss.asnumpy()[0] / (batch_id + 1),
                                sum_style_loss.asnumpy()[0] / (batch_id + 1),
                                (sum_content_loss + sum_style_loss).asnumpy()[0] / (batch_id + 1)
                )
                logging.info(mesg)
                ctx.empty_cache()
        save_model_filename = "Epoch_" + str(epoch + 1) +  "_" + str(time.ctime()).replace(' ', '_') + \
                "_" + str(cfg.content_weight) + "_" + str(cfg.style_weight) + ".params"
        if not os.path.isdir(cfg.save_model_dir):
            os.mkdir(cfg.save_model_dir)
        save_model_path = os.path.join(cfg.save_model_dir, save_model_filename)
        logging.info("Saving parameters to {}".format(save_model_path))
        style_model.collect_params().save(save_model_path)


def main():
    print("Mode: train")
    train()


if __name__ == "__main__":
    main()