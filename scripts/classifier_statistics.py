import argparse
import os
import torch as th
import pickle
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from guided_diffusion.torch_classifiers import load_classifier
import time


def main():
    args = create_argparser().parse_args()
    args.output = os.path.join(args.output, args.classifier_name)

    dist_util.setup_dist()
    logger.configure(dir=args.output)

    classifier, preprocess, module_names = load_classifier(args.classifier_name)
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def class_eval(x):
        with th.no_grad():
            logits = classifier(preprocess(x))
            return logits

    activations_size = {}
    activations_mean = {}
    activations_var = {}

    def get_activation(name):
        def hook(model, input, output):
            act = output.detach()
            if act.ndim == 4:  ## BCHW format for conv layers
                act_size = act.shape[0] * act.shape[2] * act.shape[3]
                act_mean = act.sum((0, 2, 3), keepdim=True)
                act_var = (act**2).sum((0, 2, 3), keepdim=True)
            elif act.ndim == 2:  ## BN format for fc layers
                act_size = act.shape[0] * act.shape[1]
                act_mean = act.sum((0, 1), keepdim=True)
                act_var = (act**2).sum((0, 1), keepdim=True)
            elif act.ndim == 3:  ## BTC for transformer layers
                act_size = act.shape[0] * act.shape[1]
                act_mean = act.sum((0, 1), keepdim=True)
                act_var = (act**2).sum((0, 1), keepdim=True)
            else:
                raise ValueError(f"unexpected activation shape: {act.shape}")
            if name in activations_mean:
                activations_size[name] += act_size
                activations_mean[name] += act_mean
                activations_var[name] += act_var
            else:
                activations_size[name] = act_size
                activations_mean[name] = act_mean
                activations_var[name] = act_var

        return hook

    hooks = []
    for layer_name in module_names:
        layer = dict([*classifier.named_modules()])[layer_name]
        hook = layer.register_forward_hook(get_activation(layer_name))
        hooks.append(hook)

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
        class_cond=True,
        random_crop=False,
        random_flip=False,
        drop_last=False,
    )

    correct = 0
    correct_top5 = 0
    total = 0
    time_start = time.time()
    while total < args.num_samples:
        batch_data, extra = next(data)
        labels_data = extra["y"].to(dist_util.dev())
        batch_data = batch_data.to(dist_util.dev())
        class_eval_batch = class_eval(batch_data)
        _, predicted = th.topk(class_eval_batch.data, 5, dim=1)

        # Update total and correct counts
        total += labels_data.size(0)
        correct += (predicted[:, 0] == labels_data).sum().item()

        # Update correct_top5 count
        correct_top5 += (
            predicted.eq(labels_data.view(-1, 1).expand_as(predicted)).sum().item()
        )

        logger.log(
            f"evaluated {total} samples in {time.time() - time_start:.2f} seconds"
        )

    # Calculate accuracies
    accuracy = 100 * correct / total
    accuracy_top5 = 100 * correct_top5 / total
    logger.log(f"Accuracy: {accuracy:.2f}%")
    logger.log(f"Top-5 Accuracy: {accuracy_top5:.2f}%")

    for key in activations_size.keys():
        activations_mean[key] /= activations_size[key]
        activations_var[key] /= activations_size[key]
        activations_mean[key] = activations_mean[key].cpu().numpy()
        activations_var[key] = activations_var[key].cpu().numpy()

    results = {
        "args": args,
        "accuracy": accuracy,
        "accuracy_top5": accuracy_top5,
        "activations_mean": activations_mean,
        "activations_var": activations_var,
        "activations_size": activations_size,
    }

    # Save the results
    if dist.get_rank() == 0:
        # Save the arguments of the run
        out = os.path.join(logger.get_dir(), f"act_stat_{args.classifier_name}.pk")
        logger.log(f"saving activations statistics to {out}")
        with open(out, "wb") as handle:
            pickle.dump(results, handle)

    # Clean up
    for hook in hooks:
        hook.remove()

    dist.barrier()
    logger.log("Done!")


def create_argparser():
    defaults = dict(
        classifier_name="convnext_base",
        classifier_use_fp16=False,
        num_samples=10000,
        batch_size=128,
        data_dir="datasets/ILSVRC2012/validation",
        image_size=256,
        output=os.path.join(os.getcwd(), "classifier_statistics"),
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
