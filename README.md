# Forward-backward diffusion

This is the codebase for the forward-backward experiments on ImageNet1k in [A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data](https://arxiv.org/abs/2402.16991).

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for forward-backward experiments. Please, refer to the corresponding [README](https://github.com/openai/guided-diffusion/blob/main/README.md) for installation instructions.  


## Data generation

The experiments take starting images as inputs, in this case the validation set of ImageNet from [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/) `datasets/ILSVRC2012/validation`.
The backward diffusion process is run with the [256x256 unconditional model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) trained by OpenAI.
The diffusion process is resampled to 250 steps from the 1000 initial ones to speed up the reverse process. The forward-backward experiment takes the argument `--step_reverse` between 0 and 250 indicating the time step at which the resampled diffusion process is reversed.
The output images are saved in `results/diffused_ILSVRC2012_validation`.
Additional arguments are:
* `--num_classes` the number of classes to be considered (from 1 to 1000);
* `--num_per_class` the number of images per class.

For example, the following script creates 10,000 images (10 per ImageNet1k class) reverting the diffusion prcess at the midpoint of the total time duration.

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

python forward_backward_dataset_sample.py --num_per_class 10 --num_classes 1000 --batch_size 32 --timestep_respacing 250 --model_path models/256x256_diffusion_uncond.pt $MODEL_FLAGS --data_dir datasets/ILSVRC2012/validation --step_reverse 125
```

## Classifier activation analysis

The following script collects statistics on the activations of a trained ConvNeXt Base architecture available on [torchvision](https://pytorch.org/vision/main/models/convnext.html) on the validation set of ImageNet1k. The output is saved in `classifier_statistics/convnext_base`.

```bash
python classifier_statistics.py --classifier_name convnext_base --num_samples 10000 --batch_size 128 --data_dir datasets/ILSVRC2012/validation
```

The following script evaluates the activations of the same network on the generated images and computes the differences with those of the starting images.

```bash
python evaluate_classifier.py --classifier_name convnext_base --starting_data_dir datasets/ILSVRC2012/validation --time_series 125 --batch_size 128
```

## Reference

Sclocchi, A., Favero, A. and Wyart, M., 2024. A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data. *arXiv preprint arXiv:2402.16991*.
