#!/home/fmgarmor/miot_env/bin/python3
import os
import argparse
import json
import accelerate
import torch
from tqdm import trange, tqdm

import k_diffusion as K


def main():
    '''Examples of runs:
    $  nohup ./sample.py --config configs/config_wisdm_128x128_loto.json -n 2000  > sample-loto.log &

    $  nohup ./sample.py --config configs/config_wisdm_128x128_loso.json -n 2000  > sample-loso.log &
    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n-folds', type=int, default=3, help='the number of folds')
    p.add_argument('--n-classes', type=int, default=5, help='the number of classes')

    p.add_argument('--batch-size', type=int, default=16,
                   help='the batch size')
    p.add_argument('--config', type=str, default="configs/config_wisdm_128x128_loso.json",
                   help='the model config')
    p.add_argument('-n', type=int, default=2000,
                   help='the number of images to sample')
    
    p.add_argument('--result-folder', type=str, default='results/',
                   help='the output folder')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    args = p.parse_args()

    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    sampling_method = dataset_config["sampling"]

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    for cl in range(args.n_classes):
        for fold in range(args.n_folds):

            with open(f"{args.result_folder}/{sampling_method}/fold_{fold}/class_{cl}/model_fold_{fold}_class_{cl}_state.json", 'r') as fcc_file:
                fcc_data = json.load(fcc_file)
                print(fcc_data["latest_checkpoint"])

            ckpt = torch.load(fcc_data["latest_checkpoint"], map_location='cpu')
            # inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
            inner_model = K.models.ImageDenoiserModelV1(
                model_config['input_channels'], # input channels
                model_config['mapping_out'], # mapping out
                model_config['depths'], # depths
                model_config['channels'], # channels
                model_config['self_attn_depths']
                ).eval().requires_grad_(False).to(device)

            inner_model.load_state_dict(ckpt["model"])
            # print(inner_model)
            accelerator.print('Parameters:', K.utils.n_params(inner_model))
            model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

            sigma_min = model_config['sigma_min']
            sigma_max = model_config['sigma_max']

            @torch.no_grad()
            @K.utils.eval_mode(model)
            def run(cl, fold):
                if accelerator.is_local_main_process:
                    tqdm.write('Sampling...')
                sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
                def sample_fn(n):
                    x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
                    x_0 = K.sampling.sample_lms(model, x, sigmas, disable=not accelerator.is_local_main_process)
                    return x_0
                x_0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.n, args.batch_size)
                if accelerator.is_main_process:
                    path = f'{args.result_folder}{sampling_method}/fold_{fold}/class_{cl}/sample/'
                    os.makedirs(path, exist_ok=True)
                    for i, out in enumerate(x_0):
                        filename = f"{path}sample_{i:05}.png"
                        K.utils.to_pil_image(out).save(filename)
            try:
                run(cl, fold)
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()