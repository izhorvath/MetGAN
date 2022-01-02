# MetGAN

Welcome to the repo of MetGAN - Generative Tumour Inpainting and Modality Synthesis in Light Sheet Microscopy.

See our paper here - https://arxiv.org/abs/2104.10993

To run our code , unzip the example data provided in data/sample_datasets and run:

python train.py --dataroot ./  --input_nc 1 --output_nc 1 --dataset_mode metgan  --name metgan_test  --model metgan  --load_size 256 --netG unet_spade8sm --crop_size 256 --direction AtoB --gpu_ids 0 --lambda_segmentation 100 --lambda_pair 10 --lambda_identity 0 
