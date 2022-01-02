# MetGAN

Welcome to the repo of MetGAN - Generative Tumour Inpainting and Modality Synthesis in Light Sheet Microscopy.

See our paper here - https://arxiv.org/abs/2104.10993

To run our code , unzip the example data provided in data/sample_datasets and run:

python train.py --dataroot ./  --input_nc 1 --output_nc 1 --dataset_mode metgan  --name metgan_test  --model metgan  --load_size 256 --netG unet_spade8sm --crop_size 256 --direction AtoB --gpu_ids 0 --lambda_segmentation 100 --lambda_pair 10 --lambda_identity 0 


<img width="996" alt="Screenshot 2022-01-02 at 13 35 46" src="https://user-images.githubusercontent.com/94904575/147885887-0e6750a4-9ae8-4b1c-84a7-f372678d92e3.png">


<img width="993" alt="Screenshot 2022-01-02 at 13 36 41" src="https://user-images.githubusercontent.com/94904575/147885896-a6fcdabb-2160-416d-ab0b-2d0ddaa0ec0d.png">
