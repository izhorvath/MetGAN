# MetGAN

Welcome to the repo of MetGAN - Generative Tumour Inpainting and Modality Synthesis in Light Sheet Microscopy.

See our paper here - https://arxiv.org/abs/2104.10993

The goal of this project is to combine real anatomical information with user-defined labels, in order to generate realistic images containint metastases.

<img width="911" alt="Screenshot 2022-01-02 at 13 38 14" src="https://user-images.githubusercontent.com/94904575/147886043-f2c2fc7a-8449-4629-ba7a-7a8ed38cfd68.png">

For this, we use 2 generators and 2 discriminators, alongside a pretrained segmentor. The setup is trianed in a cycle consistent manner.

<img width="996" alt="Screenshot 2022-01-02 at 13 35 46" src="https://user-images.githubusercontent.com/94904575/147886047-85081c94-5a35-48a2-9f82-97153792bdce.png">

Additionally, our generators are designed with 2 separate pathways: one for the anatomy image,having a U-net architecture, and one for the label, using Spade ResNet Blocks.

<img width="559" alt="Screenshot 2022-01-02 at 13 39 34" src="https://user-images.githubusercontent.com/94904575/147886052-9720a9fd-a95c-43b9-af38-0cb01cd4447c.png">

Our experiments show that this allows us to obtain superior results to other SOTA methods (see paper)

<img width="762" alt="Screenshot 2022-01-02 at 13 40 01" src="https://user-images.githubusercontent.com/94904575/147886060-0ed4a8d1-375e-4cfd-ac95-ebbcef841bae.png">

<img width="787" alt="Screenshot 2022-01-02 at 13 40 17" src="https://user-images.githubusercontent.com/94904575/147886067-ec5aa4df-74ee-4814-8666-94a17084cd06.png">

<img width="993" alt="Screenshot 2022-01-02 at 13 36 41" src="https://user-images.githubusercontent.com/94904575/147886073-07395159-c9a7-45d1-b55f-cbfddfe80f55.png">

<img width="824" alt="Screenshot 2022-01-02 at 13 40 32" src="https://user-images.githubusercontent.com/94904575/147886071-b0a1b395-61e4-4352-a9ab-5678f02382ee.png">

**To run our code** , unzip the example data provided in data/sample_datasets and run:

python train.py --dataroot ./  --input_nc 1 --output_nc 1 --dataset_mode metgan  --name metgan_test  --model metgan  --load_size 256 --netG unet_spade8sm --crop_size 256 --direction AtoB --gpu_ids 0 --lambda_segmentation 100 --lambda_pair 10 --lambda_identity 0 





Note: the checkpoint for the segmentation model can be downloaded, for example, from here : https://codeocean.com/capsule/0792467/tree/v1
After downloading this file, please update its path in models/metgan_model.py , line 95





