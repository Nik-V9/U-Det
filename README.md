# U-Det: A Modified U-Net architecture with bidirectional feature network for lung nodule segmentation

This repository contains the source code for the Paper titled "U-Det: A Modified U-Net architecture with bidirectional feature network for lung nodule segmentation". The source code contains the implementations for Testing and Training U-Det Deep Learning Model for Pulmonary Nodule Segmentation. The repository also contains the code for Data augmentations and Data Loaders for converting 3-D CT scan data to appropriate Slices of size 512*512. The techinal details and findings of the paper are available at the following link:

Arxiv Link: https://arxiv.org/abs/2003.09293

## Repository Structure & Files
The structure of the repository is represented below :
```bash
|   LICENSE
|   main.py
|   train.py
|   test.py
|   
+---Models
|       Encoder_BIFPN.py	
|       Encoder_BiFPN_ReLU.py		
|       UDet.py	
|       UDet_relu.py	
|       UNet_mish.py	
|       bifpn.py
|       unet.py
|       
+---Model_Helpers
|       metrics.py
|       model_helper.py
|       
+---Custom_Functions
|       custom_data_aug.py
|       custom_losses.py             
\---Data_Loader
        load_3D_data.py
```
## Demo

The Technical Details and Code Demo shall be updated soon! Make sure you check it out once it's updated!

## Data Structure and Pre-Processing

The dataset structure is the same as the one available at https://luna16.grand-challenge.org/Data/ . Additionally, Inside the data root folder (i.e. where you have your data stored) you should have two folders: one called imgs and one called masks. The imgs and masks folders should contain the slice images and segmentation masks after preprocessing the '.mhd' CT files and annotations for lung nodules. All models, results, etc. are saved to this same root directory. The argument --data_root_dir is the only required argument and should be set to the directory containing your imgs and masks folders.

The LUNA16 Data is pre-processed into images and masks with the help of RadIO python library. The required resources are present here: https://analysiscenter.github.io/radio/intro/preprocessing.html

## Citation

If you use any part of the code for any research implementation or project please do cite us, it would help out a lot. The citation is as follows:

```

@misc{keetha2020udet,
    title={U-Det: A Modified U-Net architecture with bidirectional feature network for lung nodule segmentation},
    author={Nikhil Varma Keetha and Samson Anosh Babu P and Chandra Sekhara Rao Annavarapu},
    year={2020},
    eprint={2003.09293},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}

```
The Pre-Print is currently under review at Springer Journal of Medical Systems.

## License

MIT License

Copyright (c) 2020 Nikhil, Samson, ACS Rao (U-Det)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
