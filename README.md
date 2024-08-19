# ResNet-101 and Image Classification

<p align="justify">
<strong>ResNet-101</strong>ResNet-101 is a deep convolutional neural network (CNN) that is part of the ResNet (Residual Network) family, introduced by Microsoft in 2015. It features 101 layers and builds on the innovative residual connection approach to facilitate the training of very deep networks. By using these residual shortcuts, ResNet-101 effectively mitigates the vanishing gradient problem and allows for improved performance in complex tasks. It is widely used in image classification and other computer vision applications, offering enhanced accuracy compared to shallower models while maintaining manageable computational demands.
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_ResNet101_ImageClassification_glxersgr.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_resnet101_train_uenvudnk.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_kxbchmmv/aicandy_model_pth_ruklihan.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_resnet101_test_mdilmqat.py --image_path ../image_test.jpg --model_path aicandy_model_out_kxbchmmv/aicandy_model_pth_ruklihan.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_resnet101_convert_onnx_bdoxvokd.py --model_path aicandy_model_out_kxbchmmv/aicandy_model_pth_ruklihan.pth --onnx_path aicandy_model_out_kxbchmmv/aicandy_model_onnx_pbkxuoqh.onnx --num_classes 2
```

### More Information

For a detailed overview of ResNet-101 and image classification, visit [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




