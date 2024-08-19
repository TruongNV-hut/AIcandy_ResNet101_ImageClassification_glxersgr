"""

@author:  AIcandy 
@website: aicandy.vn

"""

import torch
from PIL import Image
from torchvision import transforms
from aicandy_model_src_xmnpdags.aicandy_resnet101_model_ivtplvxo import resnet101


# python aicandy_resnet101_test_mdilmqat.py --image_path ../image_test.jpg --model_path aicandy_model_out_kxbchmmv/aicandy_model_pth_ruklihan.pth --label_path label.txt


def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = {int(line.split(": ")[0]): line.split(": ")[1].strip() for line in f}
    print('labels: ',labels)
    return labels

def predict(image_path, model_path, label_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(label_path)
    num_classes = len(labels)
    
    # Khởi tạo mô hình và tải trọng số
    model = resnet101(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Chuyển đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    predicted_class = predicted.item()
    return labels.get(predicted_class, "Unknown")

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='AIcandy.vn')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label file')

    args = parser.parse_args()
    predicted_class = predict(args.image_path, args.model_path, args.label_path)
    print(f'Predicted class: {predicted_class}')
