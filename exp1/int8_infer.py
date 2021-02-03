from torchvision import transforms, datasets
import torch
import torch.nn as nn
from dataset import QDataset, testloader
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch2trt import torch2trt
import argparse
from models import model_factory
import time
import tensorrt as trt
import os


def test(testmodel):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = testmodel(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test for Cifar10 w/ or w/o trt')
    parser.add_argument('--gpu', '-p', action='store_true', help='Trained on GPU')
    parser.add_argument('--model', '-m', default='alexnet', type=str, help='Name of Network')
    
    args = parser.parse_args()

    model_name = args.model
    model = model_factory(model_name)
    print("Testing model: %s" % model_name)

    if args.gpu and torch.cuda.is_available():
        # CuDNN must be enabled for FP16 training.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    else:
        print("-p is a must when executing this script. Exiting...")
        exit()  

    model.load_state_dict(torch.load('./weights/'+model_name+'.pt')['net'])
    
    accbefore=torch.load('./weights/'+model_name+'.pt')['acc']

    cali_augmentation = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.Resize((32, 32), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    cali_cifar10 = QDataset(transform=cali_augmentation)
    model.eval() 

    x = torch.randn([1, 3, 32, 32]).cuda() #no .half()?
    model_trt_int8 = torch2trt(model, [x],
                          fp16_mode=True,
                          int8_mode=True,
                          max_batch_size=1,
                          int8_calib_dataset=cali_cifar10,
                          int8_calib_algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2)
    '''
    start_time = time.time()
    accbefore = test(model)
    torch.cuda.synchronize()            # wait for cuda to finish (cuda is asynchronous!)
    time_spent = time.time() - start_time
    print("Acc for fp32: %.2f; Time spent: %.8fms" % (accbefore, time_spent))

    start_time = time.time()
    accafter = test(model_trt_int8)
    torch.cuda.synchronize()            # wait for cuda to finish (cuda is asynchronous!)
    time_spent = time.time() - start_time
    print("Acc for trt int8: %.2f; Time spent: %.8fms" % (accafter, time_spent))
    '''

    test_image = os.path.join('test.jpg')
    img = Image.open(test_image)
    img_tensor = cali_augmentation(img)
    #print(img_tensor)
    img_tensor = img_tensor.unsqueeze_(0)
    #print(img_tensor)
    input = img_tensor.cuda()
    
    start_time = time.time()
    y_fp32 = model(input)
    torch.cuda.synchronize()
    time_spent_fp32 = time.time()- start_time
    print('Time Spent for fp32: {:.2f}ms'.format(time_spent_fp32 * 1000))

    start_time = time.time()
    y_int8 = model_trt_int8(input)
    torch.cuda.synchronize()
    time_spent_int8 = time.time()- start_time
    print('Time Spent for int8: {:.2f}ms'.format(time_spent_int8 * 1000))
    #print(y)
    percentage_fp32 = torch.softmax(y_fp32[0], dim=0) * 100
    percentage_int8 = torch.softmax(y_int8[0], dim=0) * 100
    #print(percentage)
    cl_fp32, index_fp32 = torch.max(percentage_fp32, 0)
    cl_int8, index_int8 = torch.max(percentage_int8, 0)
    classes = ['plane','car','bird','cat','deer','dog','frog','horse', 'ship','truck']

    
    font = ImageFont.truetype('LiberationSans-Regular.ttf', 30)

    draw = ImageDraw.Draw(img)
    text = 'Mode: fp32, '+ '{:.2}ms '.format(time_spent_fp32*1000) + str(classes[index_fp32]) + ' (' + '{:.2f}'.format(cl_fp32.item()) + '%' + ')'
    draw.text((0,0), text, font=font, fill="#ffffff", spacing=0, align='left') 

    text = 'Mode: int8, ' + '{:.2}ms '.format(time_spent_int8*1000) + str(classes[index_int8]) + ' (' + '{:.2f}'.format(cl_int8.item()) + '%' + ')'
    draw.text((0,40), text, font=font, fill="#ff00ff", spacing=0, align='left') 

    img.save(test_image,'jpeg')
    
    