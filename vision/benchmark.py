import pretrainedmodels
import pretrainedmodels.utils as utils
import torchvision.models as models
import torch
from os import path
from torchvision.models import mobilenet_v2,  shufflenet_v2_x1_0, shufflenet_v2_x0_5
import random
import sys


DATA_DIR = './imagenet-val'
NUM_RETRY = 5
all_models = {
		'resnet18': pretrainedmodels.resnet18,
		'resnet34': pretrainedmodels.resnet34,
		'resnet50': pretrainedmodels.resnet50,
		'resnet101': pretrainedmodels.resnet101,
		'se_resnet50': pretrainedmodels.se_resnet50,
		'se_resnet101': pretrainedmodels.se_resnet101,
		'vgg11': pretrainedmodels.vgg11,
		'vgg11_bn': pretrainedmodels.vgg11_bn,
		'vgg13': pretrainedmodels.vgg13,
		'vgg13_bn': pretrainedmodels.vgg13_bn,
		'vgg16': pretrainedmodels.vgg16,
		'vgg16_bn': pretrainedmodels.vgg16_bn,
		'nasnetmobile': pretrainedmodels.nasnetamobile,
		'mobilenetv2': mobilenet_v2,
		'shufflenetv2_x05': shufflenet_v2_x0_5,
		'shufflenetv2_x10': shufflenet_v2_x1_0,
}

#
# def create_model_specific_data(model_name, model):
# 	transformed_images = []
# 	tf_img = utils.TransformImage(model)
# 	load_img = utils.LoadImage()
# 	for name in image_names:
# 		image_path = DATA_DIR + "/{}".format(name)
# 		input_img = load_img(image_path)
# 		input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
# 		input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
# 		transformed_images.append(input_tensor)
#
# 	transformed_images = random.choices(transformed_images, k=25)
# 	final_tensor = torch.stack(transformed_images)
# 	torch.save(final_tensor, '{}.pt'.format(model_name))
# 	print("{} => {}".format(model_name, final_tensor.shape))


def load_dataset():
	ds = torch.load('./vision/imagenet.pt')
	return ds


def load_model(model_name):
	model_func = all_models[model_name]
	if model_name in ['mobilenetv2', 'shufflenetv2_x05', 'shufflenetv2_x10']:
		return model_func(pretrained=True)
	else:
		return model_func(num_classes=1000)


def run_inference(model, dataset):
	for i in range(NUM_RETRY):
		for img in dataset:
			model(img)


if __name__ == "__main__":
	model_name = sys.argv[1]
	dataset = load_dataset().to('cpu:0')
	model = load_model(model_name).to('cpu:0')
	run_inference(model, dataset)
