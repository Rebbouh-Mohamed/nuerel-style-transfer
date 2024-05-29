from utils import gram_matrix,prepare_img,prepare_model,total_variation,TO_img,save_image,save_and_maybe_display
# from video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import streamlit as st
import time 


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
	target_content_representation = target_representations[0]
	target_style_representation = target_representations[1]

	current_set_of_feature_maps = neural_net(optimizing_img)

	current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
	content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

	style_loss = 0.0
	current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
	for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
		style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
	style_loss /= len(target_style_representation)

	tv_loss = total_variation(optimizing_img)

	total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

	return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
	# Builds function that performs a step in the tuning loop
	def tuning_step(optimizing_img):
		total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
		# Computes gradients
		total_loss.backward()
		# Updates parameters and zeroes gradients
		optimizer.step()
		optimizer.zero_grad()
		return total_loss, content_loss, style_loss, tv_loss

	# Returns the function that will be called inside the tuning loop
	return tuning_step

def neural_style_transfer(config):
	content_img_path = config['content_images_dir']
	style_img_path = config['style_images_dir']

	# out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
	# dump_path = os.path.join(config['output_img_dir'], out_dir_name)
	# os.makedirs(dump_path, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	content_img = prepare_img(content_img_path, config['height'], device)
	style_img = prepare_img(style_img_path, config['height'], device)

   
	#iniial img that start optimmizing
	init_img = content_img
	

	# we are tuning optimizing_img's pixels! (that's why requires_grad=True)
	optimizing_img = Variable(init_img, requires_grad=True)

	neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)

	content_img_set_of_feature_maps = neural_net(content_img)
	style_img_set_of_feature_maps = neural_net(style_img)

	target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
	target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
	target_representations = [target_content_representation, target_style_representation]

	# magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
	num_of_iterations = {
		"lbfgs": 1000,
		"adam":3000
	}

	#
	# Start of optimization procedure
	#
	
		# line_search_fn does not seem to have significant impact on result
	optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
	cnt = 0



	def closure():
		nonlocal cnt
		if torch.is_grad_enabled():
			optimizer.zero_grad()
		total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
		if total_loss.requires_grad:
			total_loss.backward()
		with torch.no_grad():
			print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
			#save_and_maybe_display(optimizing_img, "out", config, cnt, num_of_iterations['lbfgs'], should_display=False)

		cnt += 1
		if cnt <= num_of_iterations['lbfgs']:
			config['bar'].empty()
			config['bar'].progress(cnt/num_of_iterations['lbfgs'])
		

		return total_loss

	optimizer.step(closure)

	return TO_img(optimizing_img)

