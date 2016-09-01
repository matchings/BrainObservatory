import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

def get_responsive_cells(ns,cutoff=5):
	"""Return cells from the given NS object that have peak response greater than 5%"""
	#TODO documentation
	responsive_cells = ns.peak[ns.peak.peak_dff_ns >= 5].cell_specimen_id.values
	return responsive_cells


def get_ns_response_arrays(data_set,responsive_cells,stim_table):
	# TODO documentation
	timestamps,traces = data_set.get_dff_traces(cell_specimen_ids=responsive_cells)

	frames_arr = np.empty(0)
	images_arr = np.empty(0)
	for sweep in range(len(stim_table)):
		start = stim_table.iloc[sweep].start
		end = stim_table.iloc[sweep].end
		frames = np.arange(start,end)
		frames_arr = np.hstack((frames_arr,frames))
		image = stim_table.iloc[sweep].frame
		for i in range(len(frames)):
			images_arr = np.hstack((images_arr,image))

	traces_arr = np.empty((traces.shape[0],frames_arr.shape[0]))
	for t in range(traces.shape[0]):
		trace = np.empty(0)
		for sweep in range(len(stim_table)):
			start = stim_table.iloc[sweep].start
			end = stim_table.iloc[sweep].end
			tmp = traces[t,start:end]
			trace = np.hstack((trace,tmp))
		traces_arr[t,:] = trace
	return frames_arr, images_arr, traces_arr


def get_calcium_triggered_image_stack(cell_specimen_id, responsive_cell_id, ns, images, images_arr,traces_arr,thresh=0.5):
	# TODO documentation
	"""Returns a stack of images that seem to trigger responses, based on calcium fluorescence. Also returns a lot of other stuff."""
	cell_idx = np.where(ns.cell_id == cell_specimen_id)[0][0]

	pref_scene = ns.peak[ns.peak.cell_specimen_id == cell_specimen_id].scene_ns.values[0]
	pref_scene_sweeps = ns.stim_table[ns.stim_table.frame == pref_scene].index.values
	#	 cell_idx = data_set.get_cell_specimen_indices([cell])[0]

	condition_mean = ns.sweep_response[str(cell_idx)].iloc[pref_scene_sweeps].mean()
	frames = ns.sweeplength + ns.interlength * 2
	t = np.arange(0, frames)
	t_int = np.arange(0, frames, 6)
	t_int_ref = t_int - ns.interlength

	thresh_inds = np.where(traces_arr[responsive_cell_id, :] >= thresh)[0]
	thresh_inds = thresh_inds - 6
	thresh_vals = traces_arr[responsive_cell_id][thresh_inds]
	thresh_images = images_arr[thresh_inds]
	n_images = len(np.unique(thresh_images))
	img_stack = np.empty((thresh_images.shape[0], images[0, :, :].shape[0], images[0, :, :].shape[1]))
	for i, img in enumerate(thresh_images):
		img_stack[i, :, :] = images[img, :, :]
	return cell_idx, pref_scene, img_stack, thresh_vals, n_images, condition_mean, t, t_int, t_int_ref

def mean_image(img_stack,weights=None):
	"""Compute the mean image of an image stack, which is a 3 dimensional ndarray; the first index is the image index and the other two are pixels.
If weights are supplied, takes the weighted average. Returns a single image which is the (weighted) mean."""
	if weights is None:
		return np.mean(img_stack,axis=0)
	else:
		return np.average(img_stack,axis=0,weights=weights)

def plot_ns_summary(cell_specimen_id, responsive_cell_id, ns, images, frames_arr, images_arr, traces_arr, thresh=0.5,
					weighted=False, save_dir=None):
	# TODO documentation
	cell_idx, pref_scene, img_stack, thresh_vals, n_images, condition_mean, t, t_int, t_int_ref\
		= get_calcium_triggered_image_stack(cell_specimen_id, responsive_cell_id, ns, images, images_arr,traces_arr,thresh)
	if weighted:
		mean_img = mean_image(img_stack, weights=thresh_vals)
		# mean_img = np.average(img_stack, axis=0, weights=thresh_vals) / np.mean(images, axis=0)
	else:
		mean_img = mean_image(img_stack)
		# mean_img = np.mean(img_stack, axis=0) / np.mean(images, axis=0)

	condition_response = ns.response
	mean_image_responses = condition_response[:, cell_idx, 0]  # [image,cell,mean]

	fig, ax = plt.subplots(2, 2, figsize=(15, 10))
	ax = ax.ravel()
	ax[0].plot(t, condition_mean)
	ax[0].set_xticks(t_int)
	ax[0].set_xticklabels(t_int_ref / 30.)
	ax[0].set_xlabel('time after stimulus onset')
	ax[0].set_ylabel('dF/F')
	ax[0].set_title('cell ' + str(cell_idx) + ' mean response to pref condition')

	ax[1].imshow(images[pref_scene, :, :], cmap='gray')
	ax[1].set_title('pref_im: ' + str(pref_scene))
	ax[1].axis('off')

	ax[2].plot(mean_image_responses)
	ax[2].set_xlabel('image #')
	ax[2].set_ylabel('mean dF/F')
	ax[2].set_title('cell ' + str(cell_specimen_id) + ' normalized mean response to all conditions')

	ax[3].imshow(mean_img, cmap='gray')
	ax[3].axis('off')
	if weighted:
		ax[3].set_title('weighted mean of ' + str(n_images) + ' image conditions')
	else:
		ax[3].set_title('mean of ' + str(n_images) + ' image conditions')
	plt.tight_layout()
	if save_dir is not None:
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		save_path = os.path.join(save_dir, str(cell_specimen_id) + '.png')
		fig.savefig(save_path)
		save_matrix_as_image(mean_img, str(cell_specimen_id) + '_avgstim.png', save_path=save_dir)
	# ax[4:5].plot(traces[cell_idx,:])

def save_matrix_as_image(m, filename, save_path='', file_extension='png', color_map='gray'):
	"""Saves the given matrix as an image file without axes or excess space around it.

Input:
	m : a 2-dimensional ndarray
	filename : string for the name of the output image
	save_path : where to save the image. By default this is python's current working directory (see os.getcwd())
	file_extension : what format to save the image. Default is png
	color_map : color scheme for the output image. Default is 'gray'"""
	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.imshow(m, aspect='auto', cmap=color_map)
	fig.savefig(os.path.join(save_path,filename + file_extension))

def gaussian_blur(img,sigma=19.6):
	"""Applies a gaussian blur to given image and returns the result. By default, sigma is 19.6, which is roughly equal, in pixels, to 2 degrees of the mouse's visual field (ignoring the warping caused by using a flat screen)"""
	return gaussian_filter(img,sigma)
#
# def plot_mean_image(cell_specimen_id,ns,images,images_arr,traces_arr,thresh=0.3,weighted=False):
# 	# cell_idx = np.where(ns.cell_id==cell_specimen_id)[0][0]
# 	# thresh_inds = np.where(traces_arr[cell_idx,:]>=thresh)[0]
# 	# thresh_inds -= 6
# 	# thresh_images = images_arr[thresh_inds]
# 	# n_images = len(np.unique(thresh_images))
# 	# img_stack = np.empty((thresh_images.shape[0],images[0,:,:].shape[0],images[0,:,:].shape[1]))
# 	#
# 	# thresh_vals = traces_arr[cell_idx][thresh_inds]
# 	#
# 	# for i,img in enumerate(thresh_images):
# 	# 	img_stack[i,:,:] = images[img,:,:]
# 	img_stack,thresh_vals,n_images = get_calcium_triggered_image_stack(cell_specimen_id, ns, images, images_arr, traces_arr, thresh)
#
# 	if weighted:
# 		mean_img = mean_image(img_stack,thresh_vals)
# 		# mean_img = np.average(img_stack,axis=0,weights=thresh_vals)
# 	else:
# 		mean_img = mean_image(img_stack)
# 		# mean_img = np.mean(img_stack,axis=0)
#
# 	fig,ax=plt.subplots()
# 	ax.imshow(mean_img,cmap='gray')
# 	if weighted:
# 		ax.set_title('cell '+str(cell_specimen_id)+', weighted mean of '+str(n_images)+' images')
# 	else:
# 		ax.set_title('cell '+str(cell_specimen_id)+', mean of '+str(n_images)+' images')
#
#
#
# def get_calcium_triggered_image_stack(cell_specimen_id, ns, images, images_arr,traces_arr,thresh=0.3):
# 	"""Returns a stack of images that seem to trigger responses"""
# 	cell_idx = np.where(ns.cell_id == cell_specimen_id)[0][0]
# 	thresh_inds = np.where(traces_arr[cell_idx,:] >= thresh)[0]
# 	thresh_inds -= 6
# 	thresh_vals = traces_arr[cell_idx][thresh_inds]
# 	thresh_images = images_arr[thresh_inds]
# 	n_images = len(np.unique(thresh_images))
# 	img_stack = np.empty((thresh_images.shape[0], images[0, :, :].shape[0], images[0, :, :].shape[1]))
# 	for i, img in enumerate(thresh_images):
# 		img_stack[i, :, :] = images[img, :, :]
# 	return img_stack,thresh_vals,n_images
#
