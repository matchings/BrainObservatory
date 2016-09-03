
# coding: utf-8

# In[1]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# In[2]:

#processing on all traces
def get_ns_response_arrays(traces,data_set,stim_table):

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


# In[192]:

def plot_ns_summary(cell_specimen_id,ns,images,frames_arr,images_arr,traces_arr,thresh=0.5,weighted=False,save_dir=False): 
        
    pref_scene = ns.peak[ns.peak.cell_specimen_id==cell_specimen_id].scene_ns.values[0]
    pref_scene_sweeps = ns.stim_table[ns.stim_table.frame==pref_scene].index.values
    #     cell_idx = data_set.get_cell_specimen_indices([cell])[0]
    cell_idx = np.where(ns.cell_id==cell_specimen_id)[0][0]

    condition_mean = ns.sweep_response[str(cell_idx)].iloc[pref_scene_sweeps].mean()
    frames = ns.sweeplength+ns.interlength*2
    t = np.arange(0,frames)
    t_int = np.arange(0,frames,6)
    t_int_ref = t_int - ns.interlength 

    cell_idx = np.where(ns.cell_id==cell_specimen_id)[0][0]
    thresh_inds = np.where(traces_arr[cell_idx,:]>=thresh)[0]
    thresh_inds = thresh_inds - 6
    thresh_vals = traces_arr[cell_idx][thresh_inds]
    thresh_images = images_arr[thresh_inds]
    n_images = len(np.unique(thresh_images))
    img_stack = np.empty((thresh_images.shape[0],images[0,:,:].shape[0],images[0,:,:].shape[1]))
    for i,img in enumerate(thresh_images):
        img_stack[i,:,:] = images[img,:,:]
    if weighted: 
        mean_image = np.average(img_stack,axis=0,weights=thresh_vals)/np.mean(images,axis=0)
    else: 
        mean_image = np.mean(img_stack,axis=0)/np.mean(images,axis=0)

    condition_response = ns.response
    mean_image_responses = condition_response[:,cell_idx,0] #[image,cell,mean]
    
    fig,ax = plt.subplots(2,2,figsize=(15,10))
    ax = ax.ravel()
    ax[0].plot(t,condition_mean)
    ax[0].set_xticks(t_int);
    ax[0].set_xticklabels(t_int_ref/30.);
    ax[0].set_xlabel('time after stimulus onset')
    ax[0].set_ylabel('dF/F')
    ax[0].set_title('cell '+str(cell_idx)+' mean response to pref condition')

    ax[1].imshow(images[pref_scene,:,:],cmap='gray')
    ax[1].set_title('pref_im: '+str(pref_scene))
    ax[1].axis('off')

    ax[2].plot(mean_image_responses)
    ax[2].set_xlabel('image #')
    ax[2].set_ylabel('mean dF/F')
    ax[2].set_title('cell '+str(cell_specimen_id)+' normalized mean response to all conditions')

    ax[3].imshow(mean_image,cmap='gray')
    ax[3].axis('off')
    if weighted:
        ax[3].set_title('weighted mean of '+str(n_images)+' image conditions')
    else:
        ax[3].set_title('mean of '+str(n_images)+' image conditions')

    if save_dir: 
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir,str(cell_specimen_id)+'.png')
        fig.save_fig(save_path)
    # ax[4:5].plot(traces[cell_idx,:])


# In[ ]:



if __name__ == "__main__":

    drive_path = '/Volumes/Brain2016 1/'
    
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    
    manifest_path = os.path.join(drive_path,'BrainObservatory','manifest.json')
    boc = BrainObservatoryCache(manifest_file=manifest_path)
    
    #get experiment_containers meeting certain criteria
    expts = boc.get_ophys_experiments(targeted_structures=['VISl'],cre_lines=['Cux2-CreERT2'],
                                      imaging_depths=[275],session_types=['three_session_B'])
    expts = pd.DataFrame(expts)
    
    # In[6]:
    
    #get experiment session id for the first experiment container
    session_id = expts.id.values[0]
    print session_id
    
    data_set = boc.get_ophys_experiment_data(ophys_experiment_id = session_id)
    
    #get stimulus template (aka stimulus frames) for images, movies or locally sparse noise
    ns_template = data_set.get_stimulus_template(stimulus_name='natural_scenes')
    
    #stimulus specific analysis
    from allensdk.brain_observatory.natural_scenes import NaturalScenes
    ns = NaturalScenes(data_set)
    print 'done with natural scenes import'
    # In[ ]:
    
    responsive_cells = ns.peak[ns.peak.peak_dff_ns>=5].cell_specimen_id.values
    
    timestamps,traces = data_set.get_dff_traces(cell_specimen_ids=responsive_cells)
    
    frames_arr, images_arr, traces_arr = get_ns_response_arrays(traces,data_set,ns.stim_table)
    images = data_set.get_stimulus_template('natural_scenes')
    
    save_dir = os.path.join('/Users/marinag/Data/BrainObservatory/natural_scenes_plots',session_id)
    for cell_specimen_id in responsive_cells:
        plot_ns_summary(cell_specimen_id,ns,images,frames_arr,images_arr,traces_arr,thresh=0.5,weighted=False,save_dir=save_dir)
    
    
    
