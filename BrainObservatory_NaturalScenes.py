
# coding: utf-8

# In[1]:

import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:

drive_path = '/Volumes/Brain2016 1/'


# In[3]:

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_path = os.path.join(drive_path,'BrainObservatory','manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_path)


# ### get metrics for all cell_specimens in the dataset

# In[4]:

#get cell metrics dataframe
cell_specimen_df = pd.DataFrame(boc.get_cell_specimens(ids=None,experiment_container_ids=None))
cell_specimen_df.head()


# ### get experiment containers meeting certain criteria

# In[5]:

#get experiment_containers meeting certain criteria
expts = boc.get_ophys_experiments(targeted_structures=['VISl'],cre_lines=['Cux2-CreERT2'],
                                  imaging_depths=[275],session_types=['three_session_B'])
expts = pd.DataFrame(expts)
print len(expts),'experiments meet these criteria'
expts.head()


# In[68]:

all_expts = boc.get_ophys_experiments()
all_expts = pd.DataFrame(all_expts)
all_expts.head()


# In[66]:

expt_container_id = 511510753


# In[70]:

# get session id for known experiment container & session type
all_expts[(all_expts.experiment_container_id==expt_container_id)&(all_expts.session_type=='three_session_C')].id.values[0]


# In[6]:

#get experiment session id for the first experiment container
session_id = expts.id.values[0]
print session_id


# ### get data_set for one experiment session

# In[7]:

#get data for a single experiment session
data_set = boc.get_ophys_experiment_data(ophys_experiment_id = session_id)


# In[8]:

#what can you do with a data_set?
help(data_set)


# In[9]:

#get all cell_specimen_ids in this session
cell_specimen_ids = data_set.get_cell_specimen_ids()
print cell_specimen_ids[:10]


# In[10]:

#get indices for cell_specimen_ids in this experiment
#use indices to index into traces & sweep_response
cell_specimen_idx = data_set.get_cell_specimen_indices(cell_specimen_ids)
print cell_specimen_idx[:10]


# In[11]:

#what stimuli are in this session?
data_set.list_stimuli()


# In[12]:

#get stimulus template (aka stimulus frames) for images, movies or locally sparse noise
ns_template = data_set.get_stimulus_template(stimulus_name='natural_scenes')
ns_template.shape


# In[13]:

#get stimulus table
stim_table = data_set.get_stimulus_table('natural_scenes')
stim_table.head()


# ### stimulus specific analysis - natural_scenes

# In[14]:

#stimulus specific analysis
from allensdk.brain_observatory.natural_scenes import NaturalScenes
ns = NaturalScenes(data_set)


# In[15]:

#what can get from the NaturalScenes object? 
#(hint: most things are already computed for you - do ns. tab complete to check)
help(ns)


# In[16]:

#duration of each sweep in imaging frames
ns.sweeplength


# In[17]:

#duration of window before and after stimulus included in sweep response
ns.interlength


# In[18]:

#sweep_response table - dataframe of dF/F traces for every sweep (rows) for all cells (columns)
#trace includes interlength+sweeplength+interlength
sweep_response = ns.sweep_response


# In[19]:

#check length of trace for one sweep from one cell in sweep_response table
#value should equal sweeplength+interlength*2
sweep_response.iloc[0][0].shape


# In[20]:

#mean_sweep_response - mean values of traces in sweep_response during stimulus window
#sweep (rows) by cells (columns)
mean_sweep_response = ns.mean_sweep_response 
mean_sweep_response.head()


# In[21]:

#get mean response across conditions (conditions, cells, 3)
#for natural scenes, array is (#scenes, #cells, 3)
#last dimension is, for each condition: mean response, sem, p-value 
condition_response = ns.response
condition_response.shape


# In[72]:

#mean response for all image conditions for cell 
cell = 0
mean_image_responses = condition_response[:,cell,0] #[image,cell,mean]


# In[75]:

plt.plot(mean_image_responses)
plt.xlabel('image #')
plt.ylabel('mean dF/F')


# ### find pref image & plot mean response to that image

# In[76]:

ns.peak


# In[23]:

cell = ns.peak[ns.peak.peak_dff_ns>=3].cell_specimen_id.values[0]
cell


# In[24]:

pref_scene = ns.peak[ns.peak.cell_specimen_id==cell].scene_ns.values[0]
pref_scene


# In[78]:

stim_table


# In[25]:

pref_scene_sweeps = ns.stim_table[ns.stim_table.frame==pref_scene].index.values
pref_scene_sweeps


# In[27]:

#if the mean of the mean response to the preferred condition is > 5%
responsive_cells = ns.peak[ns.peak.peak_dff_ns>=5].cell_specimen_id.values
len(responsive_cells)


# In[28]:

cell = responsive_cells[1]


# In[81]:

responsive_cells


# In[87]:

ns.sweep_response


# In[88]:

plt.plot(condition_mean)


# In[29]:

cell = responsive_cells[1]
cell_idx = np.where(ns.cell_id==cell)[0][0]
pref_scene = ns.peak[ns.peak.cell_specimen_id==cell].scene_ns.values[0]
pref_scene_sweeps = ns.stim_table[ns.stim_table.frame==pref_scene].index.values

condition_mean = ns.sweep_response[str(cell_idx)].iloc[pref_scene_sweeps].mean()
frames = ns.sweeplength+ns.interlength*2
t = np.arange(0,frames)
t_int = np.arange(0,frames,6)
t_int_ref = t_int - ns.interlength 

fig,ax = plt.subplots()
ax.plot(t,condition_mean)
ax.set_xticks(t_int);
ax.set_xticklabels(t_int_ref);
# ax.set_xlabel('time after stimulus onset')
# ax.set_ylabel('dF/F')
ax.set_title('cell: '+str(cell)+', pref_im: '+str(pref_scene))


# ### make it a function & plot for all 'responsive' cells

# In[30]:

def plot_pref_condition_response(ns,cell_specimen_id):
    pref_scene = ns.peak[ns.peak.cell_specimen_id==cell_specimen_id].scene_ns.values[0]
    pref_scene_sweeps = ns.stim_table[ns.stim_table.frame==pref_scene].index.values
#     cell_idx = data_set.get_cell_specimen_indices([cell])[0]
    cell_idx = np.where(ns.cell_id==cell_specimen_id)[0][0]

    condition_mean = ns.sweep_response[str(cell_idx)].iloc[pref_scene_sweeps].mean()
    frames = ns.sweeplength+ns.interlength*2
    t = np.arange(0,frames)
    t_int = np.arange(0,frames,6)
    t_int_ref = t_int - ns.interlength 

    fig,ax = plt.subplots()
    ax.plot(t,condition_mean)
    ax.set_xticks(t_int);
    ax.set_xticklabels(t_int_ref/30.);
    ax.set_xlabel('time after stimulus onset')
    ax.set_ylabel('dF/F')
    ax.set_title('cell: '+str(cell)+', pref_im: '+str(pref_scene))


# In[39]:

images = data_set.get_stimulus_template('natural_scenes')


# In[89]:

images.shape


# In[46]:

def plot_pref_condition_response(ns,cell_specimen_id,images):
    pref_scene = ns.peak[ns.peak.cell_specimen_id==cell_specimen_id].scene_ns.values[0]
    pref_scene_sweeps = ns.stim_table[ns.stim_table.frame==pref_scene].index.values
#     cell_idx = data_set.get_cell_specimen_indices([cell])[0]
    cell_idx = np.where(ns.cell_id==cell_specimen_id)[0][0]

    condition_mean = ns.sweep_response[str(cell_idx)].iloc[pref_scene_sweeps].mean()
    frames = ns.sweeplength+ns.interlength*2
    t = np.arange(0,frames)
    t_int = np.arange(0,frames,6)
    t_int_ref = t_int - ns.interlength 

    fig,ax = plt.subplots(1,2,figsize=(12,4))
    ax = ax.ravel()
    ax[0].plot(t,condition_mean)
    ax[0].set_xticks(t_int);
    ax[0].set_xticklabels(t_int_ref/30.);
    ax[0].set_xlabel('time after stimulus onset')
    ax[0].set_ylabel('dF/F')
    ax[0].set_title('cell: '+str(cell)+', pref_im: '+str(pref_scene))
    
    ax[1].imshow(images[pref_scene,:,:],cmap='gray')


# In[47]:

plot_pref_condition_response(ns,responsive_cells[0],images)


# In[48]:

for cell in responsive_cells: 
    plot_pref_condition_response(ns,cell,images)


# In[49]:

stim_table.head()


# ### get traces & image sequence for natural scenes portion of session 

# In[51]:

timestamps,traces = data_set.get_dff_traces(cell_specimen_ids=responsive_cells)


# In[52]:

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


# In[53]:

traces_arr = np.empty((traces.shape[0],frames_arr.shape[0]))
for t in range(traces.shape[0]): 
    trace = np.empty(0)
    for sweep in range(len(stim_table)):
        start = stim_table.iloc[sweep].start
        end = stim_table.iloc[sweep].end
        tmp = traces[t,start:end]
        trace = np.hstack((trace,tmp))
    traces_arr[t,:] = trace


# In[99]:

plt.plot(traces_arr[1,:])


# In[102]:

thresh_inds = np.where(traces_arr[cell_num,:]>=0.3)[0]
thresh_inds = thresh_inds - 6


# In[374]:

thresh_vals = traces_arr[cell_num][thresh_inds]


# In[375]:

thresh_images = images_arr[thresh_inds]


# In[376]:

img_stack = np.empty((thresh_images.shape[0],images[0,:,:].shape[0],images[0,:,:].shape[1]))
for i,img in enumerate(thresh_images):
    img_stack[i,:,:] = images[img,:,:]


# In[ ]:

mean_image = np.mean(img_stack,axis=0)


# In[ ]:

plt.imshow(mean_image,cmap='gray')
plt.colorbar()


# In[377]:

weighted_mean_image = np.average(img_stack,axis=0,weights=thresh_vals)


# In[378]:

plt.imshow(weighted_mean_image,cmap='gray')
plt.colorbar()


# ### put it in a function and plot

# In[147]:

def plot_mean_image(cell_specimen_id,ns,images,images_arr,traces_arr,thresh=0.3,weighted=False):
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
        mean_image = np.average(img_stack,axis=0,weights=thresh_vals)
    else: 
        mean_image = np.mean(img_stack,axis=0)
    fig,ax=plt.subplots()
    ax.imshow(mean_image,cmap='gray')
    if weighted:
        ax.set_title('cell '+str(cell_specimen_id)+', weighted mean of '+str(n_images)+' images')
    else:
        ax.set_title('cell '+str(cell_specimen_id)+', mean of '+str(n_images)+' images')


# In[149]:

plot_mean_image(responsive_cells[9],ns,images,images_arr,traces_arr,thresh=0.3,weighted=True)


# In[150]:

plot_mean_image(responsive_cells[9],ns,images,images_arr,traces_arr,thresh=0.3,weighted=False)


# In[ ]:




# In[ ]:




# ### batch processing 

# In[156]:

#processing on all traces
def get_ns_response_arrays(data_set):
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


# In[ ]:

frames_arr, images_arr, traces_arr = get_ns_response_arrays(data_set)
images = data_set.get_stimulus_template('natural_images')


# In[192]:

def plot_ns_summary(cell_specimen_id,ns,images,frames_arr,images_arr,traces_arr,thresh=0.5,weighted=False): 
        
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

    # ax[4:5].plot(traces[cell_idx,:])


# In[193]:

plot_ns_summary(responsive_cells[1],ns,images,frames_arr,images_arr,traces_arr,thresh=0.5,weighted=False)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



