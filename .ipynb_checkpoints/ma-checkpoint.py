def select_cell_trace(lab_coord, trace_list, mode):
    #sub select cells and traces according to brain region
    import numpy as np
    if mode == 'inbrain':
        locs = np.where(lab_coord[:,4] != 'nan')
        
    else: 
        locs = np.where(lab_coord[:,4] == mode)
            
    sub_coord = lab_coord[locs][:,:4]
                                      
    sub_trace_list = list(range(len(trace_list)))
    for i in range(len(sub_trace_list)):
        if lab_coord.shape[0] != trace_list[i].shape[0]:
            print('Trace and coordinate data not same shape')
            break 
        sub_trace_list[i] = trace_list[i][locs]
    return(sub_coord, sub_trace_list)


def dim(data):
    import numpy as np
    cov = np.cov(data)
    eig = np.linalg.eigvals(cov)
    output = (((np.sum(eig))**2)/ (np.sum(eig**2))) / eig.shape[0]
    return(output)

def spike_stats(bind, dff):
    # calculate spike statistics - number of spikes per cell, mean amplitude per cell, number of continuous transients per cell, mean transient duration per cell
    import numpy as np
    import more_itertools as mit
    if bind.shape[0] != dff.shape[0]:
        print('Data not the same shape')
        return()
    spikes = np.sum(bind, axis = 1)
    mean_amp = np.mean(dff, axis = 1)
    n_trans = np.zeros((bind.shape[0]))
    corr = np.corrcoef(bind)
    for i in range(bind.shape[0]):
        si = np.where(bind[i] == 1)[0]
        n_trans[i] = len([list(group) for group in mit.consecutive_groups(si)])
    mean_transdur = spikes / n_trans
    dimen = dim(bind)
    return(spikes, mean_amp, n_trans, mean_transdur, corr, dimen)