import scipy
import numpy as np

# Convert right ips to GPS times and merge the two windows
# def get_triggers(preds_0, preds_5, width, threshold, truncation=0, window_shift=2048):#, threshold=threshold, width=[width, 3000], mean=0.95):
    
#     triggers_0, triggers_5 = {}, {} 
#     key = 'detection'
        
#     peak_0, left_0, right_0 = find_peaks(preds_0, threshold=threshold, width=[width, 3000], mean=0.95)
#     peak_5, left_5, right_5 = find_peaks(preds_5, threshold=threshold, width=[width, 3000], mean=0.95)

#     triggers_0[key] = [(x + truncation)/4096 for x in right_0]  
#     triggers_5[key] = [(x + truncation + window_shift)/4096 for x in right_5]
    
#     # Merge the two windows
#     triggers = merge_windows(triggers_0, triggers_5)
    
#     return triggers

def get_triggers(preds_0, preds_5, width, threshold, truncation=0, window_shift=2048):
    
    print("new get_trigger:", flush=True)
    triggers_0, triggers_5 = {}, {}
    key = 'detection'
    
    dynamic_mean = max(0, min(0.95, threshold - 0.05))

    peak_0, left_0, right_0 = find_peaks(preds_0, threshold=threshold, width=[width, 3000], mean=dynamic_mean)
    peak_5, left_5, right_5 = find_peaks(preds_5, threshold=threshold, width=[width, 3000], mean=dynamic_mean)

    triggers_0[key] = [(x + truncation) / 4096 for x in right_0]
    triggers_5[key] = [(x + truncation + window_shift) / 4096 for x in right_5]

    # Merge the two windows
    triggers = merge_windows(triggers_0, triggers_5)

    return triggers

# func. to find peaks
def find_peaks(preds, threshold = 0.9, width = [1500,3000], mean = 0.9):
    '''
    preds: 1D numpy array of sigmoid output from the NN
    '''
    test_p = preds
    
    peaks, properties =  scipy.signal.find_peaks(test_p, height=threshold, width = width, distance = 4096*1 )

    left = properties['left_ips']
    right = properties['right_ips']

    f_left = []
    f_right = []
    for i in range(len(left)):
        sliced = test_p[int(left[i]):int(right[i])] 
        # if (np.mean(sliced>mean)>mean):
        if (np.mean(sliced>mean)>0.5):
            f_left.append(int(left[i]))
            f_right.append(int(right[i]))
            
    return peaks, f_left, f_right

# func. to merge windows
def merge_windows(triggers_0, triggers_5):

    triggers = {}
    
    for key in triggers_0.keys():

        right_0 = triggers_0[key]
        right_5 = triggers_5[key]

        combined = right_0.copy()
        
        for r_5 in right_5:
            keep = True
            for r_0 in right_0:
#                 if abs(r_5 - r_0) < 1:
                if abs(r_5 - r_0) == 1/2:
                    keep = False
            if keep:
                combined.append(r_5)
        
        triggers[key] = combined
    
    return triggers


