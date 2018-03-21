dataset_dir = "/vol/vssp/msos/qk/Datasets/dcase2013/sed/singlesounds_stereo/singlesounds_stereo"
# os_test_data_dir = "/vol/vssp/msos/qk/Datasets/dcase2013/sed/dcase2013_event_detection_testset_OS"

workspace = "/vol/vssp/msos/qk/workspaces/weak_source_separation/dcase2013_task2"

sample_rate = 16000
n_window = 1024
n_overlap = 360     # To ensure 240 frames in an audio clip (10 s). 
n_step = n_window - n_overlap
step_in_sec = float(n_step) / sample_rate

clip_duration = 10. # Duration of an audio clip
onset_list = [0.25, 2.75, 5.25, 7.75]   # Starting time of audio events (s)
event_max_len = 2.  # Maximum length of an audio event (s)

events = ['alert', 'clearthroat', 'cough', 'doorslam', 'drawer', 'keyboard', 
          'keys', 'knock', 'laughter', 'mouse', 'pageturn', 'pendrop', 
          'phone', 'printer', 'speech', 'switch']          
          
lb_to_ix = {lb: i for i, lb in enumerate(events)}
ix_to_lb = {i: lb for i, lb in enumerate(events)}

n_folds = 4     # Number of folds
te_fold = 0     # The fold for testing, other folds for training
