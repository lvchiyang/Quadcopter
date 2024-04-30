import torch  
  
if torch.cuda.is_available():  
    print('CUDA is available!  Training on GPU ...')  
    num_gpus = torch.cuda.device_count()  
    for i in range(num_gpus):  
        gpu = torch.device(f'cuda:{i}')  
        print(f'GPU {i} is {gpu} with {torch.cuda.get_device_name(i)}')  
else:  
    print('CUDA is not available.  Training on CPU ...')