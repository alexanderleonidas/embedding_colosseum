
import matplotlib.pyplot as plt
import numpy as np
import collections
from qiskit import transpile,execute
from qiskit.tools.monitor import job_monitor


# Define a function named 'calc_difference' that takes an 'image' as input
def calc_difference(image,input_image):
    # Initialize a variable 'summe' to keep track of the sum of differences
    summe = 0
    
    # Iterate through the range of the length of the 'image'
    for i in range(len(image)):
        # Calculate the absolute difference between elements of 'image' and 'input_image'
        diff = abs(image[i] - input_image[i])
        
        # Add the absolute difference to the 'summe' variable
        summe = summe + diff
    
    # Calculate the percentage difference using the obtained 'summe' value
    # Normalize the sum by dividing by the length of the 'image' and then by 2.55
    prozent = summe / len(image) / 2.55
    
    # Round the 'prozent' value to 2 decimal places
    prozent = prozent.round(2)
    
    # Return the calculated percentage difference
    return prozent

def execute_sim(circs, numOfShots, backend, sim_bool):
    
    '''Execute the encoding and append the result to a list'''
    
    print('Used backend: ', backend)
    #job = backend.run(transpile(circs, backend), shots=numOfShots)
    job = execute(circs, backend, shots=numOfShots)
    print('Job_ID: ', job.job_id())
    result = job.result()
    
    if sim_bool == False:
        print(job_monitor(job))
        # timecalculation(job)
    else:
        execution_time = result.time_taken # Calculate time
        print('Execution time: ', str(round(execution_time,2))+'s')
    
    counts_job = result.get_counts()
    
    return counts_job

def visualize_images_grid(num_pixels,num_rows, input_images,num_shots):
    num_cols = len(num_pixels)
    if num_cols == 1:
        pixels = num_pixels[0]
        plt.imshow(input_images[0].reshape(pixels, pixels), cmap='gray',vmin=0, vmax=255)
        plt.title(f"{num_shots} shots for Pixel Size: {pixels}x{pixels}")
    else:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
        fig.suptitle("Images Grid", fontsize=16)
    
        for j, (pixels, images) in enumerate(zip(num_pixels, input_images)):
            axes[j].imshow(images.reshape(pixels, pixels), cmap='gray',vmin=0, vmax=255)
            axes[j].set_xticks(np.arange(0, pixels, 1))
            axes[j].set_yticks(np.arange(0, pixels, 1))
            
            axes[j].set_title(f"{num_shots} shots for Pixel Size: {pixels}x{pixels}")

        plt.show()

def select_labels_to_keep(a, b,label1,label2):
    keep = (b == label1) | (b == label2)
    a, b = a[keep], b[keep]
    b = b == label1
    return a,b

def remove_contradicting_images(xs, ys):
    # In the end "mapping" will hold the number of unique images
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Establish the labels for each image
    for x,y in zip(xs,ys):
      orig_x[tuple(x.flatten())] = x
      mapping[tuple(x.flatten())].add(y)
    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Images that match multiple labels are discarded
          pass
    
    # Number of unique images of digit 3
    unique_images_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    
    # Number of unique images of digit 6
    unique_images_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    
    return np.array(new_x), np.array(new_y)