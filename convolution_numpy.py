import numpy as np
from math import ceil
import struct
import cv2

#defining the required relu and log activation functions
def relu(x):
    return (np.maximum(x,0))
def log(x):
    return (1/(1 + np.exp(-1*x)))
def d_log(x):
    return (log(x) * ( 1 - log(x)))

#defining max pooling function
def maxpool(m):
    out_h = m.shape[0]
    out_w = m.shape[1]
    pool = []

    for i in range(0,out_h,3):
        val = []
        for j in range(0,out_h,3):
            m1 = np.amax(m[i][j:j+3])
            m2 = np.amax(m[i+1][j:j+3])
            m3 = np.amax(m[i+2][j:j+3])
            val.append(max(m1,m2,m3))
        pool.append(val)

    return(pool)


#defining the convolution function
def convolution2d(conv_input, conv_kernel, bias=[1,0,1,0], strides=(2, 2), padding = "same"):
    
    input_w, input_h = conv_input.shape[0], conv_input.shape[1]      # input_width and input_height
    kernel_w, kernel_h = conv_kernel.shape[0], conv_kernel.shape[1]  # kernel_width and kernel_height
    output_depth = conv_kernel.shape[2]
    
    if(padding == "same"):
    
        output_height = int(ceil(float(input_h) / float(strides[0])))
        output_width = int(ceil(float(input_w) / float(strides[1])))
    
        # Calculate the number of zeros which are needed to add as padding
        pad_along_height = max((output_height - 1) * strides[0] + kernel_h - input_h, 0)
        pad_along_width = max((output_width - 1) * strides[1] + kernel_w - input_w, 0)
        pad_top = pad_along_height 
        pad_bottom = pad_along_height - pad_top     
        pad_left = pad_along_width     
        pad_right = pad_along_width - pad_left      
        output = np.zeros((output_height, output_width, output_depth))  
       
        image_padded = np.zeros((conv_input.shape[0] + pad_along_height, conv_input.shape[1] + pad_along_width))
    
        #image_padded[pad_top:-pad_bottom, pad_left:-pad_right] = conv_input
        image_padded[pad_top:, pad_left:] = conv_input
    
        for ch in range(output_depth):
            for x in range(output_width):  # Loop over every pixel of the output
                for y in range(output_height):
                    # element-wise multiplication of the kernel and the image
                    output[y, x, ch] = (conv_kernel[...,ch] * image_padded[y * strides[0]:y * strides[0] + kernel_h, x * strides[1]:x * strides[1] + kernel_w]).sum() + bias[ch] 
    
    
    elif (padding == "valid"):
        output_height = int(ceil(float(input_h - kernel_h + 1) / float(strides[0])))
        output_width = int(ceil(float(input_w - kernel_w + 1) / float(strides[1])))
        output = np.zeros((output_height, output_width, output_depth))  # convolution output
        
        for ch in range(output_depth):
            for x in range(output_width):  # Loop over every pixel of the output
                for y in range(output_height):
                    # element-wise multiplication of the kernel and the image
                    output[y, x, ch] = (conv_kernel[...,ch] * conv_input[y * strides[0]:y * strides[0] + kernel_h, x * strides[1]:x * strides[1] + kernel_w]).sum() + bias[ch]
    
    return (output)


# ---- Cost for testing ------
def testing(X,Y):
    
    cost_after_train = 0
    final_out = []
    for i in range(len(X)):
    
        layer_1 = convolution2d(X[i],w1)
        layer_1_act = relu(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act,-1),axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = log(layer_2) 
        result = np.where(layer_2_act == np.amax(layer_2_act))
        result = np.array(result[1][0])
    
        #cost = np.square(layer_2_act- Y[i]).sum() * 0.5
        cost = np.square(result - Y[i]).sum() * 0.5
        cost_after_train = cost_after_train + cost
        
        #final_out = np.append(final_out,layer_2_act)
        final_out = np.append(final_out,result)
        
    return(cost_after_train, final_out)


#importing the mnist dataset
X_train = 'path of mnist dataset training data'
y_train = 'path of mnist dataset testing data'

#reading the class data y
with open(y_train, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)  
        
#reading the image data x
with open(X_train, 'rb') as fim:
    magic, num, rows, cols = struct.unpack(">IIII", fim.read(16))
    img = np.fromfile(fim, dtype=np.uint8)
    
    image_mat = np.empty([60000,32,32])
    i = 0
    
    #resizing mnist images from 28x28 to 32x32
    for j in range(0,47040000,784):
        img1 = img[j:j+784]
        res = cv2.resize(img1, dsize=(32, 32), interpolation=cv2.INTER_CUBIC) 
        image_mat[i] = res
        i+=1
    
    #defining the feature vector X and output class vector Y for training
    Y = lbl[:50000]
    X = image_mat[:50000]
    
    #defining the feature vector X and output class vector Y for testing
    Y_test = lbl[50000:]
    X_test = image_mat[50000:]
    
    print(X.shape)
    print(Y.shape)


# Declare Weights
w1 = np.random.randn(3,3,4) 
w2 = np.random.randn(1024,10)

# Declare hyper Parameters
num_epoch = 10
learning_rate = 0.7

cost_before_train = 0
cost_after_train = 0
final_out, start_out = np.array([[]]), np.array([[]])


# ---- Cost before training model ------
for i in range(len(X)):
    
    layer_1 = convolution2d(X[i],w1)
    #print("layer1",layer_1.shape)
    
    layer_1_act = relu(layer_1)
    #print(layer_1_act.shape)
    
    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act,-1),axis=0)
    #print(layer_1_act_vec.shape)
    
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)   
    result = np.where(layer_2_act == np.amax(layer_2_act))
    result = np.array(result[0])
    
    #cost = np.square(layer_2_act- Y[i]).sum() * 0.5
    cost = np.square(result - Y[i]).sum() * 0.5
    cost_before_train = cost_before_train + cost
    
    #start_out = np.append(start_out,layer_2_act)
    start_out = np.append(start_out,result)

# training the model
for iter in range(num_epoch):
    
    for i in range(len(X)):
        
        #forward pass
    
        layer_1 = convolution2d(X[i],w1)    #input layer and convolution layer
        layer_1_act = relu(layer_1)         #relu function 

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act,-1),axis=0)
        layer_2 = layer_1_act_vec.dot(w2)   # relu and fully connected layer
        layer_2_act = log(layer_2)          
        result = np.where(layer_2_act == np.amax(layer_2_act))  #fully connected layer and output layer
        result = np.array(result[1][0])
    
        #cost = np.square(layer_2_act- Y[i]).sum() * 0.5
        cost = np.square(result - Y[i]).sum() * 0.5          #mean squared error value 
        
        print("Current iter : ",iter , " Current train: ",i, " Current cost: ",cost)

        #backward pass with error correction and weight updation
        
        #grad_2_part_1 = layer_2_act- Y[i]
        grad_2_part_1 = result - Y[i]
        
        grad_2_part_2 = d_log(layer_2)
        grad_2_part_3 = layer_1_act_vec
        grad_2 =  grad_2_part_3.T.dot(grad_2_part_1*grad_2_part_2)      

        grad_1_part_1 = (grad_2_part_1*grad_2_part_2).dot(w2.T)
        grad_1_part_2 = relu(layer_1)
        grad_1_part_3 = X[i]

        grad_1_part_1_reshape = np.reshape(grad_1_part_1,(16,16,4))
        grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
        
        grad_1 = np.rot90(convolution2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2), padding = "valid"),2)
        grad_1_pool = np.empty((3,3,4))
        
        for i in range(grad_1.shape[2]):
            grad_1_pool[...,i] = np.array(maxpool(grad_1[...,i])) 
        
        
        w2 = w2 - grad_2 * learning_rate
        w1 = w1 - grad_1_pool * learning_rate
        
#calling testing function
cost_after_train, final_out = testing(X_test,Y_test)

# ----- Print Results ---
print("\nW1 weights :",w1, "\n\nw2 weights :", w2)
print("----------------")
print("Cost before Training: ",cost_before_train)
print("Cost after Training: ",cost_after_train)
print("----------------")
print("Start Output : ", start_out)
print("Final Output : ", final_out)
print("Actual output  : ", Y.T)

