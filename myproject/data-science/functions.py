import tensorflow as tf
from tensorflow.keras import Input, layers, Sequential, optimizers, losses, callbacks
import numpy as np

# Assume tar.gz file to be extracted at project directory.
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# This function returns specified number of samples from the data (with same number of each class)
def get_sample(labels,images,n):
    """
    Compute solution(s) to ODE(s) using any explicit RK method with fixed step size.

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        y0 (ndarray): initial condition(s) for dependent variable(s).
        t0 (float): start value of independent variable.
        t1 (float):	stop value of independent variable.
        h (float): fixed step size along independent variable.
        alpha (ndarray): weights in the Butcher tableau.
        beta (ndarray): nodes in the Butcher tableau.
        gamma (ndarray): RK matrix in the Butcher tableau.
        *args : optional system parameters to pass to derivative function.

    Returns:
        t (ndarray): independent variable values at which dependent variable(s) calculated.
        y (ndarray): dependent variable(s) solved at t values.
    """
    # set sedd for consistency
    np.random.seed(123)
    data_images=[]
    data_labels=[]
    for i in range(10):
        idx=np.random.choice(np.where(labels==i)[0],int(n/10),replace=False)
        idx.sort()
        data_images.append(np.array(images)[idx])
        data_labels.append(np.array(labels)[idx])
    data_images=np.reshape(data_images,(n,32,32,3))    
    data_labels=np.reshape(data_labels,(n))
    # shufftle the data
    np.random.seed(123)  
    np.random.shuffle(data_labels)  
    np.random.seed(123)  
    np.random.shuffle(data_images)

    return data_labels,data_images

def model_builder(hp):
  # Choose an optimal unit value between 64-256 dropout value 0.05-0.15
  hp_units = hp.Int('units', min_value=64, max_value=256, step=32)
  hp_dropout = hp.Choice('dropout', values=[0.05, 0.1, 0.15])

  model = tf.keras.Sequential([
    # keras preprocessing layer for random flip and rotation of image
    tf.keras.layers.RandomFlip("horizontal", input_shape=(32,32,3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    # resize image and scale RBG value to 0-1
    tf.keras.layers.Resizing(32, 32),
    tf.keras.layers.Rescaling(1./255),

    # convolution and pooling layer for image_height, image_width and color_channels
    tf.keras.layers.Conv2D(16,3,padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32,3,padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64,3,padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    #dropping out the output units randomly from the applied layer with some proportion
    tf.keras.layers.Dropout(hp_dropout),
    tf.keras.layers.Flatten(),

    # Tune the number of units in the first Dense layer
    tf.keras.layers.Dense(units=hp_units, activation='relu'),
    tf.keras.layers.Dense(10),])
  
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

def load_hyperParamSummary():
    # read the text file and remove \n, space and : string on each line
    with open('my_dir/intro_to_kt/summary.txt') as f:
        lines = [line.replace('\n','').replace(' ','').split(':') for line in f.readlines()] 

    # initialise summary dictionary  
    summary_dict={'units':[],'dropout':[],'learning_rate':[],'tuner/epochs':[],'Score':[]}
    
    # get parameters to correcting attribute
    for line in lines:
        if len(line)==2 and list(summary_dict.keys()).count(line[0])==1:
            summary_dict[line[0]].append(line[1])

    return summary_dict        

def predict(model,image,label):
    names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    # convert to np array
    image=np.array(image)
    
    # if there is one image
    if len(np.shape(image))==3:
        # resize image before passing it through the model
        resize_and_rescale = tf.keras.Sequential([layers.Resizing(32, 32),])
        augmented_image = resize_and_rescale(tf.expand_dims(image, 0))
        
        # make a prediction
        predictions=model.predict(tf.expand_dims(augmented_image[0], 0),verbose=0)

        # calculate probability
        score = tf.nn.softmax(predictions[0])

        # find index of label of interest and label that gives best prediction
        idx=names.index(label)
        best_pred_label=np.argmax(score)

        pred_label=int(best_pred_label==idx)
        pred_percentage=100 * score.numpy()[idx]

        return pred_label,pred_percentage

    # for multiple images    
    else:
        # initialise
        pred_label=[]
        pred_percentage=[]

        # resize image before passing it through the model
        resize_and_rescale = tf.keras.Sequential([layers.Resizing(32, 32),])
        augmented_image = resize_and_rescale(image)
        
        # make a prediction
        predictions=model.predict(augmented_image,verbose=0)
        
        for p in predictions:
            # calculate probability on each class
            score = tf.nn.softmax(p)

            # find index of label of interest and label that gives best prediction
            idx=names.index(label)
            best_pred_label=np.argmax(score)
            
            # append the result
            pred_label.append(int(best_pred_label==idx)) # check best predicted class matches to chosen class
            pred_percentage.append(100 * score.numpy()[idx]) # confidence percent for chosen class

        return pred_label,pred_percentage
