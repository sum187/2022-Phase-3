import tensorflow as tf
from tensorflow.keras import Input, layers, Sequential, optimizers, losses, callbacks
import numpy as np

# Assume tar.gz file to be extracted at project directory.
def unpickle(file):
    """
    # This function returns specified number of samples from the data (with same number of each class)

    Args:
        func (callable): derivative function that returns an ndarray of derivative values.
        labels (ndarray): initial condition(s) for dependent variable(s).
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
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def get_sample(labels,images,n):
    """
    # This function returns specified number of random samples from the data (with same number of each class)

    Args:
        labels (ndarray): labels indicating corresponding images
        images (ndarray): images in any shape 
        n (float): desired number of samples

    Returns:
        data_labels (ndarray): sample of label indexes for corresponding corresponding images
        data_images (ndarray): sample images in given shape
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

def load_image_data(train='off'):
    """
    # This function returns testing labels and images by default. Additionally, returns training labels and images 
      when train is set to 'on' 

    Args:
        train (string): set to 'on' to return training labels and images. This value is 'off' by default

    Returns:
        label_names (list): name of labels
        test_labels (ndarray): testing labels for fitting image detection model
        test_images (ndarray): testing images          ""   ""
        train_labels (ndarray): training labels        ""   ""
        train_images (ndarray): training images        ""   "" 
    """
    assert train=='off' or train=='on','{} is invalid input. train can only take \'off\' or \'on\' as an input'.format(train)
    folder='cifar-10-batches\\' # folder name that contains cifar-10 files

    # training,test and name variables 
    train_batch1=unpickle(folder+'data_batch_1')
    test=unpickle(folder+'test_batch')
    label_names=unpickle(folder+'batches.meta')['label_names']

    # store training batches (1 to 5) in one variable
    # reshape rgb data 2d (nx3072) array to 4d (nx32x32x3) for both test and training data
    train_batches=[]
    for num in range(1,6):
        # reshape to have colour channel at last dimension
        dict=unpickle(folder+'data_batch_'+str(num))
        dict['data']=np.reshape(dict['data'],(10000,3,32,32))   
        dict['data']=np.transpose(dict['data'],[0,2,3,1])

        # remove keys that is unnecessary in this analysis
        dict.pop('batch_label')
        dict.pop('filenames')
        train_batches.append(dict)
        
    # remove keys that is unnecessary in this analysis
    test.pop('batch_label')
    test.pop('filenames')
    test['data']=np.reshape(test['data'],(10000,3,32,32))  
    test['data']=np.transpose(test['data'],[0,2,3,1])

    # split training batches to label and image batches
    train_images_batches = [train['data'] for train in train_batches]
    train_labels_batches = np.array([train['labels'] for train in train_batches])
    
    # combine batches to one batch
    train_labels_combined=np.reshape(train_labels_batches,(50000))
    train_images_combined=np.reshape(train_images_batches,(50000,32,32,3))

    # get 10000 training and 2000 test samples
    train_labels,train_images=get_sample(train_labels_combined,train_images_combined,40000)
    test_labels,test_images=get_sample(np.array(test['labels']),test['data'],8000)

    if train == 'on':
        return label_names,test_labels,test_images,train_labels,train_images
    else:
        return label_names,test_labels,test_images


def model_builder(hp):
  """
    # This function return model that is able to detect chosen label in the images

    Args:
        hp (???): automatically generated by keras Hyperband Tuner

    Returns:
        model (???): image detection model 
  """
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
    """
    # This function return summary to hyperparameter tuning

    Args:

    Returns:
        summary_dict (dictionary): range of parameters used duing hyperparameter tuning and corresponding validation accuracy 
    """
    # read the text file and remove \n, space and : string on each line
    with open('my_dir/intro_to_kt/summary.txt') as f:
        lines = [line.replace('\n','').replace(' ','').split(':') for line in f.readlines()] 

    # initialise summary dictionary  
    summary_dict={'units':[],'dropout':[],'learning_rate':[],'tuner/epochs':[],'Score':[]}
    
    # get parameters to correcting attribute
    for line in lines:
        if len(line)==2 and list(summary_dict.keys()).count(line[0])==1:
            if line[0]=='Score':
               summary_dict[line[0]].append(float(line[1]))
            else:
               summary_dict[line[0]].append(line[1])  
    summary_dict['combined']=[]

    # store combined form of parameter values only(exclude Score value)
    for i in range(28):
        summary_dict['combined'].append('{}\n{}\n{}\n{}'.format(
            summary_dict['units'][i],summary_dict['dropout'][i],summary_dict['learning_rate'][i],summary_dict['tuner/epochs'][i]
    ))           

    return summary_dict        

def predict(image,label,value='off'):
    """
    # This function calculates the probability that there is chosen label in the images given. Additionally, this function
      can return the label that is most likely in the image and its corresponding probability 

    Args:
        image (ndarray): image or images to be examined
        label (list): label or labels of given image
        value (string): set to 'on' to return more results

    Returns:
        pred_label (ndarray): 1 if chosen label is detected, 0 otherwise
        pred_percentage (ndarray): probability/probabilities that there is chosen label in the image(s)
        bestScoreLabelName (ndarray): label that is most likely in the image (only return for on image input)
        best_pred_percentage (ndarray): corresponding probability to above (only return for on image input)
    """
    # raise error
    assert value=='off' or value=='on', 'value must be either \'on\' or \'off\' but {} was given'.format(value)
    
    # load model
    model = tf.keras.models.load_model('output\my_model')
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
        best_pred_percentage=100 * score.numpy()[best_pred_label]
        bestScoreLabelName=names[best_pred_label]
        
        if value=='on':
            return pred_label,pred_percentage,bestScoreLabelName,best_pred_percentage
        elif value=='off':
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
