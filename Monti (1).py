#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, Reshape
from keras.preprocessing.image import ImageDataGenerator
import os
import warnings
import sys
warnings.filterwarnings("ignore", category=DeprecationWarning)
direct = sys.argv[1]

os.getcwd()


# In[2]:


#function to move files for processing before
def move_func(x):
    
    #personal directory
    for file in os.listdir('/Users/Derek/Desktop/Images'):
        if file[0] == 'n':
            final = ''
            x = file.split('-')
            for i in x[1:]:
                new = i.split('_')
                for sub in new:
                    final += sub[0].upper() + sub[1:]
                    final += '_'
            os.system('mv ' + file + ' ' + final)
            print(final)


# In[20]:


#creates generators to feed to model with a .2 validation split and flip augmentation
data = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=.2)
train_gen = data.flow_from_directory('/Users/Derek/Desktop/ImageTrain', subset = 'training', class_mode = 'categorical', target_size = (270,201))#201 for easinesss in the core$
test_gen = data.flow_from_directory('/Users/Derek/Desktop/ImageTrain', subset = 'validation', class_mode = 'categorical', target_size = (270,201))

#creates and finds all posssible classes which are dog breeds in this case
class_list = []
for key in train_gen.class_indices:
    key = key.split('_')
    final = ''
    for spl in key:
        final += spl + ' '
    final = final[:-1]
    if final[-1] == ' ':
        final = final[:-1]
    class_list.append(final)

    
#creates the model VGG16 pop off the final layer and 
from keras.applications.vgg16 import VGG16
model = Sequential()
model.add(VGG16(include_top = False, input_shape = (270,201,3)))
model.layers.pop()
for layer in model.layers:
    layer.trainable = False
model.add(Flatten())
model.add(Dense(5000, activation = 'relu'))
model.add(Dropout(0.2))

#numbers of classes are len of the class list
model.add(Dense(len(class_list), activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[21]:


model.summary()


# In[7]:


#checkpoints the model and then is able to use the best
for i in range(5):
    try: model.fit_generator(train_gen, epochs=1, validation_data=test_gen, callbacks = cp_callback, steps_per_epoch=16508/32, validation_steps=4072/32)
    except NameError: model.fit_generator(train_gen, epochs=1, validation_data=test_gen, steps_per_epoch=16508/32, validation_steps=4072/32)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath='checkpoint' + str(i),
                                                     save_weights_only=True,
                                                     verbose=1)


# In[10]:


#converts into a h5 file that can then be used in an iOS app
import coremltools
model.save('your_model.h5')
coreml_model = coremltools.converters.keras.convert('your_model.h5', image_scale = 1./255., 
                                                    class_labels=class_list, input_names='Image', 
                                                    image_input_names = "Image")

#need to convert the doubles to floats in order to allow for processing on iOS
spec = coreml_model.get_spec()
coremltools.utils.convert_double_to_float_multiarray_type(spec)
coreml_model = coremltools.models.MLModel(spec)

coreml_model.save('my_model.mlmodel')


# In[ ]:




