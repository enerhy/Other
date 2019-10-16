# KERAS

'''----Saving the model----'''
regressor.save('Repetition_RNN_adam')


'''----------Loading----------'''
'---The part of the code before and after the model itself need to be executed---'
from keras.models import load_model
regressor = load_model('Repetition_RNN')


'''---Example of using Callbacks---'''
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Compiling the RNN
regressor.compile(optimizer='adam', loss="mean_squared_error")  # Can change loss to mean-squared-error if you require.
 
# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
 
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, shuffle=True, epochs=100,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)




