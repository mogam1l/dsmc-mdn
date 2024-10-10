import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

print("Tensorflow Version: ", tf.__version__)
print("Tensorflow Probability Version: ", tfp.__version__)

data = pd.read_csv('collision_dataset.txt')
data.head()

def dscatter(x,y):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x,y,c=z, s=10)



plt.figure(figsize=[10,3])

plt.subplot(1,3,1)
plt.xlabel("Etr (K)")
plt.ylabel("Etr' (K)")
dscatter(data['Etr'],data['Etrp'])


plt.subplot(1,3,2)
plt.xlabel("Er,A (K)")
plt.ylabel("Er,A' (K)")
dscatter(data['Er1'],data['Er1p'])


plt.subplot(1,3,3)
plt.xlabel("Er,B (K)")
plt.ylabel("Er,B' (K)")
dscatter(data['Er2'],data['Er2p'])
plt.tight_layout()
plt.show()


train, test = train_test_split(data, test_size=0.3)



# Variables within the training set
Ec_train = np.array(train[['Etr']])+np.array(train[['Er1']])+np.array(train[['Er2']])
Ecp_train = np.array(train[['Etrp']])+np.array(train[['Er1p']])+np.array(train[['Er2p']])
eps_t_train = np.array(train[['Etr']])/Ec_train
eps_tp_train = np.array(train[['Etrp']])/Ecp_train
eps_r1_train = np.array(train[['Er1']])/(np.array(train[['Er1']])+np.array(train[['Er2']]))
eps_r1p_train = np.array(train[['Er1p']])/(np.array(train[['Er1p']])+np.array(train[['Er2p']]))

# Variables within the test set
Ec_test = np.array(test[['Etr']])+np.array(test[['Er1']])+np.array(test[['Er2']])
Ecp_test = np.array(test[['Etrp']])+np.array(test[['Er1p']])+np.array(test[['Er2p']])
eps_t_test = np.array(test[['Etr']])/Ec_test
eps_tp_test = np.array(test[['Etrp']])/Ecp_test
eps_r1_test = np.array(test[['Er1']])/(np.array(test[['Er1']])+np.array(test[['Er2']]))
eps_r1p_test = np.array(test[['Er1p']])/(np.array(test[['Er1p']])+np.array(test[['Er2p']]))



def inv_sigmoid(x):
    return np.log((x)/(1-(x)))

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig


x_train = np.hstack((np.log(Ec_train),  inv_sigmoid(eps_t_train),  inv_sigmoid(eps_r1_train)))
y_train = np.hstack((np.log(Ecp_train), inv_sigmoid(eps_tp_train), inv_sigmoid(eps_r1p_train)))
x_test = np.hstack((np.log(Ec_test),    inv_sigmoid(eps_t_test),   inv_sigmoid(eps_r1_test)))
y_test = np.hstack((np.log(Ecp_test),   inv_sigmoid(eps_tp_test),  inv_sigmoid(eps_r1p_test)))



plt.figure(figsize=[10,3])

plt.subplot(1,3,1)
plt.xlabel("$E_c^{(p)}$")
plt.ylabel("$E_c'^{(p)}$")
dscatter(x_train[...,0],y_train[...,0])


plt.subplot(1,3,2)
plt.xlabel(r"$\varepsilon_{t}^{(p)}$")
plt.ylabel(r"$\varepsilon_{t}'^{(p)}$")
dscatter(x_train[...,1],y_train[...,1])


plt.subplot(1,3,3)
plt.xlabel(r"$\varepsilon_{r,A}^{(p)}$")
plt.ylabel(r"$\varepsilon_{r,A}'^{(p)}$")
dscatter(x_train[...,2],y_train[...,2])

plt.tight_layout()
plt.show()



y_test = y_test[:,1:]
y_train = y_train[:,1:]

print('Shape of x-test:', x_test.shape)
print('Shape of x-train:', x_train.shape)
print('Shape of y-test:', y_test.shape)
print('Shape of y-train:', y_train.shape)



def plot_loss(history):
  plt.figure()
  plt.plot(history.history['loss'], label='Loss')
  plt.plot(history.history['val_loss'], label='Val loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(False)

def build_model(NGAUSSIANS, ACTIVATION, NNEURONS):
    
    event_shape = [2]
    num_components = NGAUSSIANS
    params_size = tfpl.MixtureSameFamily.params_size(num_components,
                    component_params_size=tfpl.IndependentNormal.params_size(event_shape))

    negloglik = lambda y, p_y: -p_y.log_prob(y)

    model = tf.keras.models.Sequential([
       tf.keras.layers.Dense(NNEURONS, activation=ACTIVATION),
       tf.keras.layers.Dense(params_size, activation=None),
       tfpl.MixtureSameFamily(num_components, tfpl.IndependentNormal(event_shape)),
    ])
    
    model.compile(optimizer=tf.optimizers.Adam(learning_rate = 1e-4), loss=negloglik)

    return model



CB = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
]

MaxEpochs = 1000



model = build_model(20, 'relu', 8)

history = model.fit(x_train, 
                    y_train, 
                    epochs = MaxEpochs, 
                    verbose = 1,
                    validation_data=(x_test,y_test),
                    callbacks=CB)

plot_loss(history)


preds = model.predict(x_test)

plt.figure(figsize=[8,6])

plt.subplot(2,2,1)
plt.title('MDN predictions')
dscatter(x_test[...,1],y_test[...,0])
plt.xlabel(r"$\varepsilon_{t}^{(p)}$")
plt.ylabel(r"$\varepsilon_{t}'^{(p)}$")
plt.xlim([-4,4])
plt.ylim([-4,4])

plt.subplot(2,2,2)
plt.title('Original data')
dscatter(x_test[...,1],preds[...,0])
plt.xlabel(r"$\varepsilon_{t}^{(p)}$")
plt.ylabel(r"$\varepsilon_{t}'^{(p)}$")
plt.xlim([-4,4])
plt.ylim([-4,4])

plt.subplot(2,2,3)
plt.title('Original data')
dscatter(x_test[...,2],y_test[...,1])
plt.xlabel(r"$\varepsilon_{r,A}^{(p)}$")
plt.ylabel(r"$\varepsilon_{r,A}'^{(p)}$")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.subplot(2,2,4)
plt.title('Original data')
dscatter(x_test[...,2],preds[...,1])
plt.xlabel(r"$\varepsilon_{r,A}^{(p)}$")
plt.ylabel(r"$\varepsilon_{r,A}'^{(p)}$")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.tight_layout()
plt.show()


plt.figure(figsize=[8,9])

plt.subplot(3,2,1)
plt.title('Original data')
plt.xlabel("Etr (K)")
plt.ylabel("Etr' (K)")
dscatter(sigmoid(x_test[...,1])*np.exp(x_test[...,0]),sigmoid(y_test[...,0])*np.exp(x_test[...,0]))
plt.xlim([0, 6000])
plt.ylim([0, 10000])

plt.subplot(3,2,2)
plt.title('Predicted data')
plt.xlabel("Etr (K)")
plt.ylabel("Etr' (K)")
dscatter(sigmoid(x_test[...,1])*np.exp(x_test[...,0]), sigmoid(preds[...,0])*np.exp(x_test[...,0]))
plt.xlim([0, 6000])
plt.ylim([0, 10000])

plt.subplot(3,2,3)
plt.xlabel("Er,A (K)")
plt.ylabel("Er,A' (K)")
dscatter((1-sigmoid(x_test[...,1]))*sigmoid(x_test[...,2])*np.exp(x_test[...,0]),(1-sigmoid(y_test[...,0]))*sigmoid(y_test[...,1])*np.exp(x_test[...,0]))
plt.xlim([0, 3000])
plt.ylim([0, 6000])

plt.subplot(3,2,4)
plt.xlabel("Er,A (K)")
plt.ylabel("Er,A' (K)")
dscatter((1-sigmoid(x_test[...,1]))*sigmoid(x_test[...,2])*np.exp(x_test[...,0]),(1-sigmoid(preds[...,0]))*sigmoid(preds[...,1])*np.exp(x_test[...,0]))
plt.xlim([0, 3000])
plt.ylim([0, 6000])

plt.subplot(3,2,5)
plt.xlabel("Er,B (K)")
plt.ylabel("Er,B' (K)")
dscatter((1-sigmoid(x_test[...,1]))*(1-sigmoid(x_test[...,2]))*np.exp(x_test[...,0]),(1-sigmoid(y_test[...,0]))*(1-sigmoid(y_test[...,1]))*np.exp(x_test[...,0]))
plt.xlim([0, 3000])
plt.ylim([0, 6000])

plt.subplot(3,2,6)
plt.xlabel("Er,B (K)")
plt.ylabel("Er,B' (K)")
dscatter((1-sigmoid(x_test[...,1]))*(1-sigmoid(x_test[...,2]))*np.exp(x_test[...,0]),(1-sigmoid(preds[...,0]))*(1-sigmoid(preds[...,1]))*np.exp(x_test[...,0]))
plt.xlim([0, 3000])
plt.ylim([0, 6000])

plt.tight_layout()
plt.show()


w1 = model.get_weights()[0]
b1 = model.get_weights()[1]
w2 = model.get_weights()[2]
b2 = model.get_weights()[3]

w1 = np.reshape(w1,np.product(w1.shape))
w2 = np.reshape(w2,np.product(w2.shape))

np.savetxt("w1.csv", w1, delimiter=",")
np.savetxt("b1.csv", b1, delimiter=",")
np.savetxt("w2.csv", w2, delimiter=",")
np.savetxt("b2.csv", b2, delimiter=",")