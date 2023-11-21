"""
Aditya Patel
Implementation of the GAN Algorithm on the Gaussian Mixed Model
Stat 598 - Statistical Machine Learning 
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import layers

def build_generator(latent_space):
   model = tf.keras.Sequential([
      layers.Dense(128, input_dim=latent_space, activation='relu'),
      layers.Dense(2, activation='tanh')
   ])
   return model
   
def build_discriminator(input_dim):
  model = tf.keras.Sequential([
    layers.Dense(128, input_dim=input_dim, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])
  return model

def build_GAN(generator, discriminator):
  discriminator.trainable = False
  model = tf.keras.Sequential([
    generator,
    discriminator  
  ])
  return model

def GAN_Generator(num_samples, gmm_samples, latent_space, opt='adam', bts=64, epoch_ct=300):
   # Define models
  generator = build_generator(latent_space)
  discriminator = build_discriminator(2)
  gan = build_GAN(generator, discriminator)

   # Compile models
  discriminator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  gan.compile(loss='binary_crossentropy', optimizer=opt)


   # Training
  for epoch in range(epoch_ct):
    noise = np.random.normal(0, 1, size=(bts, latent_space))
    gen_samples = generator.predict(noise)

    idx = np.random.randint(0, gmm_samples.shape[0], bts)
    real_samples = gmm_samples[idx]

    # labels for generated and real data
    labels_r = np.ones((bts, 1))
    labels_f = np.zeros((bts, 1))

    # Train Discriminator
    d_loss_r = discriminator.train_on_batch(real_samples, labels_r)
    d_loss_f = discriminator.train_on_batch(gen_samples, labels_f)
    d_loss = 0.5 * (np.add(d_loss_r, d_loss_f))

    # Train Generator
    noise = np.random.normal(0, 1, size=(bts, latent_space))
    labels_gan = np.ones((bts, 1))
    g_loss = gan.train_on_batch(noise, labels_gan)

    # Print Progress
    if epoch % 100 == 0:
      print(f'Epoch [{epoch}/{epoch_ct}]: Discriminator Loss: {d_loss[0]:.4f} | Generator Loss: {g_loss:.4f}')

  # Generate samples
  gan_samples = generator.predict(np.random.normal(0, 1, size=(num_samples, latent_space)))
  return gan_samples
  

def BivariateNormSamples(num_samples, num_components, means, var):
    covars = np.array([[[var, 0],[0, var]]] * num_components)
    
    biv_norm_samples = np.concatenate([
            np.random.multivariate_normal(means[k], covars[k], size=int(num_samples/num_components)) for k in range(num_components)
    ])
    
    return biv_norm_samples

def GAN_Driver(num_samples, num_comps, std_dev, latent_space, epochs):
  # Define GMM parameters
  variance = std_dev ** 2
  means = np.array([(2 * np.cos((2 * np.pi * k)/num_comps), 2 * np.sin((2 * np.pi * k)/num_comps)) for k in range(num_comps)])
  
  # Obtain samples from Bivariate Normal distribution
  biv_norm_samples = BivariateNormSamples(num_samples, num_comps, means, variance)
  gan_samples = GAN_Generator(num_samples, biv_norm_samples, latent_space, opt='adam', bts=64, epoch_ct=epochs)

  # Plot results
  plt.scatter(gan_samples[:, 0], gan_samples[:, 1], label='GAN Samples', alpha=0.5)
  plt.scatter(biv_norm_samples[:, 0], biv_norm_samples[:, 1], label='Bivariate Normal Samples', alpha=0.5)
  plt.title('Samples from GAN and Bivarariate Normals')
  plt.legend()
  plt.show()