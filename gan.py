#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch 
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import cv2
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from IPython.display import Image
from torchvision.utils import save_image


# In[9]:


mnist = MNIST(root = 'Trainig_data/',
             train = True,
             download = True, 
             transform = Compose([ToTensor(), Normalize(mean=(0.5), std = (0.5,))]))


# In[10]:


def denorm(x):
    out = (x+1) / 2
    return out.clamp(0, 1)


# In[11]:


# Device Configuring for better performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[12]:


img, label = mnist[0] 


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline # this is magic command')
i = denorm(img) 
plt.imshow(i[0], cmap = 'gray')
print('Label: ', label)


# In[14]:


img, label = mnist[0]
print('Label: ', label)
print(img[:,10:15,10:15])
torch.min(img), torch.max(img)


# In[15]:


from torch.utils.data import DataLoader

batch_size = 100
data_loader = DataLoader(mnist, batch_size, shuffle = True)


# batch size is some thing that is we are taking batches of 100 each so total batches will be 70000/100 i.e 700

# In[16]:


for img_batch, label_batch in data_loader:
    print('50th batch')
    print(img_batch[50].shape)
    plt.imshow(img_batch[50][0], cmap = 'gray')
    print( label_batch )
    break


# In[17]:


img_size = 784
hidden_size = 512


# In[18]:


D = nn.Sequential(
    nn.Linear(img_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())


# In[19]:


D.to(device)


# In[20]:


latent_size = 64


# In[21]:


G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, img_size),
    nn.Tanh())
  


# In[22]:


G.to(device)


# In[23]:


criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(),lr = 0.0002)
g_optimizer = torch.optim.Adam(G.parameters(),lr = 0.0002)


# In[24]:


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    
    


# In[25]:


def train_discriminator(img):
    # Created the labels which are later used as input for the BCE loss
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)
    
    #loss for real images
    outputs = D(img)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs
    
    #loss for fake images
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_scores = outputs
    
    #combine losses
    d_loss = d_loss_real + d_loss_fake
    
    #Reset Gradients
    reset_grad()
    
    # Compute gradients
    d_loss.backward()
    
    #Adjust the parameters using backprop
    d_optimizer.step()
    
    return d_loss, real_score, fake_scores


# In[26]:


x = 15
for fake_images in range(x):
    z = torch.randn(batch_size, latent_size).to(device)
    fake_images = G(z)
    D(fake_images)
    print(denorm(fake_images))


# In[27]:


def train_generator():
    # Generate fake images and calculation loss
    z = torch.randn(batch_size, latent_size). to(device)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1).to(device)
    g_loss = criterion(D(fake_images), labels)
    
    
    #Backprop and optimizer
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images
    


# In[28]:


import os 
sample_dir = 'samples/'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# In[29]:


from IPython.display import Image
from torchvision.utils import save_image

for images, _ in data_loader:
    images = images.reshape(images.size(0), 1, 28, 28)
    save_image(denorm(images), 
                os.path.join(sample_dir, 'real_images.png'), 
                nrow = 10)
    
Image(os.path.join(sample_dir, 'real_images.png'))


# In[30]:


sample_vectors = torch.randn(batch_size, latent_size).to(device)

def save_fake_images(index):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)
    
# Before training
save_fake_images(0)
Image(os.path.join(sample_dir, 'fake_images-0000.png'))


# In[31]:


get_ipython().run_cell_magic('time', '', "\nnum_epochs = 300\ntotal_step = len(data_loader)\nd_losses, g_losses, real_scores, fake_scores = [], [], [], []\n\nfor epoch in range(num_epochs):\n    for i, (images, _) in enumerate(data_loader):\n        # Load a batch & transform to vectors\n        images = images.reshape(batch_size, -1).to(device)\n        \n        # Train the discriminator and generator\n        d_loss, real_score, fake_score = train_discriminator(images)\n        g_loss, fake_images = train_generator()\n        \n        # Inspect the losses\n        if (i+1) % 200 == 0:\n            d_losses.append(d_loss.item())\n            g_losses.append(g_loss.item())\n            real_scores.append(real_score.mean().item())\n            fake_scores.append(fake_score.mean().item())\n            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' \n                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), \n                          real_score.mean().item(), fake_score.mean().item()))\n        \n    # Sample and save images\n    save_fake_images(epoch+1)")


# In[47]:


# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')


# In[33]:


import cv2
import os
from IPython.display import FileLink

video_avi = 'GAN_Output.avi'

files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'fake_images' in f]
files.sort()

out = cv2.VideoWriter(video_avi,cv2.VideoWriter_fourcc(*'MP4V'), 8, (302,302))
[out.write(cv2.imread(fname)) for fname in files]
out.release()
FileLink('GAN_Output.avi')


# In[34]:


plt.plot(d_losses, '-')
plt.plot(g_losses, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses')


# In[35]:


plt.plot(real_scores, '-')
plt.plot(fake_scores, '-')
plt.xlabel('epoch')
plt.ylabel('score')
plt.legend(['Real Score', 'Fake score'])
plt.title('Scores');


# In[36]:


get_ipython().system('pip install jovian --upgrade -q')


# In[37]:


import jovian


# In[2]:


get_ipython().system('jupyter nbconvert --to script GAN[Generative Adversarial Network].ipynb')


# In[ ]:




