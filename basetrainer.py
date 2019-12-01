"""
BaseTrainer provides base functionality for any trainer object.
It provides functionality to:
    - run one epoch for train or validation
    - fit model for n epochs
"""
import torch
from torchvision import models
import math
import matplotlib.pyplot as plt
import copy
import os
import datetime
import torch.nn as nn
import time
from torch import optim
import numpy as np
from attacks import fgsm, pgd_linf
from utils import log_textfile
from scipy.io import wavfile


class BaseTrainer():
    """Base Class for fitting models."""

    def __init__(self, model, train_dl, valid_dl, criterion, n_epochs, model_filename):
        """Initialize the BaseTrainer object."""
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.criterion = criterion
        self.opt_lrs = []
        self.model_filename = model_filename
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        

    def run_epoch(self, epoch, loader, LOGFILE_PATH, optimizer=None, attack=None, epsilon=0.3, alpha=0.1, num_iter=7):
      running_loss = 0.
      running_corrects = 0.
      running_model = 0.
      n = 0.
      counter_batch = 0
      for X, y_true in loader:
        start = time.time()
        X = X.cuda()
        y_true = y_true.cuda()
        
        if optimizer!=None:
          optimizer.zero_grad()
        if attack!=None:
          delta = attack(self.model, X, y_true, epsilon, alpha, num_iter)
          delta = delta.cuda()
          y_pred = self.model(X + delta)
        else:
          y_pred = self.model(X)
        
        loss = self.criterion(y_pred, y_true) # input criterion is negative
        if optimizer!=None:
          loss.backward()
          optimizer.step()
        
        indices = torch.argmax(y_pred, dim=1)
        running_loss += float(loss)
        running_corrects += np.sum(indices.detach().cpu().numpy() == y_true.detach().cpu().numpy())
        
        n += y_true.detach().cpu().numpy().shape[0]
        counter_batch += 1
        end = time.time()
        delta_model = end - start
        running_model += delta_model
        
        if (counter_batch%20)==0:
            print(counter_batch)
            log_textfile(LOGFILE_PATH, '[epoch: %d, batch:  %5d] loss: %.5f time model: %.5f acc: %.5f' % (epoch + 1, counter_batch, running_loss/n, running_model / n, running_corrects / n))
      return(running_loss/n, running_corrects/n)
    
    def fit_model_new(self, optimizer, n_epochs, LOGFILE_PATH, model_filename, attack, epsilon, alpha, num_iter):

        self.model.train()

        final_loss = None
        lowest_valid_loss = 99999999

        for epoch in range(n_epochs):
          self.model.train()
          train_loss, train_acc = self.run_epoch(epoch, self.train_dl, LOGFILE_PATH, optimizer, attack, epsilon, alpha, num_iter)
            
          self.model.eval()
          valid_loss, valid_acc = self.run_epoch(epoch, self.valid_dl, LOGFILE_PATH, None, attack, epsilon, alpha, num_iter)
            
          if valid_loss < lowest_valid_loss:
            lowest_valid_loss = valid_loss
            lowest_val_epoch = epoch
            torch.save(self.model, 'saved/' + model_filename)
            
          log_textfile(LOGFILE_PATH, 'epoch:' + str(epoch + 1) + ' train loss: ' + str(train_loss) + ' train acc: ' + str(train_acc) + ' valid loss: ' + str(valid_loss) + ' valid acc: ' + str(valid_acc))
          print('epoch:' + str(epoch + 1) + ' train loss: ' + str(train_loss) + ' train acc: ' + str(train_acc) + ' valid loss: ' + str(valid_loss) + ' valid acc: ' + str(valid_acc))
        print('Lowest validation loss: ' + str(lowest_valid_loss) + ', Epoch num: ' + str(lowest_val_epoch))