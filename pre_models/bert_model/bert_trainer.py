import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from .bert_saver import RunningAverage, accuracy, save_dict_to_json, save_checkpoint, load_checkpoint

class BertTrainer(object):
    def __init__(self, device, batch_size=32, n_epochs=5, min_clip_val=-1.0, max_clip_val=1.0, n_labels=2):
        """
        Args:
            device: cuda or cpu
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.min_clip_val = min_clip_val
        self.max_clip_val = max_clip_val
        self.device = device
        self.n_labels = n_labels
    
    def clip_gradient(self, model):
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(self.min_clip_val, self.max_clip_val)
    
    def train(self, model, train_data, optimizer,sum_writer, epoch, metrics=None, loss_fn=None):
        """
        Args:
            train_data: dataLoader
            model: model instance
            optimizer:
            sum_writer: tensorboarX SummaryWriter
            loss_fn: loss function to do backpropagation
            metrics: accuracy, etc
        """
        import logging
        from tqdm import trange, tqdm

        # set model to training mode
        model.train()
    
        # summary for current training loop and a running objct for loss
        summ = []
        loss_avg = RunningAverage()
        #for i, (X_batch, y_batch) in tqdm(enumerate(train_data)):
        #for i, (X_batch, y_batch) in enumerate(train_data):
        for i, batch in enumerate(tqdm(train_data, desc='train_data_iter')):
            # X_batch is a tuple
            batch = tuple(x.to(self.device) for x in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            # clear previous gradients, compute gradients of all variabels wrt loss
            optimizer.zero_grad()
            
            # compute model output and loss
            logits = model(input_ids, input_mask, segment_ids, label_ids)
            # try:
            #     pred_batch = model.get_prediction_result()
            # except: 
            #     pred_batch = model.module.get_prediction_result()
            #print('train loss shape: ', loss.shape)
            
            if self.n_labels > 2:
                pred_batch = F.softmax(logits, dim=-1)
            else:
                pred_batch = F.sigmoid(logits)
            logging.debug('bert trainer: prediction_result {}'.format(pred_batch.shape))

            loss = loss_fn(logits.view(-1, self.n_labels), label_ids.view(-1))

            if metrics is not None and 'accuracy' in metrics.keys():
                acc = metrics['accuracy'](label_ids, pred_batch)
            else:
                acc = accuracy(label_ids, pred_batch)
            #logging.info('train loss: {}, train acc: {}'.format(loss, acc))
            
            # performs updates using calculated gradients
            loss.backward()
            self.clip_gradient(model)
            optimizer.step()
        
            # Evaluate summaries only once in a while
            if i % 20==0:            
                logging.info('Step:{}, Training Loss: {:05.3f}, Training Accuracy: {:05.2f}'.format(i, loss.item(), acc.item()))

                # compute all metrics on this batch
                summary_batch = {}
                if metrics is not None:
                    summary_batch = {metric: metrics[metric](label_ids, pred_batch).item() for metric in metrics}
                summary_batch['loss'] = loss.item()   #loss.data[0]
                summ.append(summary_batch)
                # add scalars to tensorboardX
                for key, val in summary_batch.items():
                    sum_writer.add_scalar('train/{}'.format(key), val, epoch)
                      
                #y_pred_labels = torch.max(pred_batch, 1)[1].view(y_batch.size())
                #sum_writer.add_pr_curve('train/pre_recall', y_batch.data.cpu().numpy(), y_pred_labels.data.cpu().numpy(), epoch)

            # update the average loss
            loss_avg.update(loss.item())
            #t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = '; '.join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info('- Train metrics: '+ metrics_string)
        return metrics_mean
    
    def evaluate(self, model, dev_data, optimizer,sum_writer, epoch, metrics=None, loss_fn=None):
        """
        Args:
            dev_data: dataLoader for dev set
            model: model instance
            optimizer:
            sum_writer: tensorboarX SummaryWriter
            loss_fn: loss function to do backpropagation
            metrics: accuracy, etc
        """
        import logging
        from tqdm import trange, tqdm
        import torch

        # set model to evaluation mode
        model.eval()
    
        # summary for current training loop and a running objct for loss
        summ = []
        loss_avg = RunningAverage()
    
        with torch.no_grad():
            for i, batch in enumerate(dev_data):
                batch = tuple(x.to(self.device) for x in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                # compute model output and loss
                logits = model(input_ids, input_mask, segment_ids, label_ids)
                #pred_batch = model.get_logits()
                if self.n_labels > 2:
                    pred_batch = F.softmax(logits, dim=-1)
                else:
                    pred_batch = torch.sigmoid(logits)
                #logging.debug('bert trainer: prediction_result {}'.format(pred_batch.shape))

                loss = loss_fn(logits.view(-1, self.n_labels), label_ids.view(-1))
                
                #print('evaluate loss shape: ', loss.shape)
                if metrics is not None and 'accuracy' in metrics.keys():
                    acc = metrics['accuracy'](label_ids, pred_batch)
                else:
                    acc = accuracy(label_ids, pred_batch)
                #logging.info('eval loss: {}, eval acc: {}'.format(loss, acc))

                # Evaluate summaries only once in a while
                if i % 20==0:          
                    logging.info('Step:{}, Evaluate Loss: {:05.3f}, Evaluate Accuracy: {:05.2f}'.format(i, loss.item(),acc.item()))
        
                    # compute all metrics on this batch
                    summary_batch = {}
                    if metrics is not None:
                        summary_batch = {metric: metrics[metric](label_ids, pred_batch).item() for metric in metrics}
                    summary_batch['loss'] = loss.item()   #loss.data[0]
                    summ.append(summary_batch)
                    # add scalars to tensorboardX
                    for key, val in summary_batch.items():
                        sum_writer.add_scalar('evaluate/{}'.format(key), val, epoch)
                        
                    #y_pred_labels = torch.max(pred_batch, 1)[1].view(y_batch.size())
                    #sum_writer.add_pr_curve('evaluate/pre_recall', y_batch.data.cpu().numpy(), y_pred_labels.data.cpu().numpy(), epoch)
                    
                    #sum_writer.add_pr_curve('evaluate/pre_recall', y_batch.data.cpu().numpy(), pred_batch.data.cpu().numpy(), epoch)
                    
                # update the average loss
                loss_avg.update(loss.item())
                #t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

        # compute mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = '; '.join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info('- Eval metrics: '+ metrics_string)
        return metrics_mean    
        
    def train_and_evaluate(self, model, train_data, dev_data, optimizer,metrics=None, loss_fn=None,model_dir='./'):
        """
        Args:
            train_data: dataLoader for train set
            dev_data: dataloader for dev set
            model: model instance
            optimizer:
            loss_fn: loss function to do backpropagation
            metrics: accuracy, etc
        """
        import os
        import logging
        from tqdm import trange, tqdm
        from tensorboardX import SummaryWriter
        #from livelossplot import PlotLosses
    
        best_val_acc = 0.0
        sum_writer = SummaryWriter() # add summary of training to tensorboardX
        #liveloss = PlotLosses()
        for epoch in trange(self.n_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, self.n_epochs))
            logs = {}
            # train the model
            train_metrics = self.train(model, train_data, optimizer, sum_writer, epoch, metrics, loss_fn)            
            val_metrics = self.evaluate(model, dev_data, optimizer, sum_writer, epoch, metrics, loss_fn)
        
            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc

            # Save weights
            save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optim_dict' : optimizer.state_dict()}, 
                        is_best=is_best, checkpoint=model_dir)
            
            # If best_eval, best_save_path        
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc
            
                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
                save_dict_to_json(val_metrics, best_json_path)
            
            # draw liveloss plot
            for key, val in train_metrics.items():
                logs[key] = val
            for key, val in val_metrics.items():
                logs['val_'+key] = val
            #liveloss.update(logs)
            #liveloss.draw()
            
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)
        
        # export training details to json
        #batch_size = train_data.batch_size
        #seq_len = train_data.dataset.shape()[1]
        #dum_input = torch.zeros(batch_size, seq_len, dtype=torch.int64).to(model.device)
        #sum_writer.add_graph(model, dum_input, verbose=True)
        sum_writer.export_scalars_to_json(os.path.join(model_dir,'train_scalars.json'))
        sum_writer.close()
