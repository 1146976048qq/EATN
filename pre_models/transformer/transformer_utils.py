class Batch:
    """
    objects for holding a batch of data with mask during training
    """
    def __init__(self, src,trg=None, pad=0):
        self.src = src
        self.src_mask = (src!=pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:,:-1]
            self.trg_y = trg[:,1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
            
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        create a mask to hide padding and future words
        """
        tgt_mask = (tgt!=pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
    
def run_epoch(data_iter, model, loss_compute):
    """
    Standard training and logging function
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i%50==1:
            elapsed = time.time()-start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_run(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count==1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class NoamOpt:
    """
    Optim wrapper that implements rate
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """
        update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor*(self.model_size**(-0.5)*min(step**(-0.5),step*self.warmup**(-1.5)))
    
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2,4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self,size,padding_idx,smoothing=0.0):
        super(LabelSmoothing,self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing/(self.size-2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:,self.padding_idx] = 0
        mask = torch.nonzero(target.data==self.padding_idx)
        if mask.dim()>0:
            true_dist.index_fill_(0,mask.squeeze(),0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist,requires_grad=False))
    
def data_gen(V, batch, nbatches):
    """
    Generate random data for a src-tgt copy task
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:,0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src,tgt,0)
        
class SimpleLossCompute:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt 
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1,x.size(1)),y.contiguous().view(-1))/norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0]*norm 