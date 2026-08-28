import torch

class StreamManager():
    def __init__(self, config):
        self.num_init = config['num_init']
        self.num_local = config['num_local']
        self.curr_ptr = 0
        self.prev_kv_len = 0
    
    def update(self, kv, query):
        kvcache_len = kv[0].shape[-2]
        if self.num_init < 1:
            self.num_init = int(self.num_init * kvcache_len)
        if self.num_local < 1:
            self.num_local = int(self.num_local * kvcache_len)
        n_new_tokens = new_prefill = (kvcache_len - self.prev_kv_len)
        self.prev_kv_len = kvcache_len
        kv_start = kvcache_len - new_prefill
        kv_end = kvcache_len
        if n_new_tokens > self.num_local:
            n_new_tokens = self.num_local
            kv_start = kv_end - self.num_local # shift source to align
        
        if kv_end > self.num_init + self.num_local:
            # Compute write indices in circular manner
            local_start = self.num_init + self.curr_ptr
            local_end = local_start + n_new_tokens

            overlap_start = max(kv_start, local_start)
            overlap_end = min(kv_end, local_end)
            if overlap_start < overlap_end:
                # Reduce n_new_tokens to avoid overlap
                n_new_tokens = max(0, overlap_start - local_start)
                local_end = local_start + n_new_tokens
                kv_start = kv_end - n_new_tokens

            if local_end <= self.num_init + self.num_local:
                tmp_k = kv[0][:,:,local_start:local_end,:].clone()
                tmp_v = kv[1][:,:,local_start:local_end,:].clone()

                kv[0][:,:,local_start:local_end,:] = kv[0][:,:,kv_start:kv_end,:]
                kv[1][:,:,local_start:local_end,:] = kv[1][:,:,kv_start:kv_end,:]

                kv[0][:,:,kv_start:kv_end,:] = tmp_k
                kv[1][:,:,kv_start:kv_end,:] = tmp_v
            else:
                first_chunk = self.num_init + self.num_local - local_start
                second_chunk = n_new_tokens - first_chunk

                # Backup original local values
                tmp_k = torch.cat([
                    kv[0][:,:,local_start:self.num_init+self.num_local,:],
                    kv[0][:,:,self.num_init:self.num_init+second_chunk,:]
                ], dim=2)

                tmp_v = torch.cat([
                    kv[1][:,:,local_start:self.num_init+self.num_local,:],
                    kv[1][:,:,self.num_init:self.num_init+second_chunk,:]
                ], dim=2)

                # Overwrite local window with new values
                kv[0][:,:,local_start:self.num_init+self.num_local,:] = kv[0][:,:,kv_start:kv_start+first_chunk,:]
                kv[0][:,:,self.num_init:self.num_init+second_chunk,:] = kv[0][:,:,kv_start+first_chunk:kv_end,:]

                kv[1][:,:,local_start:self.num_init+self.num_local,:] = kv[1][:,:,kv_start:kv_start+first_chunk,:]
                kv[1][:,:,self.num_init:self.num_init+second_chunk,:] = kv[1][:,:,kv_start+first_chunk:kv_end,:]

                # Overwrite full KV with old local values
                kv[0][:,:,kv_start:kv_end,:] = tmp_k
                kv[1][:,:,kv_start:kv_end,:] = tmp_v
        
        return n_new_tokens

    def advance_ptr(self, n_new_tokens, kv_end):
        if kv_end > self.num_init + self.num_local:
            self.curr_ptr = (self.curr_ptr + n_new_tokens) % self.num_local

    def reset(self):
        self.curr_ptr = 0
        self.prev_kv_len = 0
