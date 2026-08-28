import torch

class QuestManager():
    def __init__(self, config):
        self.page_size = self.num_local = config['page_size']
        self.top_k = config['top_k']
        self.update_interval = config['update_interval']
        self.num_page_kv = self.top_k * self.page_size
        self.repr_keys = None
        self.curr_ptr = 0
        self.prev_kv_len = 0
        self.last_page_idx = self.num_local
        self.update_ctr = 0
    
    def update(self, kv, query):
        kvcache_len = kv[0].shape[-2]
        n_new_tokens = new_prefill = (kvcache_len - self.prev_kv_len)
        self.prev_kv_len = kvcache_len
        kv_start = kvcache_len - new_prefill
        kv_end = kvcache_len
        self.update_ctr += n_new_tokens
        if n_new_tokens > self.num_local:
            n_new_tokens = self.num_local
            kv_start = kv_end - self.num_local # shift source to align
        
        if kv_end > self.num_local:
            # Compute write indices in circular manner
            local_start = self.curr_ptr
            local_end = local_start + n_new_tokens

            overlap_start = max(kv_start, local_start)
            overlap_end = min(kv_end, local_end)
            if overlap_start < overlap_end:
                # Reduce n_new_tokens to avoid overlap
                n_new_tokens = max(0, overlap_start - local_start)
                local_end = local_start + n_new_tokens
                kv_start = kv_end - n_new_tokens

            if local_end <= self.num_local:
                tmp_k = kv[0][:,:,local_start:local_end,:].clone()
                tmp_v = kv[1][:,:,local_start:local_end,:].clone()

                kv[0][:,:,local_start:local_end,:] = kv[0][:,:,kv_start:kv_end,:]
                kv[1][:,:,local_start:local_end,:] = kv[1][:,:,kv_start:kv_end,:]

                kv[0][:,:,kv_start:kv_end,:] = tmp_k
                kv[1][:,:,kv_start:kv_end,:] = tmp_v
            else:
                first_chunk = self.num_local - local_start
                second_chunk = n_new_tokens - first_chunk

                # Backup original local values
                tmp_k = torch.cat([
                    kv[0][:,:,local_start:self.num_local,:],
                    kv[0][:,:,:second_chunk,:]
                ], dim=2)

                tmp_v = torch.cat([
                    kv[1][:,:,local_start:self.num_local,:],
                    kv[1][:,:,:second_chunk,:]
                ], dim=2)

                # Overwrite local window with new values
                kv[0][:,:,local_start:self.num_local,:] = kv[0][:,:,kv_start:kv_start+first_chunk,:]
                kv[0][:,:,:second_chunk,:] = kv[0][:,:,kv_start+first_chunk:kv_end,:]

                kv[1][:,:,local_start:self.num_local,:] = kv[1][:,:,kv_start:kv_start+first_chunk,:]
                kv[1][:,:,:second_chunk,:] = kv[1][:,:,kv_start+first_chunk:kv_end,:]

                # Overwrite full KV with old local values
                kv[0][:,:,kv_start:kv_end,:] = tmp_k
                kv[1][:,:,kv_start:kv_end,:] = tmp_v
        
        # Update repr_keys 
        self._update_repr_keys(kv, kv_end, query)
        
        # Update the chosen top-k here
        if kv[0].shape[-2] // self.page_size - 1 > self.top_k and self.update_ctr > self.update_interval * self.page_size:
            self._update_topk_pages(kv, query)


        return n_new_tokens

    def _update_topk_pages(self, kv, query):
        self.update_ctr = 0
        # query may be [B, H, D] or [B, H, T, D]; use a single representative token
        if query.dim() == 4:
            query = query[:, :, 0, :]
        sign = (query > 0) + (~(query > 0)) * -1 # 1 if > 0; otherwise, -1
        query = query * sign
        flipped_repr = self.repr_keys * sign.unsqueeze(2).unsqueeze(2)
        flipped_repr = flipped_repr.amax(dim=-2) # [batch, num_head, num_page, head_dim]
        quantized_weight = torch.matmul(
            query.unsqueeze(2),
            flipped_repr.transpose(2, 3),
        ).squeeze(2).amax(dim=1) # [batch, seq_len]
        k = min(max(3, self.top_k), quantized_weight.size(-1))
        _, topk = quantized_weight.topk(
            k=k, dim=-1
        )
        topk = topk.detach().cpu().tolist()

        # Swap pages
        for batch_idx in range(len(topk)):
            topk_dict = {}
            not_in_topk_range = []
            to_swap = []

            for i in range(len(topk[0])):
                page_idx = topk[batch_idx][i]
                topk_dict[page_idx] = 1
                if page_idx >= self.top_k:
                    not_in_topk_range.append(page_idx)
            
            for i in range(self.top_k):
                if i not in topk_dict:
                    to_swap.append(i)

            if flipped_repr.shape[2] > self.top_k:
                assert len(not_in_topk_range) == len(to_swap)
            
            for out_idx, in_idx in zip(to_swap, not_in_topk_range):
                out_start = self.num_local + out_idx * self.page_size
                out_end = out_start + self.page_size
                in_start = self.num_local + in_idx * self.page_size
                in_end = in_start + self.page_size

                # Swap kv
                temp = kv[0][:, :, out_start:out_end, :].clone()
                kv[0][:, :, out_start:out_end, :] = kv[0][:, :, in_start:in_end, :]
                kv[0][:, :, in_start:in_end, :] = temp
                temp = kv[1][:, :, out_start:out_end, :].clone()
                kv[1][:, :, out_start:out_end, :] = kv[1][:, :, in_start:in_end, :]
                kv[1][:, :, in_start:in_end, :] = temp

                # Swap repr_keys
                out_idx_tensor = self.repr_keys[:, :, out_idx : out_idx + 1, :, :].clone()
                in_idx_tensor  = self.repr_keys[:, :, in_idx  : in_idx  + 1, :, :].clone()
                self.repr_keys[:, :, out_idx : out_idx + 1, :, :] = in_idx_tensor
                self.repr_keys[:, :, in_idx  : in_idx  + 1, :, :] = out_idx_tensor

                                

    def _update_repr_keys(self, kv, kv_end, query):
        while kv_end - self.last_page_idx >= self.page_size:
            kv_slice = kv[0][:,:,self.last_page_idx:self.last_page_idx + self.page_size,:]
            max_key = kv_slice.amax(dim=-2)
            min_key = kv_slice.amin(dim=-2) #[batch, num_head, head_dim]
            if kv[0].shape[1] != query.shape[1]:
                repeat_time = query.shape[1] // kv[0].shape[1]
                min_key = min_key.repeat_interleave(repeat_time, dim=1)
                max_key = max_key.repeat_interleave(repeat_time, dim=1)
            cat_key = torch.stack([max_key, min_key], dim=2).unsqueeze(2)
            if self.repr_keys is None:
                self.repr_keys = cat_key
            else:
                self.repr_keys = torch.cat([self.repr_keys, cat_key], dim=2) # [batch, num_heads, num_page + 1, 2, head_dim]
            
            self.last_page_idx += self.page_size

    def advance_ptr(self, n_new_tokens, kv_end):
        if kv_end > self.num_local:
            self.curr_ptr = (self.curr_ptr + n_new_tokens) % self.num_local

    def reset(self):
        self.repr_keys = None
        self.curr_ptr = 0
        self.prev_kv_len = 0
        self.last_page_idx = self.num_local
        self.update_ctr = 0