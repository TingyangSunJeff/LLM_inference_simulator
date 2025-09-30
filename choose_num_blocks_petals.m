function [num_blocks] = choose_num_blocks_petals(total_memory,d_model, block_size, L, num_key_value_groups)
% replicate function '_choose_num_blocks' in petals.server.server to
% automatically calculate #blocks to place at the current server
% total_memory: value returned by 'torch.cuda.get_device_properties(self.device).total_memory'

gib = 2^30; % #bytes in a GiB
total_memory = total_memory * 10^9; % convert unit from GB to bytes
block_size = block_size * 10^9; % convert billion parameters

autograd_memory = (2*gib/14336) * d_model; % reserve 2 GiB for BLOOM-176B for gradient computation (proportional to d_model for other models); unit: bytes
if num_key_value_groups > 1
    attn_cache_tokens = 16384; % Value if multi-query attention is used
else
    attn_cache_tokens = 4096;  % Default value for single-query attention
end % an upper bound on max sequence length; see line 206 in petals.server.server
cache_bytes_per_block = 2 * d_model * attn_cache_tokens * 2; % upper bound on sc, #bytes of attention cache per block per session
cache_bytes_per_block = floor(cache_bytes_per_block / num_key_value_groups);
total_memory_per_block = block_size + cache_bytes_per_block; % upper bound on #bytes per block (including space for model parameters and attention cache), assuming a single inference session with sequence length 4096
num_blocks = min(floor((total_memory - autograd_memory) / total_memory_per_block), L);
% disp([block_size, cache_bytes_per_block, attn_cache_tokens])
% disp([total_memory, autograd_memory, total_memory_per_block])
end