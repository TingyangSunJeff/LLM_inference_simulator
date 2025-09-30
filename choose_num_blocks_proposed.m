function [num_blocks] = choose_num_blocks_proposed(total_memory,sm,sc,R, L)
% total_memory: total usable memory in GiB
% sm, sc: block size and cache size (per block per session), in GiB

num_sessions = R; % target #sessions a server needs to handle
total_memory_per_block = sm + sc*num_sessions; % upper bound on #bytes per block (including space for model parameters and attention cache), assuming a single inference session with sequence length 4096
num_blocks = min(floor(total_memory / total_memory_per_block), L); % a conservative lower bound on #blocks this server (with memory 'total_memory') should host, such that it still has memory to hold attention caches even if all the R requests are routed through it

end