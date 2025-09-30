function [start] = block_selection_petals(throughputs,m)
% simulate the block selection of Petals in 'petals.server.block_selection'
% throughputs: 1*L array of the total throughput for each block under
% existing placement (throughputs(i) is the sum throughput of all servers
% hosting block i)
% m: #blocks to be placed to the current server (block_selection is invoked
% as each server joins the swarm, sequentially)
% return 'start', the index of the first block placed at this server

L = length(throughputs); % total #blocks
throughputs = reshape(throughputs,1,L);
candidates = zeros(L-m+1,m);
for i=1:L-m+1 % for each candidate value of 'start'
    candidates(i,:) = sort(throughputs(i:i+m-1));
end
[~,I] = sortrows(candidates); % I(1) is the row index of the "minimum row" of throughputs 
start = I(1); 

end

%% test: 
% throughputs = [770*ones(1,40), zeros(1,30)];
% m = 4;
% [start] = block_selection_petals(throughputs,m)