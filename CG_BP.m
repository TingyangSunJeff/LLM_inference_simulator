function [soln_a,soln_m,soln_val, optimized_order] = CG_BP(RTT,tau,L,sm,sc,M,R, server_types)
% CG-BP (offline block placement):
% R: target #concurrent requests
% soln_a, soln_m: block placement
% soln_val: upper bound on per-token inference time as in CG-BPRR theorem
% optimized_order: the order of placing blocks on servers (to be used to
% evaluate the benchmark "Optimized Order")
n_client = size(RTT,1); % RTT(c,j): per-token RTT between each client c and each server j
n_server = size(RTT,2); 

% 1. compute #blocks per server:
soln_m = zeros(n_server,1);
for i=1:n_server 
    soln_m(i) = choose_num_blocks_proposed(M(i),sm,sc,R, L);
    if soln_m(i) > 4 & server_types(i) == "MIG"
        soln_m(i) = 4;
    end
end

% 2. greedy block placement (compute soln_a):
time_c = zeros(n_client,n_server); 
for c = 1:n_client
    time_c(c,:) = RTT(c,:) + (tau.*soln_m)'; % t^c_j: maximum inference time at server j for a request from client c (even if all the hosted blocks are processed)
end
time = max(time_c,[],1)'; % time(j) is t_j: upper bound on the inference time t^c_{ij} for any client c and any i with (i,j)\in E
time_per_block = time./soln_m; % upper bound on the average inference time per block (time_per_block(j): \tilde{t}_j)
maxtime_per_block = max(time_per_block)+1; % inference time per block at the virtual servers
Cb = zeros(1,L); % total capacity (in #requests) of servers hosting each block
Tb = R*maxtime_per_block*ones(1,L); % total (amortized) inference time spent at each block
[~,order] = sort(time_per_block);
optimized_order = order;
max_sessions = floor((M-sm.*soln_m)./(sc.*soln_m)); % (lower bound on) #sessions each server can host
soln_a = zeros(n_server,1); % soln_a(j) \in {1,...,L-soln_m(j)+1}
K = inf; % min #servers to cover all the L blocks
for i=1:n_server % for each server in increasing order of per-block inference time
    j = order(i); 
    if any(Cb<R) % if not fully serving all the blocks yet
        sumT = 0;
        for a=1:L-soln_m(j)+1
            if any(Cb(a:a+soln_m(j)-1)<R) && sum(Tb(a:a+soln_m(j)-1))>sumT % place the set of blocks that contains at least one underserved block, while having the maximum total amortized inference time
                soln_a(j) = a;
                sumT = sum(Tb(a:a+soln_m(j)-1));
            end
        end
    else % if fully serving all the blocks
        capacities = zeros(L-soln_m(j)+1,soln_m(j));
        for a=1:L-soln_m(j)+1
            capacities(a,:) = sort(Cb(a:a+soln_m(j)-1));
        end
        [~,I] = sortrows(capacities); % I(1) is the row index of the "minimum row" of capacities
        soln_a(j) = I(1); % place the set of blocks with the minimum capacities under the existing block placement in lexicongraphical order
    end
    range = soln_a(j):soln_a(j)+soln_m(j)-1; % range of blocks placed at server j
    Tb(range) = Tb(range) - (maxtime_per_block-time_per_block(j))*min(max(R-Cb(range),0), max_sessions(j)); % update the amortized inference times
    Cb(range) = Cb(range) + max_sessions(j); % update the per-block capacities
    if all(Cb>=R) % if the top i servers host all the blocks
        K = min(K,i); % K is the min #servers to cover all the blocks
    end
end
if any(Cb<R) % if not finding a feasible block placement:
    error(['CG_BP: not all the blocks are placed']);
else % if the block placement is feasible:
    soln_val = sum(time_per_block(order(1:K)).*soln_m(order(1:K))) - tau(order(K))*(sum(soln_m(order(1:K)))-L); % upper bound on per-token inference time as in CG-BPRR theorem, if #concurrent requests <= R
end



end

