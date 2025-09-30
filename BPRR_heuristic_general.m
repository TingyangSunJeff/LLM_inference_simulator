function [soln_f,soln_a,soln_m,soln_val, optimized_order] = BPRR_heuristic_general(n_client,n_server,n_request_client,lc,RTT,tau,L,sm,sc,M,E,V,R,link,link_list)
% MILP formulation of BPRR (general case):

% 1. compute #blocks per server:
soln_m = zeros(n_server,1);
for i=1:n_server 
    soln_m(i) = choose_num_blocks_proposed(M(i),sm,sc,R, L);
end

% 2. greedy block placement (compute soln_a):
time_c = zeros(n_client,n_server); 
for c = 1:n_client
    time_c(c,:) = RTT(c,:) + (tau.*soln_m)'; % t^c_j: maximum inference time at server j for a request from client c (even if all the hosted blocks are processed)
end
time = max(time_c,[],1)'; % time(j) is t_j: upper bound on the inference time t^c_{ij} for any client c and any i with (i,j)\in E
time_per_block = time./soln_m; % upper bound on the average inference time per block
maxtime_per_block = max(time_per_block)+1; % inference time per block at the virtual servers
Cb = zeros(1,L); % total capacity (in #requests) of servers hosting each block
Tb = R*maxtime_per_block*ones(1,L); % total (amortized) inference time spent at each block
[~,order] = sort(time_per_block);
optimized_order = order;
max_sessions = floor((M-sm.*soln_m)./(sc.*soln_m)); % (lower bound on) #sessions each server can host
soln_a = zeros(n_server,1); % soln_a(j) \in {1,...,L-soln_m(j)+1}
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
end
if any(Cb<R) % if not finding a feasible block placement:
    error(['BPRR_heuristic_general: placed blocks cannot fully serve all the requests']);
else % if the block placement is feasible:
    [soln_f,soln_val] = shortest_path_given_BP(soln_m,soln_a, n_client,n_server,n_request_client,lc,RTT,tau,L,E,R,link);
end

isdebug = 0;
% sanity check: soln_val should be the same as that of the optimal request routing by ILP:
if isdebug
    EPS = 10^(-4);
    soln_f_init = soln_f; soln_val_init = soln_val; 
    [soln_f,soln_val] = RR_ILP_general(soln_m,soln_a, n_client,n_server,n_request_client,lc,RTT,tau,L,sm,sc,M,E,V,R,link,link_list,soln_f_init);
    if abs(soln_val_init - soln_val) > EPS
        error(['BPRR_heuristic_general: shortest path routing achieves ' num2str(soln_val_init/R/lc) ' ms per token, ILP routing achieves ' num2str(soln_val/R/lc) ' ms per token']);
    end
end


end

function [soln_f,soln_val] = shortest_path_given_BP(soln_m,soln_a, n_client,n_server,n_request_client,lc,RTT,tau,L,E,R,link)
% Assuming each server under block placement (soln_m,soln_a) has enough
% memory for attention caches (i.e., can ignore memory capacity
% constraints) -> problem reduces to shortest path routing from c to c'
delta = 1; % a positive placeholder "cost" for the last hop, as zero-cost links will be treated as no link by Dijkstra_source
soln_val = 0; 
soln_f = zeros(E,R); 
for c=1:n_client    
    G = zeros(n_server+2); % directed adjacency matrix for servers, c (node n_server+1), and c' (node n_server+2); if G(i,j) > 0, G(i,j) is the delay on "link" (i,j), including communication and processing at j; G(i,j) = 0 means "link" (i,j) does not exist
    for i=find(soln_a'<=1) % for each server hosting the first block
        G(n_server+1,i) = RTT(c,i) + tau(i)*soln_m(i); 
    end
    for i=find((soln_a+soln_m)'>L) % for each server hosting the last block
        G(i,n_server+2) = delta; 
    end
    for i=1:n_server
        next_block = soln_a(i)+soln_m(i); 
        for j=find(soln_a'<=next_block & (soln_a+soln_m)'>next_block) % for all link (i,j) that can be traversed
            G(i,j) = RTT(c,j) + tau(j)*(soln_a(j)+soln_m(j)-soln_a(i)-soln_m(i));
        end
    end
    [spcost,sp] = Dijkstra_source(G, n_server+1); 
    path = sp{n_server+2}; % path as node sequence: n_server+1, (server indices), n_server+2
    soln_val = soln_val + n_request_client * (spcost(n_server+2)-delta);
    % soln_val = soln_val + (spcost(n_server+2)-delta);% sum up the per-token inference time across requests 
    requests = (c-1)*n_request_client + 1:n_request_client; % range of requests belonging to client c, all routed to the same path 
    soln_f(link(c,path(2)+n_client),requests) = 1; % first hop
    for i=2:length(path)-2 % middle hops
        soln_f(link(path(i)+n_client,path(i+1)+n_client), requests) = 1; 
    end
    soln_f(link(path(end-1)+n_client,c+n_client+n_server), requests) = 1; % last hop
end
soln_val = soln_val*lc; % total inference time along the chosen paths, summed over all tokens and all requests


end