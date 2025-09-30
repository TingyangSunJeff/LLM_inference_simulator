function [soln_f_petals,soln_a_petals,soln_m_petals,soln_val_petals,ttft,Waiting_time,max_concurrent] = Petals_online(server_types, cache_bytes_per_block, num_key_value_groups, order,throughput,RTT_raw,overhead_delay_petals,alloc_delay,d_model, inter_request_time,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M,E,link, allocation_delay)
% Throughput-based block placement and greedy routing as implemented in
% Petals: 
% Different from 'Petals' which does one-time block placement and request
% routing, this implementation does one-time block placement + sequential request routing.
% Note: We have only implemented the case of n_client = 1. 

% soln_f_petals: [|E|, #request]
% soln_a_petals (Block Start Position per Server): #server, 1
% soln_m_petals (Number of Blocks per Server): #server, 1
% soln_val_petals (Total Inference Time for Petals):
% ttft: average Time-to-First-Token (TTFT) over all requests

n_client = size(RTT,1); % RTT(c,j): per-token RTT between each client c and each server j
n_server = size(RTT,2); 
n_requests = length(inter_request_time); % total #requests (arriving sequentially)

is_print = 0; 
% block placement:
soln_a_petals = zeros(n_server,1); 
soln_m_petals = zeros(n_server,1);
block_throughput = zeros(1,L); % block_throughput(b): total throughput of placed instances of block b
for i=1:n_server % simulate adding one server at a time to the swarm:
    s = order(i); % server s
    soln_m_petals(s) = choose_num_blocks_petals(M(s),d_model, sm, L, num_key_value_groups);
    if soln_m_petals(s) > 4 && server_types(s) == "MIG"
        soln_m_petals(s) = 4;
    end
    soln_a_petals(s) = block_selection_petals(block_throughput, soln_m_petals(s)); 
    block_throughput(soln_a_petals(s) : soln_a_petals(s)+soln_m_petals(s)-1) = block_throughput(soln_a_petals(s) : soln_a_petals(s)+soln_m_petals(s)-1) + compute_throughput(throughput, s, soln_m_petals(s)); 
end
% [soln_a_petals soln_a_petals+soln_m_petals-1] % each row shows the first and the last block placed at each server

% request routing:
n_caches = floor((soln_m_petals*cache_bytes_per_block)./sc); % n_server*1 array, n_caches(j) is total #attention cache slots on server j
session_capacity = floor((soln_m_petals*cache_bytes_per_block) ./ (soln_m_petals*sc)); % session_capacity(j): max #inference sessions server j can host (assuming each token needs to go through all hosted blocks)
soln_f_petals = zeros(E,n_requests); % soln_f(l,r) = 1 iff (global) request r is routed on link_list(l,:)
soln_val_petals = 0; % total (i.e., sum) completion time over all the requests; each completion time is the time from a request arrival till its completion
ttft = 0; % average time to first token; averaged over all requests
Waiting_time = 0; % average waiting time over all requests
% assuming time starts at 0:
t = 0; % current time
completion_time = zeros(1,n_requests); % estimated completion time (since t = 0) for each request
state_time = zeros(n_server,n_requests); % state_time(j,r): completion time of request r on server j (0 if r does not traverse server j)
state_memory = zeros(n_server,n_requests); % state_memory(j,r): #attention caches hosted by server j for request r
c = 1; % assuming only one client; if more than one client, must also input the client who sends each request

max_concurrent = 0; % Track max concurrent requests

for r=1:n_requests
    t = t + inter_request_time(r); % arrival time of request r
    max_concurrent = max(max_concurrent, 1+sum(completion_time>t));
    G = zeros(n_server+2); % directed adjacency matrix for servers, c (node n_server+1), and c' (node n_server+2); if G(i,j) > 0, G(i,j) is the delay on "link" (i,j), including communication and processing at j; G(i,j) = 0 means "link" (i,j) does not exist
    for i=find(soln_a_petals'<=1) % for each server containing the first block
        G(n_server+1,i) = RTT_raw(c,i+n_client)/2 + overhead_delay_petals + tau(i)*soln_m_petals(i); % remember that we count the processing at each server into the delay of its incoming links
        if session_capacity(i) - sum(state_time(i,:)>t) == 0 % ~has_cache_for_petals(cache_tokens_left(i),soln_m_petals(i),lc)
            G(n_server+1,i) = G(n_server+1,i) + alloc_delay;
        end
    end
    for i=find((soln_a_petals+soln_m_petals)'>L) % for each server containing the last block
        G(i,n_server+2) = RTT_raw(c,i+n_client)/2;
    end
    for i=1:n_server
        next_block = soln_a_petals(i)+soln_m_petals(i);
        for j=find(soln_a_petals'<=next_block & (soln_a_petals+soln_m_petals)' > next_block) % for all server j containing the next block after processing at server i
            G(i,j) = RTT_raw(i+n_client,j+n_client)/2 + overhead_delay_petals + tau(j)*(soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(i)-soln_m_petals(i));
            if session_capacity(j) - sum(state_time(j,:)>t) == 0 % ~has_cache_for_petals(cache_tokens_left(j),soln_m_petals(j),lc)
                G(i,j) = G(i,j) + alloc_delay;
            end
        end
    end
    [~, sp] = Dijkstra_source(G, n_server+1); % sp{n_server+2} is the selected path for request r of client c
    path = sp{n_server+2}; % as a node sequence: n_server+1, (server indices), n_server+2

    % simulate completion time and update state variables:
    time_r = 0; % time from arrival till completion for request r
    ttft_r = 0; % TTFT for  request r
    waiting_time = 0;
    soln_f_petals(link(c,path(2)+n_client),r) = 1; % first hop: traverse server path(2)
    i = path(2);
    active_requests = find(state_time(i,:)>t); % set of active requests scheduled on server i
    if n_caches(i) - sum(state_memory(i,active_requests)) >= soln_m_petals(i)
        t_w = t; % earliest starting time if routed through server i
    else
        k = 1;
        [~,I] = sort(state_time(i,active_requests)); % active_requests(I(k+1:end)) are indices of active requests on server i in increasing completion time
        while n_caches(i) - sum(state_memory(i,active_requests(I(k+1:end)))) < soln_m_petals(i)
            k = k + 1;
        end
        t_w = state_time(i,active_requests(I(k)));
    end
    waiting_time = max(waiting_time, t_w-t);
    time_r = time_r + (RTT_input(c,i) + (lc-1)*RTT(c,i)) + (tau_input(i) + (lc-1)*tau(i))*soln_m_petals(i); % total communication+computation time at server path(2)
    ttft_r = ttft_r + RTT_input(c,i) + tau_input(i) *soln_m_petals(i); % communication+computation time at server path(2) for the first token
    for i=2:length(path)-2 % middle hops: traverse servers (path(i),path(i+1))
        soln_f_petals(link(path(i)+n_client,path(i+1)+n_client), r) = 1;
        j = path(i+1);
        active_requests = find(state_time(j,:)>t); % active requests on server j
        if n_caches(j) - sum(state_memory(j,active_requests)) >= (soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(path(i))-soln_m_petals(path(i)))
            t_w = t;
        else
            k = 1;
            [~,I] = sort(state_time(j,active_requests));
            while n_caches(j) - sum(state_memory(j,active_requests(I(k+1:end)))) < (soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(path(i))-soln_m_petals(path(i)))
                k = k + 1;
            end
            t_w = state_time(j,active_requests(I(k)));
        end
        waiting_time = max(waiting_time, t_w-t);
        time_r = time_r + (RTT_input(c,j) + (lc-1)*RTT(c,j)) + (tau_input(j) + (lc-1)*tau(j))*(soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(path(i))-soln_m_petals(path(i))); % total communication+computation time at server path(i+1)
        ttft_r = ttft_r + RTT_input(c,j)  + tau_input(j) *(soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(path(i))-soln_m_petals(path(i))); 
    end
    soln_f_petals(link(path(end-1)+n_client,c+n_client+n_server), r) = 1; % last hop
    % compute actual waiting time: time from arrival till the first
    % successful retry, with binary exponential backoff
    max_waiting_time = 60; % maximum backoff time in PETALS
    raw_waiting_time = waiting_time; % time till memory is available at the entire server chain
    if raw_waiting_time>0 % if retry is needed:
        n_retry = 1; 
        t_retry = min(2^(n_retry-1),max_waiting_time)*1e3; % note: our time unit is 'ms'
        while t_retry<raw_waiting_time
            n_retry = n_retry+1;
            t_retry = t_retry + min(2^(n_retry-1),max_waiting_time)*1e3;
        end
        waiting_time = t_retry; % time until the first successful retry
    end
    time_r = time_r + waiting_time+ initial_delay+ allocation_delay;
    ttft_r = ttft_r + waiting_time+ initial_delay+ allocation_delay;
    % update state variables:
    completion_time(r) = t + time_r; % time (since t = 0) that request r completes
    state_time(path(2:end-1),r) = completion_time(r);
    state_memory(path(2),r) = soln_m_petals(path(2));
    for i=2:length(path)-2
        state_memory(path(i+1),r) = soln_a_petals(path(i+1))+soln_m_petals(path(i+1))-soln_a_petals(path(i))-soln_m_petals(path(i));
    end
    soln_val_petals = soln_val_petals + time_r;
    ttft = ttft + ttft_r; 
    Waiting_time = Waiting_time + waiting_time; 

    if is_print
        disp(['Petals: client ' num2str(c) ', request ' num2str(r) ' is routed to server chain ' sprintf('%d,',path(2:end-1))])
    end
end
ttft = ttft / n_requests; 
Waiting_time = Waiting_time / n_requests; 

end