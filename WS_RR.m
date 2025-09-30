function [soln_f, soln_val, ttft, Waiting_time, R] = WS_RR(soln_a, soln_m, inter_request_time,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M, E, link, allocation_delay_prop)
% Waiting-penalized Shortest-path Request Routing:
% Note: We have only implemented the case of n_client = 1.
% soln_f: routing for each request
% soln_val: total (i.e., sum) completion time of all requests (the
% completion time for each request is the time from its arrival till its
% completion); soln_val/length(inter_request_time)/lc is the average time
% per token among all the generated tokens for all the requests.
% ttft: average Time-to-First-Token (TTFT) over all requests
% R: maximum #concurrently active requests
% We also record average waiting time for each request for insight:
% Waiting_time

n_client = size(RTT,1); % RTT(c,j): per-token RTT between each client c and each server j
n_server = size(RTT,2); 
n_requests = length(inter_request_time); % total #requests (arriving sequentially)
n_caches = floor((M-soln_m*sm)./sc); % n_server*1 array, n_caches(j) is total #attention cache slots on server j

R = 0; % max #concurrent requests
soln_val = 0; % total (i.e., sum) completion time over all the requests; each completion time is the time from a request arrival till its completion
ttft = 0; 
Waiting_time = 0; 
soln_f = zeros(E,n_requests); 

% assuming time starts at 0:
t = 0; % current time
delta = 1; % a positive placeholder "cost" for the last hop, as zero-cost links will be treated as no link by Dijkstra_source
completion_time = zeros(1,n_requests); % estimated completion time (since t = 0) for each request
state_time = zeros(n_server,n_requests); % state_time(j,r): completion time of request r on server j (0 if r does not traverse server j)
state_memory = zeros(n_server,n_requests); % state_memory(j,r): #attention caches hosted by server j for request r
c = 1; % assuming only one client; if more than one client, must also input the client who sends each request
for r=1:n_requests
    t = t + inter_request_time(r); % arrival time of request r
    R = max(R, 1+sum(completion_time>t)); % the max #concurrent requests must be achieved upon a new arrival; at arrival of request r, we have sum(completion_time>t) existing active requests plus the new request
    G = zeros(n_server+2); % directed adjacency matrix for servers, c (node n_server+1), and c' (node n_server+2); if G(i,j) > 0, G(i,j) is the delay on "link" (i,j), including communication and processing at j; G(i,j) = 0 means "link" (i,j) does not exist
    G_1 = zeros(size(G)); % like G, but only include the communication&computation time for generating the first token (used to evaluate TTFT)
    G_w = zeros(size(G)); % G_w(i,j) is the waiting time at link (i,j)
    for i=find(soln_a'<=1) % for each server hosting the first block
        active_requests = find(state_time(i,:)>t); % set of active requests scheduled on server i
        if n_caches(i) - sum(state_memory(i,active_requests)) >= soln_m(i)
            t_w = t; % earliest starting time if routed through server i
        else
            k = 1;
            [~,I] = sort(state_time(i,active_requests)); % active_requests(I(k+1:end)) are indices of active requests on server i in increasing completion time
            while n_caches(i) - sum(state_memory(i,active_requests(I(k+1:end)))) < soln_m(i)
                k = k + 1;
            end
            t_w = state_time(i,active_requests(I(k))); 
        end        
        G(n_server+1,i) = (t_w-t) + (RTT_input(c,i) + (lc-1)*RTT(c,i)) + (tau_input(i) + (lc-1)*tau(i))*soln_m(i); % waiting-penalized link cost
        G_1(n_server+1,i) = (t_w-t) + RTT_input(c,i) + tau_input(i)*soln_m(i);
        G_w(n_server+1,i) = t_w-t; 
    end
    for i=find((soln_a+soln_m)'>L) % for each server hosting the last block
        G(i,n_server+2) = delta; 
        % G_w(i,n_server+2) = 0; % no need to wait since there is no
        % processing at node 'n_server+2'
    end
    for i=1:n_server
        next_block = soln_a(i)+soln_m(i); 
        for j=find(soln_a'<=next_block & (soln_a+soln_m)'>next_block) % for all link (i,j) that can be traversed
            active_requests = find(state_time(j,:)>t); % active requests on server j
            if n_caches(j) - sum(state_memory(j,active_requests)) >= (soln_a(j)+soln_m(j)-soln_a(i)-soln_m(i))
                t_w = t;
            else
                k = 1;
                [~,I] = sort(state_time(j,active_requests)); 
                while n_caches(j) - sum(state_memory(j,active_requests(I(k+1:end)))) < (soln_a(j)+soln_m(j)-soln_a(i)-soln_m(i))
                    k = k + 1;
                end
                t_w = state_time(j,active_requests(I(k))); 
            end
            G(i,j) = (t_w-t) + (RTT_input(c,j) + (lc-1)*RTT(c,j)) + (tau_input(j) + (lc-1)*tau(j))*(soln_a(j)+soln_m(j)-soln_a(i)-soln_m(i));
            G_1(i,j) = (t_w-t) + RTT_input(c,j)  + tau_input(j) *(soln_a(j)+soln_m(j)-soln_a(i)-soln_m(i));
            G_w(i,j) = t_w-t; 
        end
    end
    [~,sp] = Dijkstra_source(G, n_server+1); 
    path = sp{n_server+2}; % path as node sequence: n_server+1, (server indices), n_server+2
    time_r = 0; % time from arrival till completion for request r
    ttft_r = 0; % TTFT for request r
    waiting_time = 0; 
    soln_f(link(c,path(2)+n_client),r) = 1; % first hop: traverse server path(2)
    waiting_time = max(waiting_time, G_w(n_server+1,path(2)));
    time_r = time_r + G(n_server+1,path(2)) - G_w(n_server+1,path(2)); % total communication+computation time at server path(2)
    ttft_r = ttft_r + G_1(n_server+1,path(2)) - G_w(n_server+1,path(2));
    for i=2:length(path)-2 % middle hops: traverse servers (path(i),path(i+1))
        soln_f(link(path(i)+n_client,path(i+1)+n_client), r) = 1; 
        waiting_time = max(waiting_time, G_w(path(i),path(i+1))); 
        time_r = time_r + G(path(i),path(i+1)) - G_w(path(i),path(i+1)); % total communication+computation time at server path(i+1)
        ttft_r = ttft_r + G_1(path(i),path(i+1)) - G_w(path(i),path(i+1));
    end
    soln_f(link(path(end-1)+n_client,c+n_client+n_server), r) = 1; % last hop
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
    time_r = time_r + waiting_time + initial_delay+ allocation_delay_prop;
    ttft_r = ttft_r + waiting_time + initial_delay+ allocation_delay_prop; 
    % update state variables:
    completion_time(r) = t + time_r; % time (since t = 0) that request r completes
    state_time(path(2:end-1),r) = completion_time(r); 
    state_memory(path(2),r) = soln_m(path(2));
    for i=2:length(path)-2
        state_memory(path(i+1),r) = soln_a(path(i+1))+soln_m(path(i+1))-soln_a(path(i))-soln_m(path(i));
    end
    soln_val = soln_val + time_r; 
    ttft = ttft + ttft_r; 
    Waiting_time = Waiting_time + waiting_time;
    % disp(['WS: client ' num2str(c) ', request ' num2str(r) ' is routed to server chain ' sprintf('%d,',path(2:end-1))])
end
ttft = ttft / n_requests; 
Waiting_time = Waiting_time / n_requests;


end