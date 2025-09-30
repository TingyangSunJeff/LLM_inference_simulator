function [soln_f_petals,soln_val_petals,ttft,Waiting_time] = Petals_optimized_routing(cache_bytes_per_block, soln_a_petals,soln_m_petals,  inter_request_time,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M,E,link, allocation_delay_petals)
% Throughput-based block placement as implemented in
% Petals + our optimized routing: 
% Given the block placement by Petals, optimize the request routing (in a
% myopic manner) by solving the MILP we formulate. 
% Note: We have only implemented the case of n_client = 1. 

% soln_f_petals: [|E|, #request]
% soln_a_petals (Block Start Position per Server): #server, 1
% soln_m_petals (Number of Blocks per Server): #server, 1
% soln_val_petals (Total Inference Time for Petals):
% ttft: average Time-to-First-Token (TTFT) over all requests

n_client = size(RTT,1); % RTT(c,j): per-token RTT between each client c and each server j
n_server = size(RTT,2); 
n_requests = length(inter_request_time); % total #requests (arriving sequentially)
% n_caches = floor((M-soln_m_petals*sm)./sc); % n_server*1 array, n_caches(j) is total #attention cache slots on server j
n_caches = floor((soln_m_petals*cache_bytes_per_block)./sc);
n_var = (n_server+2)^2+1; 

soln_f_petals = zeros(E,n_requests); % soln_f(l,r) = 1 iff (global) request r is routed on link_list(l,:)
soln_val_petals = 0; % total (i.e., sum) completion time over all the requests; each completion time is the time from a request arrival till its completion
ttft = 0; 
Waiting_time = 0; 
% assuming time starts at 0:
t = 0; % current time
delta = 1; % a positive placeholder "cost" for the last hop
completion_time = zeros(1,n_requests); % estimated completion time (since t = 0) for each request
state_time = zeros(n_server,n_requests); % state_time(j,r): completion time of request r on server j (0 if r does not traverse server j)
state_memory = zeros(n_server,n_requests); % state_memory(j,r): #attention caches hosted by server j for request r
c = 1; % assuming only one client; if more than one client, must also input the client who sends each request
for r=1:n_requests
    t = t + inter_request_time(r); % arrival time of request r
    G = zeros(n_server+2); % directed adjacency matrix for servers, c (node n_server+1), and c' (node n_server+2); if G(i,j) > 0, G(i,j) is the delay on "link" (i,j), including total communication and processing at j; G(i,j) = 0 means "link" (i,j) does not exist
    G_1 = zeros(size(G)); % like G, but only include the communication&computation time for the first token
    G_w = zeros(size(G)); % G_w(i,j) is the waiting time at link (i,j); G_w(i,j) = 0 means "link" (i,j) does not exist
    for i=find(soln_a_petals'<=1) % for each server hosting the first block
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
        G(n_server+1,i) = (RTT_input(c,i) + (lc-1)*RTT(c,i)) + (tau_input(i) + (lc-1)*tau(i))*soln_m_petals(i); % total comm.+comp. time at this link (\lmax t^c_{ij})
        G_1(n_server+1,i) = RTT_input(c,i)  + tau_input(i) *soln_m_petals(i); 
        G_w(n_server+1,i) = t_w-t; % waiting time at this link (t^w_{ij}(t))
    end
    for i=find((soln_a_petals+soln_m_petals)'>L) % for each server hosting the last block
        G(i,n_server+2) = delta; % so that each feasible routing link (i,j) will have G(i,j)>0
        G_1(i,n_server+2) = delta;
        % G_w(i,n_server+2) = 0; % no need to wait since there is no
        % processing at node 'n_server+2'
    end
    for i=1:n_server
        next_block = soln_a_petals(i)+soln_m_petals(i); 
        for j=find(soln_a_petals'<=next_block & (soln_a_petals+soln_m_petals)'>next_block) % for all link (i,j) that can be traversed
            active_requests = find(state_time(j,:)>t); % active requests on server j
            if n_caches(j) - sum(state_memory(j,active_requests)) >= (soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(i)-soln_m_petals(i))
                t_w = t;
            else
                k = 1;
                [~,I] = sort(state_time(j,active_requests)); 
                while n_caches(j) - sum(state_memory(j,active_requests(I(k+1:end)))) < (soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(i)-soln_m_petals(i))
                    k = k + 1;
                end
                t_w = state_time(j,active_requests(I(k))); 
            end
            G(i,j) = (RTT_input(c,j) + (lc-1)*RTT(c,j)) + (tau_input(j) + (lc-1)*tau(j))*(soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(i)-soln_m_petals(i));
            G_1(i,j) = RTT_input(c,j) + tau_input(j) *(soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(i)-soln_m_petals(i));
            G_w(i,j) = t_w-t; 
        end
    end
    % objective function:
    obj = zeros(n_var,1);
    obj_1 = zeros(n_var,1); % coefficient function for evaluating TTFT
    for i=1:n_server+2
        for j=1:n_server+2
            if G(i,j)>0 % if link (i,j) is a feasible routing link under block placement (soln_a_petals, soln_m_petals):
                obj(f_idx(i,j,n_server)) = G(i,j); 
                obj_1(f_idx(i,j,n_server)) = G_1(i,j); 
            end
        end
    end
    obj(tw_idx(n_server)) = 1; 
    obj_1(tw_idx(n_server)) = 1; 
    % inequality constraints:
    A = zeros((n_server+2)^2, n_var);
    for i=1:n_server+2
        for j=1:n_server+2
            if G(i,j)>0
                A(A_idx(i,j,n_server), f_idx(i,j,n_server)) = G_w(i,j); 
                A(A_idx(i,j,n_server), tw_idx(n_server)) = -1; 
            end
        end
    end
    b = zeros((n_server+2)^2, 1); 
    % equality constraints:
    Aeq = zeros(n_server+2, n_var);
    for j=1:n_server+2
        for i=1:n_server+2
            if G(j,i)>0
                Aeq(j, f_idx(j,i,n_server)) = 1;
            end
            if G(i,j)>0
                Aeq(j, f_idx(i,j,n_server)) = -1;
            end
        end
    end
    beq = zeros(n_server+2, 1);
    beq(n_server+1) = 1; beq(n_server+2) = -1; 
    % lower/upper bounds:
    lb = zeros(n_var,1);
    ub = ones(n_var,1); ub(tw_idx(n_server)) = inf; 
    % integer constraints:
    intcon = 1:n_var-1; 
    % call MILP solver:
    options = optimoptions('intlinprog','Display','none');
    % Gurobi's MILP solver:
    [soln,fval,~,~] = gurobi_intlinprog(obj,intcon,A,b,Aeq,beq,lb,ub,[],options); % return value: [x,fval,exitflag,output]
    % extract solution:
    % compute actual waiting time: time from arrival till the first
    % successful retry, with binary exponential backoff
    max_waiting_time = 60; % maximum backoff time in PETALS
    raw_waiting_time = soln(tw_idx(n_server)); % t^W is the raw waiting time for request r, defined as the time till memory is available at the entire server chain
    if raw_waiting_time>0 % if retry is needed:
        n_retry = 1; 
        t_retry = min(2^(n_retry-1),max_waiting_time)*1e3; % note: our time unit is 'ms'
        while t_retry<raw_waiting_time
            n_retry = n_retry+1;
            t_retry = t_retry + min(2^(n_retry-1),max_waiting_time)*1e3;
        end
        waiting_time = t_retry; % time until the first successful retry
    else
        waiting_time = 0; 
    end    
    time_r = fval + initial_delay - delta + waiting_time - raw_waiting_time + allocation_delay_petals; % time from arrival to completion for request r under optimal routing (i.e., myopic scheduling)
    ttft_r = obj_1'*soln + initial_delay - delta + waiting_time - raw_waiting_time + allocation_delay_petals; % TTFT for request r
    soln_val_petals = soln_val_petals + time_r; 
    ttft = ttft + ttft_r; 
    Waiting_time = Waiting_time + waiting_time; % t^W is the waiting time for request r

    istraversed = zeros(n_server,1);
    for i=1:n_server+2
        for j=1:n_server+2
            if G(i,j)>0 && soln(f_idx(i,j,n_server))>0 % if f_{ij} > 0
                if i == n_server+1 % i is the source
                    i1 = c; % global node index for i
                else
                    i1 = i+n_client; 
                end
                if j == n_server+2 % j is the destination
                    j1 = c+n_client+n_server; % global node index for j
                else
                    j1 = j+n_client; 
                end
                if ~(i1>=1 && i1<=2*n_client+n_server) || ~(j1>=1 && j1<=2*n_client+n_server) || link(i1,j1)==0
                    error(['global node indices: i1 = ' num2str(i1) ', j1 = ' num2str(j1)]);
                end
                soln_f_petals(link(i1,j1),r) = 1;
                if i<=n_server
                    istraversed(i) = 1;
                end
                if j<=n_server
                    istraversed(j) = 1;
                end
                if i == n_server+1 % if this is the first hop:
                    state_memory(j,r) = soln_m_petals(j); 
                elseif j<=n_server 
                    state_memory(j,r) = soln_a_petals(j)+soln_m_petals(j)-soln_a_petals(i)-soln_m_petals(i); 
                end
            end
        end
    end
    % update state variables:
    completion_time(r) = t + time_r; % time (since t = 0) that request r completes
    state_time(istraversed>0,r) = completion_time(r);   
end
ttft = ttft / n_requests; 
Waiting_time = Waiting_time / n_requests; 

end

%%%%%%%%%%%%%%%%%%%%%
function val = f_idx(i,j,n_server)
val = (i-1)*(n_server+2) + j; % index of f_{ij}
end
function val = tw_idx(n_server)
val = (n_server+2)^2 + 1; % index of t^W
end
function val = A_idx(i,j,n_server)
val = (i-1)*(n_server+2) + j; % row index of constraint (i,j) in coefficient matrix A
end