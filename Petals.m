function [soln_f_petals,soln_a_petals,soln_m_petals,soln_val_petals] = Petals(num_key_value_groups, order,throughput,RTT_raw,overhead_delay_petals,alloc_delay,d_model, n_client,n_server,n_request_client,lc,RTT,tau,L,sm,sc,M,E,R,link,link_list)
% Throughput-based block placement and greedy routing as implemented in
% Petals:
% soln_f_petals: [|E|, #request]
% soln_a_petals (Block Start Position per Server): #server, 1
% soln_m_petals (Number of Blocks per Server): #server, 1
% soln_val_petals (Total Inference Time for Petals):

is_print = 0; 
% block placement:
soln_a_petals = zeros(n_server,1); 
soln_m_petals = zeros(n_server,1);
block_throughput = zeros(1,L); % block_throughput(b): total throughput of placed instances of block b
for i=1:n_server % simulate adding one server at a time to the swarm:
    s = order(i); % server s
    soln_m_petals(s) = choose_num_blocks_petals(M(s),d_model, sm, L, num_key_value_groups);
    soln_a_petals(s) = block_selection_petals(block_throughput, soln_m_petals(s)); 
    block_throughput(soln_a_petals(s) : soln_a_petals(s)+soln_m_petals(s)-1) = block_throughput(soln_a_petals(s) : soln_a_petals(s)+soln_m_petals(s)-1) + throughput(s); 
end
% [soln_a_petals soln_a_petals+soln_m_petals-1] % each row shows the first and the last block placed at each server

% request routing:
cache_left = floor((M - soln_m_petals*sm) ./ (soln_m_petals*sc)); % cache_left(j): remaining #inference sessions server j can host (assuming each token needs to go through all hosted blocks)
soln_f_petals = zeros(E,R); % soln_f(l,r) = 1 iff (global) request r is routed on link_list(l,:)
for c = 1:n_client
    for r = 1:n_request_client
        G = zeros(n_server+2); % directed adjacency matrix for servers, c (node n_server+1), and c' (node n_server+2); if G(i,j) > 0, G(i,j) is the delay on "link" (i,j), including communication and processing at j; G(i,j) = 0 means "link" (i,j) does not exist
        for i=find(soln_a_petals'<=1) % for each server containing the first block
            G(n_server+1,i) = RTT_raw(c,i+n_client)/2 + overhead_delay_petals + tau(i)*soln_m_petals(i); % remember that we count the processing at each server into the delay of its incoming links
            if cache_left(i) == 0 % ~has_cache_for_petals(cache_tokens_left(i),soln_m_petals(i),lc)
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
                if cache_left(j) == 0 % ~has_cache_for_petals(cache_tokens_left(j),soln_m_petals(j),lc)
                    G(i,j) = G(i,j) + alloc_delay;
                end
            end
        end
        [~, sp] = Dijkstra_source(G, n_server+1); % sp{n_server+2} is the selected path for request r of client c
        path = sp{n_server+2}; % as a node sequence: n_server+1, (server indices), n_server+2
        cache_left(path(2:end-1)) = cache_left(path(2:end-1)) - 1; % decrement remaining #sessions for each server on the selected chain
        r_global = (c-1)*n_request_client + r; % global request index of this request
        soln_f_petals(link(c,path(2)+n_client),r_global) = 1; % first hop (note the conversion of node indices to our node indices in V)
        for i=2:length(path)-2 % middle hops
            soln_f_petals(link(path(i)+n_client, path(i+1)+n_client), r_global) = 1;
        end
        soln_f_petals(link(path(end-1)+n_client, c+n_client+n_server), r_global) = 1; % last hop
        if is_print
            disp(['Petals: client ' num2str(c) ', request ' num2str(r) ' is routed to server chain ' sprintf('%d,',path(2:end-1))])
        end
    end
end

[soln_val_petals] = total_inference_time(soln_f_petals,soln_a_petals,soln_m_petals, n_client,n_server,n_request_client,lc,RTT,tau,sm,sc,M,R,link_list); 


end