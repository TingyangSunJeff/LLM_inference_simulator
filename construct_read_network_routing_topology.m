function [servers, clients, RTT_raw, RTT, RTT_input, server_types, server_types_re] = construct_read_network_routing_topology(file_path, C, num_clients, high_perf_fraction, overhead_delay, overhead_delay_input)
    % Read the .graph file
    fid = fopen(file_path, 'r');
    if fid == -1
        error('Unable to open file: %s', file_path);
    end
    
    % Initialize variables
    node_list = [];
    edge_list = [];
    line = fgetl(fid);
    
    % Parse the file line by line
    while ischar(line)
        tokens = strsplit(strtrim(line));
        
        % Parse nodes
        if strcmp(tokens{1}, 'NODES')
            V = str2double(tokens{2});
            node_list = zeros(V, 3); % Store node info: [index, x, y]
            fgetl(fid); % Skip the title line "label x y"
            for i = 1:V
                line = fgetl(fid);
                tokens = strsplit(strtrim(line));
                node_list(i, :) = [str2double(tokens{1}) + 1, str2double(tokens{2}), str2double(tokens{3})]; % Adjust for 1-based indexing
            end
            
        % Parse edges
        elseif strcmp(tokens{1}, 'EDGES')
            E = str2double(tokens{2});
            edge_list = zeros(E, 5); % Store edge info: [src, dest, weight, bw, delay]
            fgetl(fid); % Skip the title line "label src dest weight bw delay"
            for i = 1:E
                line = fgetl(fid);
                tokens = strsplit(strtrim(line));
                edge_list(i, :) = [str2double(tokens{2}) + 1, str2double(tokens{3}) + 1, str2double(tokens{4}), ...
                                   str2double(tokens{5}), str2double(tokens{6})]; % Adjust for 1-based indexing
            end
        end
        line = fgetl(fid);
    end
    fclose(fid);
    
    % check range of link bandwidth/delay (for topology statistics):
%     disp(['link bandwidth in [' num2str(min(edge_list(:,4))/10^6) ',' num2str(max(edge_list(:,4))/10^6) '] Gbps'])
%     disp(['link delay in [' num2str(min(edge_list(:,5))/10^3) ',' num2str(max(edge_list(:,5))/10^3) '] ms'])

    % Create adjacency matrix and degree array
    adj_matrix = inf(V, V); % Initialize adjacency matrix for Floyd-Warshall
    degree = zeros(V, 1); % Track node degrees
    
    for i = 1:E
        src = edge_list(i, 1); % Source node (already adjusted)
        dest = edge_list(i, 2); % Destination node (already adjusted)
        delay = edge_list(i, 5); % Edge delay in microseconds (µs)
        if src > 0 && dest > 0 && src <= V && dest <= V
            adj_matrix(src, dest) = delay / 1000; % Convert µs to ms
            adj_matrix(dest, src) = delay / 1000; % Symmetric graph, convert µs to ms
            degree(src) = degree(src) + 1; % Increment degree
            degree(dest) = degree(dest) + 1;
        else
            error('Invalid edge detected: src=%d, dest=%d', src, dest);
        end
    end
    
    % Set diagonal to zero (distance from a node to itself)
    for i = 1:V
        adj_matrix(i, i) = 0;
    end
    
    % Compute shortest paths (Floyd-Warshall)
    RTT_raw = adj_matrix; % Start with the adjacency matrix (already in ms)
    for k = 1:V
        for i = 1:V
            for j = 1:V
                RTT_raw(i, j) = min(RTT_raw(i, j), RTT_raw(i, k) + RTT_raw(k, j));
            end
        end
    end
    
    % Double RTT_raw to make it round-trip delays
    RTT_raw = 2 * RTT_raw; % Each value now represents the round-trip delay in ms
    
    % Select multiple clients randomly
    [~, sorted_indices] = sort(degree, 'ascend'); % Sort nodes by degree
    clients = sorted_indices(randperm(length(sorted_indices), num_clients)); % Select multiple clients randomly
    
    % Exclude clients from server selection
    remaining_indices = setdiff(sorted_indices, clients); % Exclude clients from server pool
    servers = remaining_indices(randperm(length(remaining_indices), C)); % Randomly select C servers
    % servers = sorted_indices(randperm(length(sorted_indices), C)); % Randomly select C servers

    % servers = remaining_indices(1:C); % Select the first C remaining nodes with the lowest degrees
    
    % Assign server types (high-performance or low-performance)
    n_high_perf = round(C * high_perf_fraction);
    server_types = [repmat("A100", n_high_perf, 1); repmat("MIG", C - n_high_perf, 1)];
    server_types = server_types(randperm(C)); % Randomly shuffle server types

    % Reorder RTT_raw based on clients and server types
    % Ensure clients are first, followed by MIG servers, then A100 servers
    mig_servers = servers(server_types == "MIG");
    a100_servers = servers(server_types == "A100");
    ordered_indices = [clients; mig_servers; a100_servers]; % Ordered indices
    server_types_re = [repmat("MIG", length(mig_servers), 1); repmat("A100", length(a100_servers), 1)];
    % disp(ordered_indices)
    % disp(RTT_raw)
    RTT_raw = RTT_raw(ordered_indices, ordered_indices); % Rearrange RTT_raw
    % disp(RTT_raw)
    % Compute RTT (for client-to-server communication of one token at a time)
%     overhead_delay = 0.01; % Example overhead in ms
    RTT = inf(num_clients, length(servers)); % Initialize RTT for client-server
    for c = 1:num_clients
        for s = 1:length(servers)
            RTT(c, s) = RTT_raw(c, num_clients + s) + overhead_delay; % Combine raw RTT and overhead delay
        end
    end
    % Compute RTT_input: for client-server communication of the input
    % sequence (for prefill)
    RTT_input = inf(num_clients, length(servers)); % Initialize RTT for client-server
    for c = 1:num_clients
        for s = 1:length(servers)
            RTT_input(c, s) = RTT_raw(c, num_clients + s) + overhead_delay_input; % Combine raw RTT and overhead delay
        end
    end
end