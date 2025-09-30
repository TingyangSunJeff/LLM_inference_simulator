% General case:
file_path = 'topology/Bellcanada.graph';

% cluster size and demand:
jsonText = fileread('throughput_v5_large.json');
data = jsondecode(jsonText);
a100_key = 'bloom_device_NVIDIA_A100_80GB_PCIe_GPU_dtype_bfloat16';
mig_key = 'bloom_device_NVIDIA_MIG_1g_GPU_dtype_bfloat16';
if isfield(data, a100_key) && isfield(data, mig_key)
    a100_forward_rps = data.(a100_key).forward_rps; % Extract A100 throughput value for remote servers
    a100_network_rps = data.(a100_key).network_rps;
    mig_forward_rps = data.(mig_key).forward_rps; 
    mig_network_rps = data.(mig_key).network_rps; % Extract MIG throughput value for local servers
else
    error('Required throughput data for A100 or MIG not found in the JSON file.');
end

overhead_delay_petals = 18; % serialization overhead by Petals (in ms)
alloc_delay = 10000; % delay penalty if not enough cache space by Petals
realnet = 1; % 0: clustered setting; 1: scatter setting according to real topologies
num_MC_runs = 1; %100; % Number of Monte Carlo runs
% Proposed:
Time_proposed_inference_time = zeros(1, num_MC_runs); % overall average per-token generation time
Time_proposed_ttft = zeros(1, num_MC_runs); % average time to first token (TTFT), averaged over requests
Time_proposed_waiting = zeros(1, num_MC_runs); % average waiting time, avreaged over requests
Time_proposed_runtime = zeros(1, num_MC_runs); % For Proposed method running time
% Petals:
Time_petals_inference_time = zeros(1, num_MC_runs);
Time_petals_ttft = zeros(1, num_MC_runs);
Time_petals_waiting = zeros(1, num_MC_runs);
Time_petals_runtime = zeros(1, num_MC_runs); % For PETALS running time
% Petals with optimized order:
Time_petals_optimized_order_inference_time = zeros(1, num_MC_runs);
Time_petals_optimized_order_ttft = zeros(1, num_MC_runs);
Time_petals_optimized_order_waiting = zeros(1, num_MC_runs);
Time_petals_optimized_order_runtime = zeros(1, num_MC_runs);
% Petals with optimized request routing:
Time_petals_optimized_rr_inference_time = zeros(1, num_MC_runs);
Time_petals_optimized_rr_ttft = zeros(1, num_MC_runs);
Time_petals_optimized_rr_waiting = zeros(1, num_MC_runs);
Time_petals_optimized_rr_runtime = zeros(1, num_MC_runs);
% rng(40, 'twister');
n_requests = 50;
lambda = 0.0005; % request arrival rate (unit: requests/ms)
lc = 128; % 20; % #tokens per request

for mc_run = 1:num_MC_runs
    disp(['run ' num2str(mc_run) ':'])
    if realnet
        % Real network setup
        C = 26;
        high_perf_fraction = 0.2;        
        lc_in = 20; % input sequence length (should use lc_in to select the configuration of input-dependent parameters, e.g., overhead_delay_input, tau_input, initial_delay)
        overhead_delay = 100; % overhead for per-token RTT (all time unit is ms)
        overhead_delay_input =  0.7049 * lc_in + 67; % overhead for per-input RTT (for generating the first token) (this depends on input length!)
        initial_delay = 70; % initialization delay (this depends on input length!)
        allocation_delay_petals = 35e4;
        allocation_delay_prop = 60e3;
        [servers, clients, RTT_raw, RTT, RTT_input, server_types_old, server_types] = construct_read_network_routing_topology(file_path, C, 1, high_perf_fraction, overhead_delay, overhead_delay_input);
        % Assign server parameters
        n_server = length(servers);
        n_client = length(clients);
        
        % Initialize server parameters
        throughput_forward = zeros(n_server, 1);
        throughput_network = zeros(n_server, 1);
        tau = zeros(n_server, 1);
        tau_input = zeros(n_server, 1);
        M = zeros(n_server, 1);
        
        % Assign values based on server types
        for i = 1:n_server
            if server_types(i) == "A100"
                throughput_forward(i) = a100_forward_rps;
                throughput_network(i) = a100_network_rps;
                tau(i) = 3.2197; % High-performance inference time
                tau_input(i) = 7.43e-02 * lc_in + 32.99; % per-block prefill time (depends on input length!)
                M(i) = 84.973; % High-performance memory (GB)
            elseif server_types(i) == "MIG"
                throughput_forward(i) = mig_forward_rps;
                throughput_network(i) = mig_network_rps;
                tau(i) = 16.2488; % Low-performance inference time
                tau_input(i) = 0.1041 * lc_in + 206.3547; % per-block prefill time (depends on input length!)
                M(i) = 10.2; % Low-performance memory (GB)
            else
                error('Unknown server type: %s', server_types(i));
            end
        end

        throughput = [throughput_forward, throughput_network];
    end
model = 'BLOOM-176B'; 
switch model
    case 'BLOOM-176B'
        L = 70; % total #blocks (layers in the LLM)
        n_parameters = 2466437120; % #parameters per block (in billion)
        d_model = 14336; % dimension per embedding
        lmax = 2048; % max sequence length (including input tokens)
        num_key_value_groups = 1;
    case 'BLOOM-7B'
        L = 30;
        n_parameters = 0.23; % billion
        d_model = 4096;
        lmax = 2048; 
    otherwise
        error(['unsupported model: ' model]);
end

if num_key_value_groups > 1
    attn_cache_tokens = 16384; 
else
    attn_cache_tokens = 4096;  
end
cache_bytes_per_block = 2 * d_model * attn_cache_tokens * 2; % upper bound on sc, #bytes of attention cache per block per session
cache_bytes_per_block = floor(cache_bytes_per_block / num_key_value_groups) /10^9;
bytes_per_value = 4.25 / 8; %nf4
sm = n_parameters * bytes_per_value * 1.01 /10^9; % size of each block (GB); assuming 4-bit precision (e.g., nf4)
sc = 2*d_model*(lc+lc_in)*2/10^9; % size of each attention cache (per block per session) (GB)

% construct logical routing topology:
[E,V,link,link_list] = construct_routing_topology(n_client,n_server);
% set target concurrent #requests:
[time_per_request] = BPRR_heuristic_general_extended(1,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M); % estimate the time for a single request (without waiting)
mean_requests = lambda*time_per_request;
std_requests = sqrt(lambda*time_per_request); 
num_std = 1; % how many std is the confidence interval? (no more than 4)
R_max = floor((sum(M)-sm*(L+n_server))/sc/(L+n_server));
R = min(round(mean_requests + num_std*std_requests), R_max); 

% check feasibility:
% tmp_m = floor(M./sm); % maximum #blocks per server
tmp_m = zeros(n_server,1);
for i=1:n_server 
    tmp_m(i) = choose_num_blocks_proposed(M(i),sm,sc,R, L);
end
disp(['L = ' num2str(L) ', total #blocks placed by CG-BP = ' num2str(sum(tmp_m))]);
% tmp_sessions = floor((M-sm.*tmp_m)./(sc.*tmp_m)); % lower bound on #sessions each server can handle
% if sum(tmp_m) < L || sum(tmp_sessions.*tmp_m) < R*L % 'sum(tmp_sessions.*tmp_m)': total #attention caches the servers can hold after storing the maximum #blocks per server (each session requires one such cache per processed block); 'R*L' is the total number of attention caches needed to process R sessions
%     error('test_general_case: Insufficient capacity!');
% end
% disp(['R = ' num2str(floor(sum(tmp_sessions.*tmp_m)/L))])
% max #requests R is related to lc due to feasibility constraint:
% R = 41 if lc = 100
% R = 102 if lc = 80
% R = 115 if lc = 60
% R = 139 if lc = 40
% R = 213 if lc = 20

% pre-generate request arrival process (to ensure same input to all
% algorithms):
inter_request_time = exprnd(1/lambda, 1, n_requests); inter_request_time(1) = 0; % inter_request_time(r): time interval between r-1 and r-th requests (ms)
writematrix(inter_request_time, 'inter_arrivals_large.txt', 'Delimiter', 'tab');
%% proposed solution: joint block placement and request routing 
disp('=== Proposed solution ===');
% MAX_MILP_SIZE = 10^7; 
% milp_size = (2*n_server + 14*R*E)*(R*E*5+n_server*2); 
% if milp_size <= MAX_MILP_SIZE % only run MILP if the required memory (represented by size of the inequality coefficient matrix) is small enough (set of a billion)
%     [soln_f,soln_a,soln_m,soln_val] = BPRR_MILP_general(n_client,n_server,n_request_client,lc,RTT,tau,L,sm,sc,M,E,V,R,link,link_list);
%     Time_proposed = soln_val/R/lc; % average inference time per token (averaged over all the requests); in 'ms'
%     disp(['Proposed MILP: average time per token = ' num2str(soln_val/R/lc) ' ms'])
% end
% % sanity check: soln_val should be the same
% [soln_val] = total_inference_time(soln_f,soln_a,soln_m, n_client,n_server,n_request_client,lc,RTT,tau,sm,sc,M,R,link_list);
% % Best objective 1.980000000000e+04, best bound 1.209264224128e+04, gap 38.9260%
% % 
% num_repeats_proposed = 5;
% times_proposed = zeros(1, num_repeats_proposed);

% R = 136;

tstart_proposed = tic;
[soln_a_heu,soln_m_heu,soln_val_upper, optimized_order] = CG_BP(RTT,tau,L,sm,sc,M,R, server_types); % block placement; note: 'soln_val_upper' upper-bounds per-token inference time, assuming no more than R requests concurrently and no initial delay or extra delays for first token
[soln_f_heu, soln_val_heu, ttft_heu, waiting_heu, R_concurrent] = WS_RR(soln_a_heu, soln_m_heu, inter_request_time,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M, E, link,  allocation_delay_prop); 
Time_proposed_runtime(mc_run) = toc(tstart_proposed);

% avg_time_proposed = mean(times_proposed);
% std_time_proposed = std(times_proposed);
Time_proposed_heu = soln_val_heu / (n_requests * lc);  % average time per token 
Time_proposed_inference_time(mc_run) = Time_proposed_heu;
Time_proposed_ttft(mc_run) = ttft_heu; 
Time_proposed_waiting(mc_run) = waiting_heu; 
% disp('=== Proposed Heuristic (BPRR_heuristic_general) ===');
% disp('Execution times for each repeated run (s):');
% disp(times_proposed);
% disp(['Mean run time: ' num2str(avg_time_proposed) ' s,  Std: ' num2str(std_time_proposed) ' s']);
disp(['Proposed: average time per token = ' num2str(Time_proposed_heu) ' ms, average TTFT = ' num2str(ttft_heu) ' ms, average waiting per request = ' num2str(waiting_heu) ' ms, running time = ' num2str(Time_proposed_runtime(mc_run)) ' s, planned #concurrent requests = ' num2str(R) ', actual max #concurrent requests = ' num2str(R_concurrent)]);

%% petals:
disp('=== Petals Approach ===');
% Order = [1:n_local_server n_local_server+1:n_server;...
%     n_local_server+1:n_server 1:n_local_server];
% server_orders = {
%     [2, 9, 3, 7, 6, 5, 1, 4, 8]; 
%     [3, 1, 7, 9, 6, 4, 8, 2, 5]; 
%     [2, 8, 7, 9, 6, 3, 4, 5, 1]; 
%     [5, 1, 6, 3, 2, 7, 4, 9, 8]; 
%     [2, 3, 5, 4, 1, 7, 9, 6, 8]
% };

% num_formal_orders = 5;
% server_orders = cell(1, num_formal_orders);
% for i = 1:num_formal_orders
order = randperm(n_server); % Randomize order of server indices
% disp(order)
% end
% Set runs equal to the number of server orders
% runs = length(server_orders);
% Time_petals_inference = zeros(1, runs); 
% Time_petals_runtime = zeros(1, runs);

% for r_indx = 1:runs
    % disp(['=== Petals run ' num2str(r_indx) ' ===']);
% order = server_orders{r_indx};

tstart_petals = tic;
[soln_f_petals,soln_a_petals,soln_m_petals,soln_val_petals,ttft_petals, waiting_petals, max_concurrent] = Petals_online(server_types, cache_bytes_per_block, num_key_value_groups, order,throughput,RTT_raw,overhead_delay_petals,alloc_delay,d_model, inter_request_time,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M,E,link, allocation_delay_petals);
Time_petals_runtime(mc_run) = toc(tstart_petals);
inference_petals = soln_val_petals / (n_requests * lc);
Time_petals_inference_time(mc_run) = inference_petals;
Time_petals_ttft(mc_run) = ttft_petals; 
Time_petals_waiting(mc_run) = waiting_petals; 
    % disp(['  servers join order: ' num2str(order) ...
    %       ', avg time per token = ' num2str(Time_petals_inference(r_indx)) ' ms']);
    % disp(['  Petals code time = ' num2str(Time_petals_runtime(r_indx)) ' s']);

    %--- (Commented) ablation / ILP routing:
    % [soln_f_RR, soln_val_RR] = RR_ILP_general(...);
    % if ~isempty(soln_f_RR)
    %     ...
    % end
% end

% avg_time_petals = mean(Time_petals_runtime);
% std_time_petals = std(Time_petals_runtime);

% % disp('Execution times for each run (s):');
% % disp(Time_petals_runtime);
% % disp(['Mean run time: ' num2str(avg_time_petals) ' s,  Std: ' num2str(std_time_petals) ' s']);
% disp('Per-run average time per token (ms):');
% disp(inference_petals);
disp(['Petals: average time = ' num2str(inference_petals) ' ms, average TTFT = ' num2str(ttft_petals) ' ms, average waiting = ' num2str(waiting_petals) ' ms, running time = ' num2str(Time_petals_runtime(mc_run)) ' s, max # concurrent requests = ' num2str(max_concurrent)]);

%%
disp('=== Petals - optimized order ===');
% Petals - optimized order
tstart_optimized_order = tic;
[soln_f_petals_optimized, soln_a_petals_optimized, soln_m_petals_optimized, soln_val_petals_optimized, ttft_petals_optimized, waiting_petals_optimized, max_concurrent] = Petals_online(server_types, cache_bytes_per_block, num_key_value_groups, optimized_order,throughput,RTT_raw,overhead_delay_petals,alloc_delay,d_model, inter_request_time,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M,E,link, allocation_delay_petals);
Time_petals_optimized_order_runtime(mc_run) = toc(tstart_optimized_order);
Time_petals_optimized_order_inference_time(mc_run) = soln_val_petals_optimized / (n_requests * lc);
Time_petals_optimized_order_ttft(mc_run) = ttft_petals_optimized; 
Time_petals_optimized_order_waiting(mc_run) = waiting_petals_optimized; 

disp(['Petals - optimized order: average time per token = ' num2str(Time_petals_optimized_order_inference_time(mc_run)) ' ms, average TTFT = ' num2str(ttft_petals_optimized) ' ms, running time = ' num2str(Time_petals_optimized_order_runtime(mc_run)) ' s, max # concurrent requests = ' num2str(max_concurrent)]);

%%
disp('=== Petals - optimize request routing  ===');
% Petals - optimize request routing 
tstart_petals_rr = tic;
[soln_f_optimized_rr, soln_val_optimized_rr, ttft_optimized_rr, waiting_optimized_rr] = Petals_optimized_routing(cache_bytes_per_block, soln_a_petals,soln_m_petals,  inter_request_time,lc,initial_delay,RTT,RTT_input,tau,tau_input,L,sm,sc,M,E,link, allocation_delay_petals);
Time_petals_optimized_rr_runtime(mc_run) = toc(tstart_petals_rr);
Time_petals_optimized_rr_inference_time(mc_run) = soln_val_optimized_rr / (n_requests * lc);
Time_petals_optimized_rr_ttft(mc_run) = ttft_optimized_rr; 
Time_petals_optimized_rr_waiting(mc_run) = waiting_optimized_rr; 

disp(['Petals - optimized request routing: average time per token = ' num2str(Time_petals_optimized_rr_inference_time(mc_run) ) ' ms, average TTFT = ' num2str(ttft_optimized_rr) ' ms, running time = ' num2str(Time_petals_optimized_rr_runtime(mc_run)) ' s']);

disp(' ')
end


% Calculate average and standard deviation for Proposed
average_time_proposed = mean(Time_proposed_inference_time);
std_time_proposed = std(Time_proposed_inference_time);
average_ttft_proposed = mean(Time_proposed_ttft);
std_ttft_proposed = std(Time_proposed_ttft);
average_waiting_proposed = mean(Time_proposed_waiting);
std_waiting_proposed = std(Time_proposed_waiting);
average_runtime_proposed = mean(Time_proposed_runtime);
std_runtime_proposed = std(Time_proposed_runtime);

% Calculate average and standard deviation for PETALS
average_time_petals = mean(Time_petals_inference_time);
std_time_petals = std(Time_petals_inference_time);
average_ttft_petals = mean(Time_petals_ttft);
std_ttft_petals = std(Time_petals_ttft); 
average_waiting_petals = mean(Time_petals_waiting);
std_waiting_petals = std(Time_petals_waiting);
average_runtime_petals = mean(Time_petals_runtime);
std_runtime_petals = std(Time_petals_runtime);


% Calculate average and standard deviation for PETALS - optimized order
average_time_petals_optimized_order = mean(Time_petals_optimized_order_inference_time);
std_time_petals_optimized_order = std(Time_petals_optimized_order_inference_time);
average_ttft_petals_optimized_order = mean(Time_petals_optimized_order_ttft);
std_ttft_petals_optimized_order = std(Time_petals_optimized_order_ttft); 
average_waiting_petals_optimized_order = mean(Time_petals_optimized_order_waiting);
std_waiting_petals_optimized_order = std(Time_petals_optimized_order_waiting);
average_runtime_petals_optimized_order = mean(Time_petals_optimized_order_runtime);
std_runtime_petals_optimized_order = std(Time_petals_optimized_order_runtime);

% Calculate average and standard deviation for PETALS - optimized RR
average_time_petals_optimized_rr = mean(Time_petals_optimized_rr_inference_time);
std_time_petals_optimized_rr = std(Time_petals_optimized_rr_inference_time);
average_ttft_petals_optimized_rr = mean(Time_petals_optimized_rr_ttft);
std_ttft_petals_optimized_rr = std(Time_petals_optimized_rr_ttft);
average_waiting_petals_optimized_rr = mean(Time_petals_optimized_rr_waiting);
std_waiting_petals_optimized_rr = std(Time_petals_optimized_rr_waiting);
average_runtime_petals_optimized_rr = mean(Time_petals_optimized_rr_runtime);
std_runtime_petals_optimized_rr = std(Time_petals_optimized_rr_runtime);

% Display the results
disp(' ')
disp('=== Summary of Results ===');

% PETALS Benchmark
disp('--- PETALS Results ---');
disp(['Average inference time per token (ms): ', num2str(average_time_petals)]);
disp(['Standard deviation of time per token (ms): ', num2str(std_time_petals)]);
disp(['Average time to first token (ms): ', num2str(average_ttft_petals)]);
disp(['Standard deviation of time to first token (ms): ', num2str(std_ttft_petals)]);
disp(['Average remaining inference time (ms): ', num2str((average_time_petals * lc - average_ttft_petals)/(lc - 1))]);
% disp(['Average waiting time (ms): ', num2str(average_waiting_petals)]);
% disp(['Standard deviation of waiting time (ms): ', num2str(std_waiting_petals)]);
disp(['Average runtime (s): ', num2str(average_runtime_petals * 2)]);
disp(['Standard deviation of runtime (s): ', num2str(std_runtime_petals)]);


% Proposed Benchmark
disp('--- Proposed Results ---');
disp(['Average inference time per token (ms): ', num2str(average_time_proposed)]);
disp(['Standard deviation of inference time (ms): ', num2str(std_time_proposed)]);
disp(['Average time to first token (ms): ', num2str(average_ttft_proposed)]);
disp(['Standard deviation of time to first token (ms): ', num2str(std_ttft_proposed)]);
disp(['Average remaining inference time (ms): ', num2str((average_time_proposed * lc - average_ttft_proposed)/(lc - 1))]);

% disp(['Average waiting time (ms): ', num2str(average_waiting_proposed)]);
% disp(['Standard deviation of waiting time (ms): ', num2str(std_waiting_proposed)]);
disp(['Average runtime (s): ', num2str(average_runtime_proposed * 2)]);
disp(['Standard deviation of runtime (s): ', num2str(std_runtime_proposed)]);

