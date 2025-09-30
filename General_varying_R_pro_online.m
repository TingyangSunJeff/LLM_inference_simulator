% File: General_varying_R_Proposed.m
% Sweep over different target concurrent‐request values R for the Proposed method only,
% and plot average per‐token inference time for each R (including the heuristic‐chosen R).

clear; clc;
% rng(42,'twister');

%% — Setup constants and inputs (same as your original) —
file_path = 'topology/GtsCe.graph';
[~, network_name, ~] = fileparts(file_path);
node_counts_map = struct('Abvt',23,'Bellcanada',48,'GtsCe',149);
node_count = node_counts_map.(network_name);
C = round(0.4 * node_count);

% fixed high_perf_fraction
high_perf_fraction = 0.2;

% Read throughput
data     = jsondecode(fileread('throughput_v5.json'));
a100_key = 'bloom_device_NVIDIA_A100_80GB_PCIe_GPU_dtype_bfloat16';
mig_key  = 'bloom_device_NVIDIA_MIG_1g_GPU_dtype_bfloat16';
a100_forward_rps = data.(a100_key).forward_rps;
a100_network_rps = data.(a100_key).network_rps;
mig_forward_rps  = data.(mig_key).forward_rps;
mig_network_rps  = data.(mig_key).network_rps;

% delays & MC settings
overhead_delay_petals   = 18;
alloc_delay             = 1e4;
allocation_delay_prop   = 6e4;
num_MC_runs             = 20;
n_requests              = 100;
lc                      = 128;
lc_in                   = 20;
lambda                  = 5e-4;

% model & cache sizing
L                    = 70;
n_parameters         = 2466437120;
d_model              = 14336;
num_kv               = 1;
attn_cache_tokens    = (num_kv>1)*16384 + (num_kv==1)*4096;
cache_bytes_per_block= floor((2*d_model*attn_cache_tokens*2)/num_kv)/1e9;
bytes_per_value      = 4.25/8;
sm                   = n_parameters * bytes_per_value * 1.01 /1e9;
sc                   = 2*d_model*(lc+lc_in)*2 /1e9;

%% — Build one network & compute heuristic R —
% topology + server params
overhead_delay = 100;
overhead_in    = 0.7049*lc_in + 67;
initial_delay  = 70;



[servers,clients,RTT_raw,RTT,RTT_input,~,server_types] = ...
        construct_read_network_routing_topology(file_path, C, 1, ...
                                         high_perf_fraction, ...
                                         overhead_delay, overhead_in);
        n_server = numel(servers);
        % assign throughput, tau, tau_input, M
        throughput_forward = zeros(n_server,1);
        throughput_network = zeros(n_server,1);
        tau                = zeros(n_server,1);
        tau_input          = zeros(n_server,1);
        M                  = zeros(n_server,1);
        for j = 1:n_server
          if server_types(j)=="A100"
            throughput_forward(j)=a100_forward_rps;
            throughput_network(j)=a100_network_rps;
            tau(j)=3.2197;
            tau_input(j)=0.0743*lc_in+32.99;
            M(j)=86.13;
          else
            throughput_forward(j)=mig_forward_rps;
            throughput_network(j)=mig_network_rps;
            tau(j)=16.2488;
            tau_input(j)=0.1041*lc_in+206.35;
            M(j)=9.1;
          end
        end
        throughput = [throughput_forward, throughput_network];
        inter_request_time = exprnd(1/lambda,1,n_requests);
        inter_request_time(1)=0;
        [E,V,link,link_list] = construct_routing_topology(numel(clients),n_server);
        % time the CG_BP + WS_RR call

% heuristic R
time_per_request = BPRR_heuristic_general_extended(1,lc,initial_delay,RTT,...
                                                  RTT_input,tau,tau_input,...
                                                  L,sm,sc,M);
mean_r = lambda * time_per_request;
std_r  = sqrt(lambda * time_per_request);
num_std= 1;
R_max  = floor((sum(M)-sm*(L+n_server))/(sc*(L+n_server)));
R_heur = min(round(mean_r + num_std*std_r), R_max);

% choose R values to sweep (include R_heuristic)
% R_values = [17,34,51,68,85];
R_values = [8,16,32,48,64];

%% — Sweep Proposed over R_values (measuring both latency & runtime) —
avg_prop    = zeros(size(R_values));
avg_runtime = zeros(size(R_values));
std_prop    = zeros(size(R_values));
for ix = 1:numel(R_values)
    R_i = R_values(ix);
    fprintf('\n=== R = %d ===\n', R_i);
    
    results_latency = zeros(1,num_MC_runs);
    results_runtime = zeros(1,num_MC_runs);

    for mc = 1:num_MC_runs
        [servers,clients,RTT_raw,RTT,RTT_input,~,server_types] = ...
        construct_read_network_routing_topology(file_path, C, 1, ...
                                         high_perf_fraction, ...
                                         overhead_delay, overhead_in);
        n_server = numel(servers);
        % assign throughput, tau, tau_input, M
        throughput_forward = zeros(n_server,1);
        throughput_network = zeros(n_server,1);
        tau                = zeros(n_server,1);
        tau_input          = zeros(n_server,1);
        M                  = zeros(n_server,1);
        for j = 1:n_server
          if server_types(j)=="A100"
            throughput_forward(j)=a100_forward_rps;
            throughput_network(j)=a100_network_rps;
            tau(j)=3.2197;
            tau_input(j)=0.0743*lc_in+32.99;
            M(j)=86.13;
          else
            throughput_forward(j)=mig_forward_rps;
            throughput_network(j)=mig_network_rps;
            tau(j)=16.2488;
            tau_input(j)=0.1041*lc_in+206.35;
            M(j)=9.1;
          end
        end
        throughput = [throughput_forward, throughput_network];
        inter_request_time = exprnd(1/lambda,1,n_requests);
        inter_request_time(1)=0;
        [E,V,link,link_list] = construct_routing_topology(numel(clients),n_server);
        % time the CG_BP + WS_RR call
        t0 = tic;
          [soln_a,soln_m,~,~] = CG_BP(RTT,tau,L,sm,sc,M,R_i,server_types);
          [~, val, ~, ~, ~] = WS_RR(soln_a,soln_m,inter_request_time,lc, ...
                                   initial_delay,RTT,RTT_input, ...
                                   tau,tau_input,L,sm,sc,M, ...
                                   E,link,allocation_delay_prop);
        results_runtime(mc) = toc(t0);
        
        % per-token inference time
        results_latency(mc) = val/(n_requests*lc);
    end
    
    avg_prop(ix)    = mean(results_latency);
    avg_runtime(ix) = mean(results_runtime);
    std_prop(ix)    = std(results_latency);
end

%% — Plot both metrics — 
figure;
set(gcf,'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman');

% evenly spaced positions
x = 1:numel(R_values);

% inference‐time plot
b1 = bar(x, avg_prop, 'BarWidth', 0.4, 'FaceColor',[0 0.6 0]);
hold on;
errorbar( x, ...
          avg_prop, ...
          std_prop, ...
          'k', ...
          'LineStyle','none', ...
          'LineWidth',2, ...
          'CapSize',18);
hold off;
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize             = 22;
% ax.YLim     = [0 1800];
xlabel('Target concurrent number of requests','Interpreter','latex','FontSize',24,'FontWeight','bold');
ylabel('Avg per-token inference time (ms)','Interpreter','latex','FontSize',24,'FontWeight','bold');
xticks(x);
xticklabels(string(R_values));  % show your actual R's
grid on;
legend(b1, {'Proposed'},'FontSize',24,'Location','northeast');
fname = sprintf('Proposed_InferenceTime_vs_R_%s.pdf',network_name);
exportgraphics(gcf,fname,'ContentType','vector','BackgroundColor','none');

%% — Plot Avg runtime alone — 
figure;
b2 = set(gcf,'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman');

bar(x, avg_runtime, 'BarWidth', 0.5, 'FaceColor',[0 0.6 0]);
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize             = 22;
% ax.YLim     = [0 0.3];
xlabel('Target concurrent number of requests','Interpreter','latex','FontSize',24,'FontWeight','bold');
ylabel('Avg runtime (s)','Interpreter','latex','FontSize',24,'FontWeight','bold');
xticks(x);
xticklabels(string(R_values));
grid on;
legend(b2, {'Proposed'},'FontSize',24,'Location','northeast');
fname = sprintf('Proposed_Runtime_vs_R_%s.pdf',network_name);
exportgraphics(gcf,fname,'ContentType','vector','BackgroundColor','none');