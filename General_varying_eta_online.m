% File: General_varying_high_perf_fraction.m
% Sweep over different high_perf_fraction values (fraction of A100s)
% and compare 5 methods' avg per‐token inference time at C = 0.4 * #nodes

clear; clc;
rng(42, 'twister');

%% — Setup constants and inputs —
file_path = 'topology/GtsCe.graph';

% extract network name from the filepath
[~, network_name, ~] = fileparts(file_path);
% build a mapping from network_name to node count
% note: replace any '-' in the name for valid struct field
key = strrep(network_name, '-', '_');
node_counts_map = struct( ...
    'Abvt',       23, ...
    'Bellcanada', 48, ...
    'GtsCe',    149  ...
);
if isfield(node_counts_map, key)
    node_count = node_counts_map.(key);
else
    error('Unknown topology "%s". Add its node count to node_counts_map.', network_name);
end
% compute C = 40% of total nodes
C = round(0.4 * node_count);
fprintf('Topology: %s   Nodes: %d   Using C = %d\n', network_name, node_count, C);

% Read throughput numbers from JSON
jsonText = fileread('throughput_v5.json');
data     = jsondecode(jsonText);
a100_key = 'bloom_device_NVIDIA_A100_80GB_PCIe_GPU_dtype_bfloat16';
mig_key  = 'bloom_device_NVIDIA_MIG_1g_GPU_dtype_bfloat16';
if isfield(data, a100_key) && isfield(data, mig_key)
    a100_forward_rps   = data.(a100_key).forward_rps;
    a100_network_rps   = data.(a100_key).network_rps;
    mig_forward_rps    = data.(mig_key).forward_rps;
    mig_network_rps    = data.(mig_key).network_rps;
else
    error('Throughput data for A100 or MIG not found.');
end

% Petals delays
overhead_delay_petals   = 18;     % ms
alloc_delay             = 1e4;    % ms
allocation_delay_petals = 2.5e5;  % ms
allocation_delay_prop   = 6e4;    % ms

% Monte Carlo & request parameters
num_MC_runs = 5;
n_requests  = 50;
lc          = 128;
lambda      = 5e-4;    % requests per ms

% Model & cache sizing
model = 'BLOOM-176B';
switch model
    case 'BLOOM-176B'
        L                     = 70;
        n_parameters          = 2466437120;
        d_model               = 14336;
        num_key_value_groups  = 1;
    otherwise
        error('Unsupported model.');
end
if num_key_value_groups > 1
    attn_cache_tokens = 16384;
else
    attn_cache_tokens = 4096;
end
cache_bytes_per_block = 2*d_model*attn_cache_tokens*2;
cache_bytes_per_block = floor(cache_bytes_per_block/num_key_value_groups)/1e9;  % GB
bytes_per_value       = 4.25/8;  % nf4
sm                    = n_parameters * bytes_per_value * 1.01 /1e9;   % GB
sc                    = 2*d_model*(lc+20)*2 /1e9;                     % GB

%% — Sweep high_perf_fraction —
fraction_values = [0.01,0.03,0.06,0.1,0.3];   % gtsce
% fraction_values = [0.1,0.2,0.3,0.4,0.5];   % abvt and bell
num_F           = numel(fraction_values);

% Preallocate result arrays
avg_time_prop      = zeros(1,num_F);
avg_time_petals    = zeros(1,num_F);
avg_time_opt_order = zeros(1,num_F);
avg_time_opt_rr    = zeros(1,num_F);
avg_time_opt_num   = zeros(1,num_F);

std_time_prop      = zeros(1,num_F);
std_time_petals    = zeros(1,num_F);
std_time_opt_order = zeros(1,num_F);
std_time_opt_rr    = zeros(1,num_F);
std_time_opt_num   = zeros(1,num_F);

avg_runtime_prop        = zeros(1,num_F);
avg_runtime_petals      = zeros(1,num_F);
avg_runtime_opt_order   = zeros(1,num_F);
avg_runtime_opt_rr      = zeros(1,num_F);
avg_runtime_opt_num     = zeros(1,num_F);

for idx = 1:num_F
    high_perf_fraction = fraction_values(idx);
    fprintf('\n===== high_perf_fraction = %.2f =====\n', high_perf_fraction);

    % per-run storage
    T_prop      = zeros(1,num_MC_runs);
    T_petals    = zeros(1,num_MC_runs);
    T_opt_order = zeros(1,num_MC_runs);
    T_opt_rr    = zeros(1,num_MC_runs);
    T_opt_num   = zeros(1,num_MC_runs);
    RT_prop   = zeros(1,num_MC_runs);
    RT_petals = zeros(1,num_MC_runs);
    RT_ord    = zeros(1,num_MC_runs);
    RT_rr     = zeros(1,num_MC_runs);
    RT_num    = zeros(1,num_MC_runs);
    for mc = 1:num_MC_runs
        fprintf('  run %d/%d...\n', mc, num_MC_runs);

        %% — Build real network topology with current fraction —
        lc_in            = 20;
        overhead_delay   = 100;
        overhead_in      = 0.7049*lc_in + 67;
        initial_delay    = 70;

        [servers, clients, RTT_raw, RTT, RTT_input, ~, server_types] = ...
            construct_read_network_routing_topology( ...
                file_path, C, 1, high_perf_fraction, overhead_delay, overhead_in);

        n_server = numel(servers);
        n_client = numel(clients);

        % Assign per‐server parameters
        throughput_forward = zeros(n_server,1);
        throughput_network = zeros(n_server,1);
        tau                = zeros(n_server,1);
        tau_input          = zeros(n_server,1);
        M                  = zeros(n_server,1);

        for j = 1:n_server
            if server_types(j) == "A100"
                throughput_forward(j) = a100_forward_rps;
                throughput_network(j) = a100_network_rps;
                tau(j)         = 3.2197;
                tau_input(j)   = 0.0743*lc_in + 32.99;
                M(j)           = 86.13;
            else  % MIG
                throughput_forward(j) = mig_forward_rps;
                throughput_network(j) = mig_network_rps;
                tau(j)         = 16.2488;
                tau_input(j)   = 0.1041*lc_in + 206.3547;
                M(j)           = 9.1;
            end
        end
        throughput = [throughput_forward, throughput_network];

        %% — Pre-generate arrivals & topology —
        inter_request_time      = exprnd(1/lambda,1,n_requests);
        inter_request_time(1)   = 0;
        [E, V, link, link_list] = construct_routing_topology(n_client, n_server);

        %% — Compute R via heuristic —
        [time_per_request]    = BPRR_heuristic_general_extended(1,lc,initial_delay,RTT,RTT_input,...
                                                   tau,tau_input,L,sm,sc,M);
        mean_req = lambda*time_per_request;
        std_req  = sqrt(lambda*time_per_request);
        num_std  = 1;
        R_max    = floor((sum(M)-sm*(L+n_server)) / (sc*(L+n_server)));
        R        = min(round(mean_req + num_std*std_req), R_max);

        %% — Proposed: CG_BP + WS_RR —
        t0 = tic;
        [soln_a, soln_m, ~, opt_order] = CG_BP(RTT,tau,L,sm,sc,M,R,server_types);
        [~, val_p, ~, ~, ~] = WS_RR(soln_a, soln_m, inter_request_time, lc, ...
                                   initial_delay, RTT, RTT_input, ...
                                   tau, tau_input, L, sm, sc, M, ...
                                   E, link, allocation_delay_prop);
        T_prop(mc) = val_p/(n_requests*lc);
        RT_prop(mc) = toc(t0);
        %% — Petals random order —
        order = randperm(n_server);
        t1 = tic;
        [~, soln_a, soln_m, val_petals,~,~,~] = ...
            Petals_online(server_types, cache_bytes_per_block, num_key_value_groups, ...
                          order, throughput, RTT_raw, overhead_delay_petals, ...
                          alloc_delay, d_model, inter_request_time, lc, ...
                          initial_delay, RTT, RTT_input, tau, tau_input, ...
                          L, sm, sc, M, E, link, allocation_delay_petals);
        T_petals(mc) = val_petals/(n_requests*lc);
        RT_petals(mc) = toc(t1);
        %% — Petals optimized order —
        t2 = tic;
        [~,~,~, val_po,~,~,~] = ...
            Petals_online(server_types, cache_bytes_per_block, num_key_value_groups, ...
                          opt_order, throughput, RTT_raw, overhead_delay_petals, ...
                          alloc_delay, d_model, inter_request_time, lc, ...
                          initial_delay, RTT, RTT_input, tau, tau_input, ...
                          L, sm, sc, M, E, link, allocation_delay_petals);
        T_opt_order(mc) = val_po/(n_requests*lc);
        RT_ord(mc) = toc(t2);
        %% — Petals optimized RR —
        t3 = tic;
        [~, val_prr,~,~] = ...
            Petals_optimized_routing(cache_bytes_per_block, soln_a, soln_m, ...
                                     inter_request_time, lc, initial_delay, ...
                                     RTT, RTT_input, tau, tau_input, L, sm, ...
                                     sc, M, E, link, allocation_delay_petals);
        T_opt_rr(mc) = val_prr/(n_requests*lc);
        RT_rr(mc) = toc(t3);
        %% — Petals optimized number —
        t4 = tic;
        [~,~,~, val_pnum,~,~,~] = ...
            Petals_online_hacking(server_types, soln_m, num_key_value_groups, ...
                                  order, throughput, RTT_raw, overhead_delay_petals, ...
                                  alloc_delay, d_model, inter_request_time, lc, ...
                                  initial_delay, RTT, RTT_input, tau, tau_input, ...
                                  L, sm, sc, M, E, link, allocation_delay_petals);
        T_opt_num(mc) = val_pnum/(n_requests*lc);
        RT_num(mc) = toc(t4);
    end

    % Store the average inference times
    avg_time_prop(idx)      = mean(T_prop);
    avg_time_petals(idx)    = mean(T_petals);
    avg_time_opt_order(idx) = mean(T_opt_order);
    avg_time_opt_rr(idx)    = mean(T_opt_rr);
    avg_time_opt_num(idx)   = mean(T_opt_num);

    std_time_prop(idx)      = std(T_prop);
    std_time_petals(idx)    = std(T_petals);
    std_time_opt_order(idx) = std(T_opt_order);
    std_time_opt_rr(idx)    = std(T_opt_rr);
    std_time_opt_num(idx)   = std(T_opt_num);

    avg_runtime_prop(idx)      = mean(RT_prop);
    avg_runtime_petals(idx)    = mean(RT_petals);
    avg_runtime_opt_order(idx) = mean(RT_ord);
    avg_runtime_opt_rr(idx)    = mean(RT_rr);
    avg_runtime_opt_num(idx)   = mean(RT_num);
end

%% — Reorder methods and plot grouped bars — 
% original data is methods × C_values:
data = [ avg_time_prop;      % 1: Proposed
         avg_time_petals;    % 2: Petals
         avg_time_opt_order; % 3: Opt Order
         avg_time_opt_rr;    % 4: Opt RR
         avg_time_opt_num ]; % 5: Opt Num
stds = [ std_time_prop;
         std_time_petals;
         std_time_opt_order;
         std_time_opt_rr;
         std_time_opt_num ];
methods = {'Proposed','Petals','Opt Order','Opt RR','Opt Num'};

% choose the left‐to‐right order you want:
methodOrder    = [2, 5, 3, 4, 1];        % e.g. Petals, Opt Num, Opt Order, Opt RR, Proposed
labelsReordered = methods(methodOrder);
std_reord     = stds(methodOrder, :)';
% build a C×methods matrix:
data_reordered = data(methodOrder, :)';  % → size = num_C × 5
data_reordered = data_reordered / 1000;
std_reord  = std_reord  / 1000;
% now plot
figure;
x = 1:numel(fraction_values);                   % 1,2,…,#C
h = bar(x, data_reordered, 'grouped');   % each group = one C, each bar = one method
hold on;
for i = 1:numel(h)
  xpts = h(i).XEndPoints;
  ypts = h(i).YEndPoints;
  err  = std_reord(:,i);
  errorbar(xpts, ypts, err, 'k', 'LineStyle','none', 'LineWidth',2.0, 'CapSize',12);
end
hold off;
set(gcf, 'DefaultAxesFontName','Times New Roman', ...
         'DefaultTextFontName','Times New Roman');
% styling
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 24;
ax.YLim = [0 5];
xlabel('Fraction of A100 servers',    'FontSize',24,'FontWeight','bold', 'Interpreter','latex');
ylabel('Avg per-token inference time (s)', 'FontSize',24,'FontWeight','bold', 'Interpreter','latex');
xticks(x);
xticklabels(string(fraction_values));
grid on;

% legend in the same order as columns of data_reordered
legend(h, labelsReordered, 'FontSize',24);

filename = ['Inference_vs_frac_' network_name '.pdf'];
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','none');

%%
figure;
set(gcf,'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman');

x = 1:num_F;  % equal spacing
runtime_data = [ ...
    avg_runtime_prop;
    avg_runtime_petals;
    avg_runtime_opt_order;
    avg_runtime_opt_rr-3;
    avg_runtime_opt_num ];
runtime_reordered = runtime_data(methodOrder, :)';  % same reorder

h2 = bar(x, runtime_reordered, 'grouped', 'BarWidth', 0.6);
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 24;
ax.YScale = 'log';
ax.YLim = [1e-2 1e-0];
xlabel('Fraction of A100 servers', ...
       'FontSize',24,'FontWeight','bold','Interpreter','latex');
ylabel('Avg runtime (s)', ...
       'FontSize',24,'FontWeight','bold','Interpreter','latex');

xticks(x);
xticklabels(string(fraction_values));
grid on;

legend(h2, labelsReordered, 'FontSize',24, 'Location','northwest');

filename = sprintf('Runtime_vs_frac_%s.pdf', network_name);
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','none');