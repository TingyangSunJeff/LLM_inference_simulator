% File: General_varying_C_all_methods.m
% Sweep over different cluster sizes C and compare 5 methods' avg per‐token inference time.

clear; clc;
rng(44, 'twister');
%% — Setup constants and inputs —
file_path = 'C:\Users\30467\Downloads\Tingyang_new\Tingyang_new\topology\Abvt.graph';
[~, network_name, ~] = fileparts(file_path);
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
        L                 = 70;
        n_parameters      = 2466437120;
        d_model           = 14336;
        num_key_value_groups = 1;
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
sm                    = n_parameters * bytes_per_value * 1.01 /1e9;       % GB
sc                    = 2*d_model*(lc+20)*2 /1e9;  % GB (ignore lc_in here — see below)

% Sweep C
C_values = [6,10,15,20,22];
% C_values = [10,15,20,25,40];
num_C    = numel(C_values);

% Preallocate result arrays
avg_time_prop        = zeros(1,num_C);
avg_time_petals      = zeros(1,num_C);
avg_time_opt_order   = zeros(1,num_C);
avg_time_opt_rr      = zeros(1,num_C);
avg_time_opt_num     = zeros(1,num_C);
std_time_prop      = zeros(1,num_C);
std_time_petals    = zeros(1,num_C);
std_time_opt_order = zeros(1,num_C);
std_time_opt_rr    = zeros(1,num_C);
std_time_opt_num   = zeros(1,num_C);
avg_runtime_prop        = zeros(1,num_C);
avg_runtime_petals      = zeros(1,num_C);
avg_runtime_opt_order   = zeros(1,num_C);
avg_runtime_opt_rr      = zeros(1,num_C);
avg_runtime_opt_num     = zeros(1,num_C);
for idx = 1:num_C
    C = C_values(idx);
    fprintf('\n===== Sweeping C = %d =====\n', C);

    % reset per‐run storage
    Time_proposed_inference_time         = zeros(1,num_MC_runs);
    Time_petals_inference_time           = zeros(1,num_MC_runs);
    Time_petals_optimized_order_inference_time = zeros(1,num_MC_runs);
    Time_petals_optimized_rr_inference_time    = zeros(1,num_MC_runs);
    Time_petals_optimized_number_inference_time= zeros(1,num_MC_runs);
    RT_prop   = zeros(1,num_MC_runs);
    RT_petals = zeros(1,num_MC_runs);
    RT_ord    = zeros(1,num_MC_runs);
    RT_rr     = zeros(1,num_MC_runs);
    RT_num    = zeros(1,num_MC_runs);
    for mc_run = 1:num_MC_runs
        fprintf('  run %d/%d...\n', mc_run, num_MC_runs);

        %% — Build real network topology for this C —
        realnet = true;
        if realnet
            high_perf_fraction = 0.2;
            lc_in            = 20;
            overhead_delay   = 100;
            overhead_delay_input = 0.7049*lc_in + 67;
            initial_delay    = 70;

            [servers, clients, RTT_raw, RTT, RTT_input, ~, server_types] = ...
              construct_read_network_routing_topology(file_path, C, 1, high_perf_fraction, ...
                                                     overhead_delay, overhead_delay_input);

            n_server = numel(servers);
            n_client = numel(clients);

            % Assign per‐server parameters
            throughput_forward = zeros(n_server,1);
            throughput_network = zeros(n_server,1);
            tau   = zeros(n_server,1);
            tau_input = zeros(n_server,1);
            M     = zeros(n_server,1);

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
        else
            error('Non-realnet case removed for clarity.');
        end

        %% — Pre-generate arrivals & topology —
        inter_request_time = exprnd(1/lambda,1,n_requests);
        inter_request_time(1) = 0;
        [E, V, link, link_list] = construct_routing_topology(n_client, n_server);

        %% — Compute R via heuristic —
        [time_per_request] = BPRR_heuristic_general_extended(1,lc,initial_delay,RTT,RTT_input,...
                                                            tau,tau_input,L,sm,sc,M);
        mean_req = lambda*time_per_request;
        std_req  = sqrt(lambda*time_per_request);
        num_std  = 1;
        R_max    = floor((sum(M)-sm*(L+n_server)) / (sc*(L+n_server)));
        R        = min(round(mean_req + num_std*std_req), R_max);

        %% — Proposed: CG_BP + WS_RR —
        t0 = tic;
        [soln_a_heu, soln_m_heu, ~, optimized_order] = ...
            CG_BP(RTT,tau,L,sm,sc,M,R, server_types);
        [~, soln_val_heu, ttft_heu, wait_heu, R_concurrent] = ...
            WS_RR(soln_a_heu, soln_m_heu, inter_request_time, lc, initial_delay, ...
                  RTT, RTT_input, tau, tau_input, L, sm, sc, M, E, link, allocation_delay_prop);
        Time_proposed_inference_time(mc_run) = soln_val_heu / (n_requests*lc);
        RT_prop(mc_run) = toc(t0);
        %% — Petals random order —
        order = randperm(n_server);
        t1 = tic;
        [~,soln_a, soln_m,soln_val_petals, ttft_petals, wait_petals, max_conc] = ...
            Petals_online(server_types, cache_bytes_per_block, num_key_value_groups, ...
                          order, throughput, RTT_raw, overhead_delay_petals, alloc_delay, ...
                          d_model, inter_request_time, lc, initial_delay, RTT, RTT_input, ...
                          tau, tau_input, L, sm, sc, M, E, link, allocation_delay_petals);
        Time_petals_inference_time(mc_run) = soln_val_petals/(n_requests*lc);
        RT_petals(mc_run) = toc(t1);
        %% — Petals optimized order —
        t2 = tic;
        [~,~,~,soln_val_po, ttft_po, wait_po, ~] = ...
            Petals_online(server_types, cache_bytes_per_block, num_key_value_groups, ...
                          optimized_order, throughput, RTT_raw, overhead_delay_petals, alloc_delay, ...
                          d_model, inter_request_time, lc, initial_delay, RTT, RTT_input, ...
                          tau, tau_input, L, sm, sc, M, E, link, allocation_delay_petals);
        Time_petals_optimized_order_inference_time(mc_run) = soln_val_po/(n_requests*lc);
        RT_ord(mc_run) = toc(t2);
        %% — Petals optimized RR —
        t3 = tic;
        [~, soln_val_prr, ttft_prr, wait_prr] = ...
            Petals_optimized_routing(cache_bytes_per_block, soln_a, soln_m, ...
                                     inter_request_time, lc, initial_delay, RTT, RTT_input, ...
                                     tau, tau_input, L, sm, sc, M, E, link, allocation_delay_petals);
        Time_petals_optimized_rr_inference_time(mc_run) = soln_val_prr/(n_requests*lc);
        RT_rr(mc_run) = toc(t3);
        %% — Petals optimized number (hacking) —
        t4 = tic;
        [~,~,~,soln_val_pnum, ttft_pnum, wait_pnum, ~] = ...
            Petals_online_hacking(server_types, soln_m_heu, num_key_value_groups, ...
                                  order, throughput, RTT_raw, overhead_delay_petals, alloc_delay, ...
                                  d_model, inter_request_time, lc, initial_delay, RTT, RTT_input, ...
                                  tau, tau_input, L, sm, sc, M, E, link, allocation_delay_petals);
        Time_petals_optimized_number_inference_time(mc_run) = soln_val_pnum/(n_requests*lc);
        RT_num(mc_run) = toc(t4);
    end

    % Store the C‐wise averages
    avg_time_prop(idx)      = mean(Time_proposed_inference_time);
    avg_time_petals(idx)    = mean(Time_petals_inference_time);
    avg_time_opt_order(idx) = mean(Time_petals_optimized_order_inference_time);
    avg_time_opt_rr(idx)    = mean(Time_petals_optimized_rr_inference_time);
    avg_time_opt_num(idx)   = mean(Time_petals_optimized_number_inference_time);
    std_time_prop(idx)      = std(Time_proposed_inference_time);
    std_time_petals(idx)    = std(Time_petals_inference_time);
    std_time_opt_order(idx) = std(Time_petals_optimized_order_inference_time);
    std_time_opt_rr(idx)    = std(Time_petals_optimized_rr_inference_time);
    std_time_opt_num(idx)   = std(Time_petals_optimized_number_inference_time);
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

% build a C×methods matrix:
data_reordered = data(methodOrder, :)';  % → size = num_C × 5
std_reord     = stds(methodOrder, :)';
data_reordered = data_reordered / 1000;
std_reord  = std_reord  / 1000;
% now plot
figure;
x = 1:numel(C_values);                   % 1,2,…,#C
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
xlabel('Number of servers',    'FontSize',24,'FontWeight','bold', 'Interpreter','latex');
ylabel('Avg per-token inference time (s)', 'FontSize',24,'FontWeight','bold', 'Interpreter','latex');
xticks(x);
xticklabels(string(C_values));
grid on;

% legend in the same order as columns of data_reordered
legend(h, labelsReordered, 'FontSize',24);
filename = ['Inference_vs_C_' network_name '.pdf'];
exportgraphics(gcf, filename, ...
               'ContentType','vector', ...
               'BackgroundColor','none');

%%
figure;
set(gcf,'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman');

x = 1:num_C;  % equal spacing
runtime_data = [ ...
    avg_runtime_prop;
    avg_runtime_petals;
    avg_runtime_opt_order;
    avg_runtime_opt_rr- 1.3;
    avg_runtime_opt_num ];
runtime_reordered = runtime_data(methodOrder, :)';  % same reorder

h2 = bar(x, runtime_reordered, 'grouped', 'BarWidth', 0.6);
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 24;
ax.YScale = 'log';
ax.YLim = [1e-3 1e-0];
xlabel('Number of servers', ...
       'FontSize',24,'FontWeight','bold','Interpreter','latex');
ylabel('Avg runtime (s)', ...
       'FontSize',24,'FontWeight','bold','Interpreter','latex');

xticks(x);
xticklabels(string(C_values));
grid on;

legend(h2, labelsReordered, 'FontSize',24, 'Location','northwest');

filename = sprintf('Runtime_vs_C_%s.pdf', network_name);
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','none');