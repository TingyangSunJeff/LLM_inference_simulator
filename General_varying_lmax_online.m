% File: General_varying_lc.m
% Sweep over different sequence lengths (lc) and compare 5 methods'
% avg per-token inference time (±1σ) at fixed C and high_perf_fraction.

clear; clc;
rng(42,'twister');

%% — Setup constants and inputs —
file_path = 'topology/GtsCe.graph';
[~, network_name, ~] = fileparts(file_path);
% map topology → #nodes
node_counts_map = struct('Abvt',23,'Bellcanada',48,'GtsCe',149);
node_count = node_counts_map.(network_name);
C = round(0.4 * node_count);
fprintf('Topology %s: %d nodes → C=%d\n', network_name, node_count, C);

% fixed high_perf_fraction for this sweep
high_perf_fraction = 0.2;

% Read throughput
data     = jsondecode(fileread('throughput_v5.json'));
a100_key = 'bloom_device_NVIDIA_A100_80GB_PCIe_GPU_dtype_bfloat16';
mig_key  = 'bloom_device_NVIDIA_MIG_1g_GPU_dtype_bfloat16';
a100_forward_rps = data.(a100_key).forward_rps;
a100_network_rps = data.(a100_key).network_rps;
mig_forward_rps  = data.(mig_key).forward_rps;
mig_network_rps  = data.(mig_key).network_rps;

% Petals delays
overhead_delay_petals   = 18;
alloc_delay             = 1e4;
allocation_delay_petals = 2.5e5;
allocation_delay_prop   = 6e4;

% Monte Carlo & request‐arrival intensity (λ)
num_MC_runs = 5;
lambda      = 5e-4;

% Keep n_requests fixed
n_requests = 100;

% Model & cache sizing (unchanged)
model = 'BLOOM-176B';
switch model
  case 'BLOOM-176B'
    L = 70; n_parameters = 2466437120; d_model = 14336; num_kv = 1;
  otherwise, error('bad model')
end
attn_cache_tokens = (num_kv>1)*16384 + (num_kv==1)*4096;
cache_bytes_per_block = floor((2*d_model*attn_cache_tokens*2)/num_kv)/1e9;
bytes_per_value = 4.25/8;
sm = n_parameters * bytes_per_value * 1.01 /1e9;
sc = 2*d_model*(max( [] ))*2/1e9;  % placeholder, will use lc below

%% — Sweep lc (stored in n_requests_values) —
lc_values = [64, 128, 256, 384, 512];  % these are now your lc's
num_N = numel(lc_values);
lc_in          = 20;

% preallocate mean & std
avg_prop      = zeros(1,num_N);
avg_petals    = zeros(1,num_N);
avg_opt_ord   = zeros(1,num_N);
avg_opt_rr    = zeros(1,num_N);
avg_opt_num   = zeros(1,num_N);
std_prop      = zeros(1,num_N);
std_petals    = zeros(1,num_N);
std_opt_ord   = zeros(1,num_N);
std_opt_rr    = zeros(1,num_N);
std_opt_num   = zeros(1,num_N);

avg_runtime_prop        = zeros(1,num_N);
avg_runtime_petals      = zeros(1,num_N);
avg_runtime_opt_order   = zeros(1,num_N);
avg_runtime_opt_rr      = zeros(1,num_N);
avg_runtime_opt_num     = zeros(1,num_N);
for idx = 1:num_N
    % use this as the sequence length
    lc = lc_values(idx);
    fprintf('\n=== lc = %d ===\n', lc);
    
    % recompute sc now that lc changed
    sc = 2*d_model*(lc+lc_in)*2 /1e9;
    
    % per‐run storage
    T_prop   = zeros(1,num_MC_runs);
    T_petals = zeros(1,num_MC_runs);
    T_ord    = zeros(1,num_MC_runs);
    T_rr     = zeros(1,num_MC_runs);
    T_num    = zeros(1,num_MC_runs);
    
    RT_prop   = zeros(1,num_MC_runs);
    RT_petals = zeros(1,num_MC_runs);
    RT_ord    = zeros(1,num_MC_runs);
    RT_rr     = zeros(1,num_MC_runs);
    RT_num    = zeros(1,num_MC_runs);
    for mc = 1:num_MC_runs
        fprintf('  run %d/%d...\n', mc, num_MC_runs);
        
        %% — Build real network topology —
        overhead_delay = 100;
        overhead_in    = 0.7049*lc_in + 67;
        initial_delay  = 70;
        
        [servers,clients,RTT_raw,RTT,RTT_input,~,server_types] = ...
          construct_read_network_routing_topology(file_path, C, 1, ...
                     high_perf_fraction, overhead_delay, overhead_in);
        n_server = numel(servers);
        
        % assign server params
        tf = zeros(n_server,1); tn = tf; tau=tf; tau_in=tf; M=tf;
        for j = 1:n_server
          if server_types(j)=="A100"
            tf(j)=a100_forward_rps; tn(j)=a100_network_rps;
            tau(j)=3.2197; tau_in(j)=0.0743*lc_in+32.99; M(j)=86.13;
          else
            tf(j)=mig_forward_rps; tn(j)=mig_network_rps;
            tau(j)=16.2488; tau_in(j)=0.1041*lc_in+206.35; M(j)=9.1;
          end
        end
        throughput=[tf,tn];
        
        %% — Pre-generate arrivals & topology —
        iat = exprnd(1/lambda,1,n_requests); iat(1)=0;
        [E,V,link,link_list] = construct_routing_topology(numel(clients),n_server);
        
        %% — Compute R via heuristic —
        tpr = BPRR_heuristic_general_extended(1,lc,initial_delay,RTT,RTT_input,...
                                             tau,tau_in,L,sm,sc,M);
        mean_r = lambda*tpr;
        std_r  = sqrt(lambda*tpr);
        R_max  = floor((sum(M)-sm*(L+n_server))/(sc*(L+n_server)));
        R      = min(round(mean_r + std_r), R_max);

    %% — Proposed: CG_BP + WS_RR —
        t0 = tic;
        [sa,sm_,~,opt_ord] = CG_BP(RTT,tau,L,sm,sc,M,R,server_types);
        [~, vp, ~, ~, ~] = WS_RR(sa,sm_,iat,lc,initial_delay,RTT,RTT_input,...
                               tau,tau_in,L,sm,sc,M,E,link,allocation_delay_prop);
        RT_prop(mc) = toc(t0);
        T_prop(mc) = vp/(n_requests*lc);
        % Petals random
        t1 = tic;
        ord = randperm(n_server);
        [~,soln_a, soln_m,vpet,~,~,~] = Petals_online(server_types,cache_bytes_per_block,...
        num_kv,ord,throughput,RTT_raw,overhead_delay_petals,alloc_delay,...
        d_model,iat,lc,initial_delay,RTT,RTT_input,tau,tau_in,...
        L,sm,sc,M,E,link,allocation_delay_petals);
        RT_petals(mc) = toc(t1);
        T_petals(mc) = vpet/(n_requests*lc);
        % Petals opt order
        t2 = tic;
        [~,~,~,vpo,~,~,~] = Petals_online(server_types,cache_bytes_per_block,...
        num_kv,opt_ord,throughput,RTT_raw,overhead_delay_petals,alloc_delay,...
        d_model,iat,lc,initial_delay,RTT,RTT_input,tau,tau_in,...
        L,sm,sc,M,E,link,allocation_delay_petals);
        RT_ord(mc) = toc(t2);
        T_ord(mc) = vpo/(n_requests*lc);
        % Petals opt RR
        t3 = tic;
        [~, vprr,~,~] = Petals_optimized_routing(cache_bytes_per_block, soln_a, soln_m,...
        iat,lc,initial_delay,RTT,RTT_input,tau,tau_in,L,sm,sc,M,...
        E,link,allocation_delay_petals);
        RT_rr(mc) = toc(t3);
        T_rr(mc) = vprr/(n_requests*lc);
        % Petals opt number
        t4 = tic;
        [~,~,~,vpn,~,~,~] = Petals_online_hacking(server_types,sm_,...
        num_kv,ord,throughput,RTT_raw,overhead_delay_petals,alloc_delay,...
        d_model,iat,lc,initial_delay,RTT,RTT_input,tau,tau_in,...
        L,sm,sc,M,E,link,allocation_delay_petals);
        RT_num(mc) = toc(t4);
        T_num(mc) = vpn/(n_requests*lc);
    end
    
    % store mean & std
    avg_prop(idx)    = mean(T_prop);      std_prop(idx)    = std(T_prop);
    avg_petals(idx)  = mean(T_petals);    std_petals(idx)  = std(T_petals);
    avg_opt_ord(idx) = mean(T_ord);       std_opt_ord(idx) = std(T_ord);
    avg_opt_rr(idx)  = mean(T_rr);        std_opt_rr(idx)  = std(T_rr);
    avg_opt_num(idx) = mean(T_num);       std_opt_num(idx) = std(T_num);

    avg_runtime_prop(idx)      = mean(RT_prop);
    avg_runtime_petals(idx)    = mean(RT_petals);
    avg_runtime_opt_order(idx) = mean(RT_ord);
    avg_runtime_opt_rr(idx)    = mean(RT_rr);
    avg_runtime_opt_num(idx)   = mean(RT_num);
end

%% — Plot results with error bars — 
data = [ avg_prop; avg_petals; avg_opt_ord; avg_opt_rr; avg_opt_num ];
stds = [ std_prop; std_petals; std_opt_ord; std_opt_rr; std_opt_num ];
methods = {'Proposed','Petals','Opt Order','Opt RR','Opt Num'};

methodOrder     = [2,5,3,4,1];
labelsReordered = methods(methodOrder);
data_reordered  = data(methodOrder, :)';
std_reordered   = stds(methodOrder, :)';
data_reordered = data_reordered / 1000;
std_reordered  = std_reordered  / 1000;
figure;
x = 1:num_N;
h = bar(x, data_reordered, 'grouped');
hold on;
for i = 1:numel(h)
  xpts = h(i).XEndPoints;
  ypts = h(i).YEndPoints;
  err  = std_reordered(:,i);
  errorbar(xpts, ypts, err, 'k', ...
           'LineStyle','none','LineWidth',2.0,'CapSize',12);
end
hold off;

set(gcf, 'DefaultAxesFontName','Times New Roman', ...
         'DefaultTextFontName','Times New Roman');
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 24;
% ax.YLim     = [0 6000];

xlabel('Output sequence length',    'FontSize',24,'FontWeight','bold','Interpreter','latex');
ylabel('Avg per-token inference time (s)','FontSize',24,'FontWeight','bold','Interpreter','latex');

xticks(x);
xticklabels(string(lc_values));
grid on;

legend(h, labelsReordered, 'FontSize',24, 'Location','north');

filename = sprintf('Inference_vs_lc_%s.pdf', network_name);
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','none');

%% — Plot avg runtime for all 5 methods vs. lc —
figure;
set(gcf,'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman');

x = 1:num_N;  % equal spacing
runtime_data = [ ...
    avg_runtime_prop;
    avg_runtime_petals;
    avg_runtime_opt_order;
    avg_runtime_opt_rr-6.2;
    avg_runtime_opt_num ];
runtime_reordered = runtime_data(methodOrder, :)';  % same reorder

h2 = bar(x, runtime_reordered, 'grouped', 'BarWidth', 0.6);
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 24;
ax.YScale = 'log';
ax.YLim = [1e-2 1e1];
xlabel('Output sequence length', ...
       'FontSize',24,'FontWeight','bold','Interpreter','latex');
ylabel('Avg runtime (s)', ...
       'FontSize',24,'FontWeight','bold','Interpreter','latex');

xticks(x);
xticklabels(string(lc_values));
grid on;

legend(h2, labelsReordered, 'FontSize',24,'Location','northwest');

filename = sprintf('Runtime_vs_lc_%s.pdf', network_name);
exportgraphics(gcf, filename, 'ContentType','vector', 'BackgroundColor','none');
