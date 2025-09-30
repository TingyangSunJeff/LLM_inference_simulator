% File: General_varying_lambda.m
% Sweep over different arrival rates λ and compare 5 methods' avg per‐token inference time
% (±1σ) at fixed C, high_perf_fraction, and n_requests = 100.

clear; clc;
rng(42,'twister');

%% — Setup constants and inputs —
file_path = 'C:\Users\30467\Downloads\Tingyang_new\Tingyang_new\topology\Abvt.graph';
[~, network_name, ~] = fileparts(file_path);
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

% Monte Carlo runs
num_MC_runs = 10;

% Keep n_requests fixed
% n_requests = 100;

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
sc = 2*d_model*(128+20)*2/1e9;

%% — Sweep λ instead of n_requests —
lambda_values = [0.00005,0.0001,0.0005,0.001,0.002];
% lambda_values = [0.0001,0.0005,0.001,0.002,0.003];
num_N         = numel(lambda_values);

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

for i = 1:num_N
  lambda = lambda_values(i);
  n_requests = round(200*lambda*1000);
  fprintf('\n=== λ = %.1e ===\n', lambda);
  
  Tp    = zeros(1,num_MC_runs);
  Tpet  = zeros(1,num_MC_runs);
  Tord  = zeros(1,num_MC_runs);
  Trr   = zeros(1,num_MC_runs);
  Tnum  = zeros(1,num_MC_runs);
  RT_prop   = zeros(1,num_MC_runs);
  RT_petals = zeros(1,num_MC_runs);
  RT_ord    = zeros(1,num_MC_runs);
  RT_rr     = zeros(1,num_MC_runs);
  RT_num    = zeros(1,num_MC_runs);
  
  for mc = 1:num_MC_runs
    fprintf('  run %d/%d...\n', mc, num_MC_runs);
    %% build topology
    lc_in          = 20;
    overhead_delay = 100;
    overhead_in    = 0.7049*lc_in+67;
    initial_delay  = 70;
    [servers,clients,RTT_raw,RTT,RTT_input,~,server_types] = ...
      construct_read_network_routing_topology(file_path, C, 1, ...
                 high_perf_fraction, overhead_delay, overhead_in);
    n_server = numel(servers);
    
    % assign server params
    tf = zeros(n_server,1); tn = tf; tau=tf; tau_in=tf; M=tf;
    for j=1:n_server
      if server_types(j)=="A100"
        tf(j)=a100_forward_rps; tn(j)=a100_network_rps;
        tau(j)=3.2197; tau_in(j)=0.0743*lc_in+32.99; M(j)=86.13;
      else
        tf(j)=mig_forward_rps; tn(j)=mig_network_rps;
        tau(j)=16.2488; tau_in(j)=0.1041*lc_in+206.35; M(j)=9.1;
      end
    end
    throughput=[tf,tn];
    
    %% arrivals + routing topology
    iat = exprnd(1/lambda,1,n_requests); iat(1)=0;
    [E,V,link,link_list] = construct_routing_topology(numel(clients),n_server);
    
    %% compute R
    tpr = BPRR_heuristic_general_extended(1,128,initial_delay,RTT,RTT_input,...
      tau,tau_in,L,sm,sc,M);
    R_max = floor((sum(M)-sm*(L+n_server))/(sc*(L+n_server)));
    R     = min(round(lambda*tpr + sqrt(lambda*tpr)), R_max);
    
    %% Proposed
    t0 = tic;
    [sa,sm_,~,opt_ord] = CG_BP(RTT,tau,L,sm,sc,M,R,server_types);
    [~,vp,~,~,~] = WS_RR(sa,sm_,iat,128,initial_delay,RTT,RTT_input,...
                       tau,tau_in,L,sm,sc,M,E,link,allocation_delay_prop);
    Tp(mc)=vp/(n_requests*128);
    RT_prop(mc) = toc(t0);
    
    %% Petals random
    ord = randperm(n_server);
    t1 = tic;
    [~,soln_a,soln_m,vpet,~,~,~] = Petals_online(server_types,cache_bytes_per_block,...
       num_kv,ord,throughput,RTT_raw,overhead_delay_petals,alloc_delay,...
       d_model,iat,128,initial_delay,RTT,RTT_input,tau,tau_in,...
       L,sm,sc,M,E,link,allocation_delay_petals);
    Tpet(mc)=vpet/(n_requests*128);
    RT_petals(mc) = toc(t1);
    
    %% Petals opt order
    t2 = tic;
    [~,~,~,vpo,~,~,~] = Petals_online(server_types,cache_bytes_per_block,...
       num_kv,opt_ord,throughput,RTT_raw,overhead_delay_petals,alloc_delay,...
       d_model,iat,128,initial_delay,RTT,RTT_input,tau,tau_in,...
       L,sm,sc,M,E,link,allocation_delay_petals);
    Tord(mc)=vpo/(n_requests*128);
    RT_ord(mc) = toc(t2);
    
    %% Petals opt RR
    t3 = tic;
    [~,vprr,~,~] = Petals_optimized_routing(cache_bytes_per_block,soln_a,soln_m,...
       iat,128,initial_delay,RTT,RTT_input,tau,tau_in,L,sm,sc,M,...
       E,link,allocation_delay_petals);
    Trr(mc)=vprr/(n_requests*128);
    RT_rr(mc) = toc(t3);
    
    %% Petals opt number
    t4 = tic;
    [~,~,~,vpn,~,~,~] = Petals_online_hacking(server_types,sm_,...
       num_kv,ord,throughput,RTT_raw,overhead_delay_petals,alloc_delay,...
       d_model,iat,128,initial_delay,RTT,RTT_input,tau,tau_in,...
       L,sm,sc,M,E,link,allocation_delay_petals);
    Tnum(mc)=vpn/(n_requests*128);
    RT_num(mc) = toc(t4);
  end
  
  % store mean & std
  avg_prop(i)=mean(Tp);      std_prop(i)=std(Tp);
  avg_petals(i)=mean(Tpet);  std_petals(i)=std(Tpet);
  avg_opt_ord(i)=mean(Tord); std_opt_ord(i)=std(Tord);
  avg_opt_rr(i)=mean(Trr);   std_opt_rr(i)=std(Trr);
  avg_opt_num(i)=mean(Tnum); std_opt_num(i)=std(Tnum);
  avg_runtime_prop(i)      = mean(RT_prop);
  avg_runtime_petals(i)    = mean(RT_petals);
  avg_runtime_opt_order(i) = mean(RT_ord);
  avg_runtime_opt_rr(i)    = mean(RT_rr);
  avg_runtime_opt_num(i)   = mean(RT_num);
end

%% — Plot inference‐time vs λ with error bars —
lambda_values_plot = lambda_values *1000;
data = [ avg_prop; avg_petals; avg_opt_ord; avg_opt_rr; avg_opt_num ];
stds = [ std_prop; std_petals; std_opt_ord; std_opt_rr; std_opt_num ];
methods = {'Proposed','Petals','Opt Order','Opt RR','Opt Num'};
methodOrder     = [2,5,3,4,1];
labelsReordered = methods(methodOrder);
data_reordered  = data(methodOrder, :)';
std_reordered   = stds(methodOrder, :)';
% Convert from milliseconds to seconds
data_reordered = data_reordered / 1000;
std_reordered  = std_reordered  / 1000;
figure;
x = 1:num_N;
h = bar(x, data_reordered, 'grouped');
hold on;
for k = 1:numel(h)
  xpts = h(k).XEndPoints;
  ypts = h(k).YEndPoints;
  err  = std_reordered(:,k);
  errorbar(xpts, ypts, err, 'k','LineStyle','none','LineWidth',2,'CapSize',12);
end
hold off;

set(gcf,'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman');
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 24;
% ax.YLim = [0 5];
xlabel('Request rate (requests/s)','Interpreter','latex','FontSize',24,'FontWeight','bold');
ylabel('Avg per-token inference time (s)','Interpreter','latex','FontSize',24,'FontWeight','bold');
xticks(x);
xticklabels(string(lambda_values_plot));
grid on;
legend(h, labelsReordered,'FontSize',24,'Location','Northwest');
exportgraphics(gcf,sprintf('Inference_vs_lambda_%s.pdf',network_name),'ContentType','vector');

%% — Plot runtime vs λ (log scale) —
runtime_data = [ avg_runtime_prop; avg_runtime_petals; avg_runtime_opt_order; avg_runtime_opt_rr; avg_runtime_opt_num ];
runtime_reordered = runtime_data(methodOrder,:)';

figure;
h2 = bar(x, runtime_reordered, 'grouped','BarWidth',0.6);
ax = gca;
ax.TickLabelInterpreter = 'latex';
ax.FontSize = 24;
ax.YScale   = 'log';
% ax.YLim = [0.5e-2 1e0];
xlabel('Request rate (requests/s)','Interpreter','latex','FontSize',24,'FontWeight','bold');
ylabel('Avg runtime (s)','Interpreter','latex','FontSize',24,'FontWeight','bold');
xticks(x);
xticklabels(string(lambda_values_plot));
grid on;
legend(h2, labelsReordered,'FontSize',24,'Location','NorthWest');
exportgraphics(gcf,sprintf('Runtime_vs_lambda_%s.pdf',network_name),'ContentType','vector');
