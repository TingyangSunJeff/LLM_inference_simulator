function [soln_f,soln_a,soln_m,soln_val] = greedy_stage_assignment(m, n_server,lc,RTT,tau,L,sm,sc,M,E,V,R,link,link_list)
% Special case (single-client, identical #blocks per server): GSA algorithm
% m: #blocks per server
n_client = size(RTT,1); % should have n_client = 1
if n_client ~= 1
    error(['greedy_stage_assignment: require n_client = 1, receive n_client = ' num2str(n_client)]);
end
n_request_client = R; 
max_sessions = floor((M-sm*m)./(sc*m)); % max #inference sessions each server can handle, i.e., \bar{f}_j
time = RTT' + tau*m; % inference time at each server
t0 = 1+max(time); % an upper bound on the inference time for the virtual 'server'
n = ceil(L/m); % #stages
S = zeros(n_server,1); % S(j)\in {1,...,n} is the stage server j is assigned to
C = zeros(n,1); % total capacity (in #requests) per stage
T = ones(n,1) * t0*R; % sum inference time per stage
[~, order] = sort(time); 
for j = 1:n_server % for each server order(j)
    underserved = find(C<R);
    if ~isempty(underserved)
        [~,k] = max(T(underserved)); k = underserved(k); % stage with the largest inference time among the underserved stages (i.e., k^*)
    else % the rest of the servers will not be used, assign them to balance throughput
        [~,k] = min(C); 
    end
    S(order(j)) = k; % assign j-th fastest server to stage k
    T(k) = T(k) - (t0-time(order(j)))*min(max(0,R-C(k)), max_sessions(order(j)));
    C(k) = C(k) + max_sessions(order(j));
%     % optionally: early stop if all stages are fully served (the rest of
%     % the servers will not be used)
%     if all(C>=R)
%         break;
%     end
end
soln_a = (S-1).*m+1; % starting block per server
soln_m = ones(n_server,1) * m; % #blocks per server
soln_m(S==n) = L - (n-1)*m; % servers in the last stage will only process the rest of blocks
% greedy request routing:
soln_f = zeros(E,R); 
order_stage = cell(n,1); % servers in each stage sorted in increasing order of inference time
residual_sessions = max_sessions; % remaining #requests each server can handle
for k = 1:n
    I = find(S==k); 
    [~,order_stage{k}] = sort(time(I)); order_stage{k} = I(order_stage{k}); 
end
c = 1; c1 = 1+n_server+1; % S-client, D-client
for r=1:R
    i = c; 
    for k=1:n
        I = find(residual_sessions(order_stage{k})>0,1,'first'); I = order_stage{k}(I); % 'I' is the next fastest server in stage k with residual capacity
        j = I+n_client; % global server index
        soln_f(link(i,j), r) = 1; % sequentially, route each request to the fastest server with available capacity in the next stage
        residual_sessions(I) = residual_sessions(I)-1; 
        i = j; 
    end
    soln_f(link(i,c1), r) = 1; 
end

% evaluate the total inference time under this solution:
[soln_val] = total_inference_time(soln_f,soln_a,soln_m, n_client,n_server,n_request_client,lc,RTT,tau,sm,sc,M,R,link_list); 
% sanity check: soln_val <= lc*sum(T)
EPS = 10^(-4);
if soln_val - lc*sum(T) > EPS
    error(['Error in greedy_stage_assignment: actual total inference time should <= lc*sum(T), as the last stage may process fewer blocks']);
end




end