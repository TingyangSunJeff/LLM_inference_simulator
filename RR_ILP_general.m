function [soln_f,soln_val] = RR_ILP_general(soln_m,soln_a, n_client,n_server,n_request_client,lc,RTT,tau,L,sm,sc,M,E,V,R,link,link_list,soln_f_init)
% compute the optimal request routing under given block placement (soln_m,soln_a) by solving (19):
% Note: This is used for ablation study (to compare with Petals' RR under
% the same block placement). 

n_var_RR = n_client*E;

% (19a):
cost = zeros(n_var_RR,1);
for c = 1:n_client
    for l = 1:E
        if link_list(l,1)>0 % if link l exists
            i = link_list(l,1); j = link_list(l,2); 
            if j<=n_client+n_server % if not the last hop (cost = 0 for the last hop)
                j_server = j-n_client; % convert global index to local index among servers
                if i<=n_client % if the first hop                    
                    cost((c-1)*E+l) = lc*(RTT(c,j_server)+tau(j_server)*(soln_a(j_server)+soln_m(j_server)-1));
                else % a server-server hop
                    i_server = i-n_client; 
                    cost((c-1)*E+l) = lc*(RTT(c,j_server)+tau(j_server)*(soln_a(j_server)+soln_m(j_server)-soln_a(i_server)-soln_m(i_server))); 
                end
            end
        end
    end
end
% (19b):
A = zeros(n_server,n_var_RR);
b = zeros(n_server,1);
for j=1:n_server
    b(j) = M(j) - sm*soln_m(j); 
    j_node = j+n_client; % global index
    for c=1:n_client
        for i=1:n_client+n_server
            if link(i,j_node) > 0 % if the link exists
                if i<=n_client % if the first hop
                    A(j,(c-1)*E+link(i,j_node)) = sc*(soln_a(j)+soln_m(j)-1);
                else % i is a server
                    i_server = i-n_client; 
                    A(j,(c-1)*E+link(i,j_node)) = sc*(soln_a(j)+soln_m(j)-soln_a(i_server)-soln_m(i_server));
                end
            end
        end
    end
end
% (19c):
Aeq = zeros(n_client*V,n_var_RR);
beq = zeros(n_client*V,1);
for c=1:n_client
    for j=1:V
        if j==c % j is S-client c
            beq((c-1)*V+j) = n_request_client;
        elseif j==c+n_client+n_server % j is D-client c'
            beq((c-1)*V+j) = -n_request_client;
        end%otherwise, beq = 0
        for i=1:V
            if link(j,i)>0
                Aeq((c-1)*V+j, (c-1)*E+link(j,i)) = 1;
            end
            if link(i,j)>0
                Aeq((c-1)*V+j, (c-1)*E+link(i,j)) = -1;
            end
        end
    end
end
% (19d)-(19e):
lb = zeros(n_var_RR,1);
ub = zeros(n_var_RR,1);
for l = 1:E
    if link_list(l,1)>0 % if link l exists
        i = link_list(l,1); j = link_list(l,2);
        if i <= n_client % if first hop
            e = (soln_a(j-n_client)==1 && soln_m(j-n_client)>=1);
        elseif j > n_client+n_server % last hop
            e = (soln_a(i-n_client)+soln_m(i-n_client) == L+1);
        else % middle hop
            e = (soln_a(j-n_client)<=soln_a(i-n_client)+soln_m(i-n_client) && soln_a(i-n_client)+soln_m(i-n_client)<=soln_a(j-n_client)+soln_m(j-n_client)-1);
        end % e == 1 iff link (i,j) is feasible
        if e > 0
            ub(([1:n_client]-1).*E+l) = n_request_client; 
        end
    end
end
% integer constraints:
intcon = 1:n_var_RR; 
% initial solution:
soln_init = zeros(n_var_RR,1); % soln_init((c-1)*E+l): #requests of client c initially routed through link l
for c=1:n_client
    for r=1:n_request_client
        r1 = (c-1)*n_request_client + r; % global request index of this request
        I = find(soln_f_init(:,r1)); % links traversed by global request r1
        soln_init((c-1)*E+I) = soln_init((c-1)*E+I)+1;
    end
end

% call MILP solver:
tolerance_ILP = 1e-4;
intlinprog_MaxTime = 200; 
options = optimoptions('intlinprog','RelativeGapTolerance',tolerance_ILP,'MaxTime',intlinprog_MaxTime,'Display','none');
% Gurobi's MILP solver:
if max(A*soln_init-b) <= 0 % if the initial solution is feasible, then use it
    [soln,soln_val,~,~] = gurobi_intlinprog(cost,intcon,A,b,Aeq,beq,lb,ub,soln_init,options); % return value: [x,fval,exitflag,output]
else % if the initial solution is infeasible, then ignore it
    disp(['RR_ILP_general: given initial solution is infeasible! ignored']);
    % if the original routing is infeasible for (19) (i.e., cannot run all sessions
    % concurrently), then do not compare it with the optimal solution to
    % (19) (as optimal concurrent inference may not take shorter time than inference with some waiting)
%     [soln,soln_val,~,~] = gurobi_intlinprog(cost,intcon,A,b,Aeq,beq,lb,ub,[],options);     
%     if isempty(soln) % if the ILP is infeasible:
        soln_f = [];
        soln_val = []; 
        return;
%     end
end
% sanity check: this 'soln_val' should equal the soln_val returned by 'total_inference_time'
% if ~isempty(soln_val)
%     disp(['ILP: average time per token = ' num2str(soln_val/R/lc)]);
% end

soln_f_client = reshape(soln,E,n_client); % soln_f_client(l,c): #requests of client c routed through link l

% convert to per-request routing (to evaluate inference time by
% 'total_inference_time.m'
soln_f = zeros(E,R);
for c=1:n_client
    G = zeros(V); % directed adjacency matrix for routing topology; if G(i,j) > 0, G(i,j) is #requests of client c routed on "link" (i,j); G(i,j) = 0 means "link" (i,j) is not used
    for l = find(soln_f_client(:,c)') % for each link traversed by requests of client c
        G(link_list(l,1),link_list(l,2)) = soln_f_client(l,c); 
    end
    for r=1:n_request_client
        r_global = (c-1)*n_request_client + r; 
        v = c;
        while v ~= c+n_client+n_server
            v1 = find(G(v,:),1,'first'); 
            soln_f(link(v,v1),r_global) = 1;
            G(v,v1) = G(v,v1) - 1;
            v = v1;
        end
    end % G should be all 0's after this loop
    if max(G(:)) > 0
        error(['BPRR_heuristic_general: Not all link traversals are used in routing for client ' num2str(c)]);
    end
end

% % evaluate the total inference time under this solution:
% [soln_val] = total_inference_time(soln_f,soln_a,soln_m, n_client,n_server,n_request_client,lc,RTT,tau,sm,sc,M,R,link_list);

isdebug = 0; 
if isdebug
    %% debug: check the selected "routing paths" (as node sequence)
    Path = cell(R,1);
    for c = 1:n_client
        for r1 = 1:n_request_client
            r = (c-1)*n_request_client + r1;
            links = find(soln_f(:,r));
            Path{r} = zeros(1,1+length(links));
            links_list = link_list(links,:);
            Path{r}(1) = c;
            for i = 2:length(links)+1
                Path{r}(i) = links_list(find(links_list(:,1)==Path{r}(i-1)),2);
            end
        end
    end
end


end