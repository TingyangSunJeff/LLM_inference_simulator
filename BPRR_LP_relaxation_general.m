function [soln_f,soln_a,soln_m,soln_val] = BPRR_LP_relaxation_general(n_client,n_server,n_request_client,lc,RTT,tau,L,sm,sc,M,E,V,R,link,link_list)
% this algorithm does NOT work.

% MILP formulation of BPRR (general case):
n_var = R*E*5+n_server*2; % #variables
n_cons_ineq = 2*n_server + 14*R*E; % #constraints "<="
n_cons_eq = R*V; % #constraints "="
cost = zeros(n_var,1); % coefficient in objective function: min cost*x
A = zeros(n_cons_ineq,n_var); % constraint: A*x <= b
b = zeros(n_cons_ineq,1);
Aeq = zeros(n_cons_eq,n_var);
beq = zeros(n_cons_eq,1);
% (13a):
for c=1:n_client
    for r = 1:n_request_client
        for l=1:E
            if link_list(l,1)>0 && link_list(l,2)<=n_client+n_server % if link (i,j) exists and j is a real server (not a D-client)
                i = link_list(l,1); j = link_list(l,2); 
                j_server = j-n_client; % convert global index to local index among servers
                r1 = (c-1)*n_request_client + r; % global request index of this request
                t_cj = RTT(c,j_server);
                tau_j = tau(j_server); 
                cost((r1-1)*E+l) = lc*t_cj; % coefficient of f^r_{ij}
                cost(R*E+2*n_server + (r1-1)*E+l) = lc*tau_j; % coefficient of alpha^r_{ij}
                cost(R*E*2+2*n_server + (r1-1)*E+l) = -lc*tau_j; % coefficient of beta^r_{ij}
                cost(R*E*3+2*n_server + (r1-1)*E+l) = lc*tau_j; % coefficient of gamma^r_{ij}
                cost(R*E*4+2*n_server + (r1-1)*E+l) = -lc*tau_j; % coefficient of delta^r_{ij}
            end
        end
    end
end
% (13b):
b(1:n_server) = M; 
for j=1:n_server
    A(j,R*E+n_server+j) = sm; % coeff. of m_j
    j_node = n_client+j; % convert j to its global node index
    for c=1:n_client
        for r=1:n_request_client
            r1 = (c-1)*n_request_client + r; % global request index 
            for i=1:n_client+n_server
                if link(i,j_node)>0 % if the link exists
                    A(j,alpha(r1,i,j_node, R,E,n_server,link)) = sc;
                    A(j,gamma(r1,i,j_node, R,E,n_server,link)) = sc;
                    A(j,beta(r1,i,j_node, R,E,n_server,link)) = -sc;
                    A(j,delta(r1,i,j_node, R,E,n_server,link)) = -sc; 
                end
            end
        end
    end
end
% (13c):
for c=1:n_client
    for r=1:n_request_client
        r1 = (c-1)*n_request_client + r; % global request index
        for j=1:V
            if j==c % j is S-client c
                beq((r1-1)*V+j) = 1;
            elseif j==c+n_client+n_server % j is D-client c'
                beq((r1-1)*V+j) = -1;
            end%otherwise, beq = 0
            for i=1:V
                if link(j,i)>0
                    Aeq((r1-1)*V+j, f(r1,j,i, E,link)) = 1;
                end
                if link(i,j)>0
                    Aeq((r1-1)*V+j, f(r1,i,j, E,link)) = -1; 
                end
            end
        end
    end
end
% (13d): n_server+[1:n_server]
b(n_server+1:2*n_server)=L+1; 
for j=1:n_server
    A(n_server+j, a(j, R,E)) = 1;
    A(n_server+j, m(j, R,E,n_server)) = 1;
end
% (13e): 2*n_server+[1:R*E]
for r=1:R % global request index
    for l=1:E
        if link_list(l,1)>0 % if link l exists
            i = link_list(l,1); j = link_list(l,2);  
            if i<=n_client % i is an S-client
                b(2*n_server + (r-1)*E+l) = 1;
                A(2*n_server + (r-1)*E+l, alpha(r,i,j, R,E,n_server,link)) = 1; 
            else % i must be a server
                % b(2*n_server + (r-1)*E+l) = 0;
                A(2*n_server + (r-1)*E+l, a(i-n_client, R,E)) = -1; % 'i-n_client' is the local server index of node i
                A(2*n_server + (r-1)*E+l, m(i-n_client, R,E,n_server)) = -1; 
                A(2*n_server + (r-1)*E+l, alpha(r,i,j, R,E,n_server,link)) = 1;
            end
        end
    end
end
% (13f): 2*n_server+R*E+[1:R*E]
for r=1:R % global request index
    for l=1:E
        if link_list(l,1)>0
            i = link_list(l,1); j = link_list(l,2);  
            if j>n_client+n_server % j is a D-client
                b(2*n_server+R*E+(r-1)*E+l) = L+1; 
                A(2*n_server+R*E+(r-1)*E+l, beta(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = 1;
            else % j must be a server
                b(2*n_server+R*E+(r-1)*E+l) = -1; 
                A(2*n_server+R*E+(r-1)*E+l, a(j-n_client, R,E)) = -1; 
                A(2*n_server+R*E+(r-1)*E+l, m(j-n_client, R,E,n_server)) = -1; 
                A(2*n_server+R*E+(r-1)*E+l, beta(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = 1;
            end
        end
    end
end
% (9): 2*n_server+2*R*E+[1:3*R*E]
for r=1:R % global requst index
    for l=1:E
        if link_list(l,1)>0
            i = link_list(l,1); j = link_list(l,2);  % global indices
            if j>n_client+n_server % j is D-client
                A(2*n_server+2*R*E+(r-1)*E+l, f(r,i,j, E,link)) = -(L+1);
                A(2*n_server+2*R*E+(r-1)*E+l, alpha(r,i,j, R,E,n_server,link)) = 1; 
                A(2*n_server+4*R*E+(r-1)*E+l, f(r,i,j, E,link)) = (L+1);
                A(2*n_server+4*R*E+(r-1)*E+l, alpha(r,i,j, R,E,n_server,link)) = -1;
            else % j is real server
                A(2*n_server+2*R*E+(r-1)*E+l, f(r,i,j, E,link)) = -(L+1);
                A(2*n_server+2*R*E+(r-1)*E+l, alpha(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+3*R*E+(r-1)*E+l, a(j-n_client, R,E)) = -1;
                A(2*n_server+3*R*E+(r-1)*E+l, alpha(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+4*R*E+(r-1)*E+l, a(j-n_client, R,E)) = 1;
                A(2*n_server+4*R*E+(r-1)*E+l, f(r,i,j, E,link)) = L+1; 
                A(2*n_server+4*R*E+(r-1)*E+l, alpha(r,i,j, R,E,n_server,link)) = -1;
                b(2*n_server+4*R*E+(r-1)*E+l) = L+1; 
            end
        end
    end
end
% (10): 2*n_server+5*R*E+[1:3*R*E]
for r=1:R % global requst index
    for l=1:E
        if link_list(l,1)>0
            i = link_list(l,1); j = link_list(l,2);  % global indices
            if i<=n_client % if i is S-client
                A(2*n_server+6*R*E+(r-1)*E+l, beta(r,i,j, R,E,n_server,link)) = 1;
            else % i is real server
                A(2*n_server+5*R*E+(r-1)*E+l, f(r,i,j, E,link)) = -L;
                A(2*n_server+5*R*E+(r-1)*E+l, beta(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+6*R*E+(r-1)*E+l, a(i-n_client, R,E)) = -1;
                A(2*n_server+6*R*E+(r-1)*E+l, beta(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+7*R*E+(r-1)*E+l, a(i-n_client, R,E)) = 1;
                A(2*n_server+7*R*E+(r-1)*E+l, f(r,i,j, E,link)) = L;
                A(2*n_server+7*R*E+(r-1)*E+l, beta(r,i,j, R,E,n_server,link)) = -1;
                b(2*n_server+7*R*E+(r-1)*E+l) = L; 
            end
        end
    end
end
% (11): 2*n_server+8*R*E+[1:3*R*E]
for r=1:R % global requst index
    for l=1:E
        if link_list(l,1)>0
            i = link_list(l,1); j = link_list(l,2);  % global indices
            if j>n_client+n_server % if j is D-client
                A(2*n_server+8*R*E+(r-1)*E+l, f(r,i,j, E,link)) = -L;
                A(2*n_server+8*R*E+(r-1)*E+l, gamma(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+9*R*E+(r-1)*E+l, gamma(r,i,j, R,E,n_server,link)) = 1;
                b(2*n_server+9*R*E+(r-1)*E+l) = 1;
                A(2*n_server+10*R*E+(r-1)*E+l, f(r,i,j, E,link)) = L;
                A(2*n_server+10*R*E+(r-1)*E+l, gamma(r,i,j, R,E,n_server,link)) = -1;
                b(2*n_server+10*R*E+(r-1)*E+l) = L-1; 
            else % j is real server
                A(2*n_server+8*R*E+(r-1)*E+l, f(r,i,j, E,link)) = -L; 
                A(2*n_server+8*R*E+(r-1)*E+l, gamma(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+9*R*E+(r-1)*E+l, m(j-n_client, R,E,n_server)) = -1;
                A(2*n_server+9*R*E+(r-1)*E+l, gamma(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+10*R*E+(r-1)*E+l, m(j-n_client, R,E,n_server)) = 1;
                A(2*n_server+10*R*E+(r-1)*E+l, f(r,i,j, E,link)) = L;
                A(2*n_server+10*R*E+(r-1)*E+l, gamma(r,i,j, R,E,n_server,link)) = -1;
                b(2*n_server+10*R*E+(r-1)*E+l) = L;
            end
        end
    end
end
% (12): 2*n_server+11*R*E+[1:3*R*E]
for r=1:R % global requst index
    for l=1:E
        if link_list(l,1)>0
            i = link_list(l,1); j = link_list(l,2);  % global indices
            if i<=n_client % if i is S-client
                A(2*n_server+11*R*E+(r-1)*E+l, f(r,i,j, E,link)) = -L;
                A(2*n_server+11*R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+12*R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = 1;
                b(2*n_server+12*R*E+(r-1)*E+l) = 1;
                A(2*n_server+13*R*E+(r-1)*E+l, f(r,i,j, E,link)) = L;
                A(2*n_server+13*R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = -1;
                b(2*n_server+13*R*E+(r-1)*E+l) = L-1;
            else % i is real server
                A(2*n_server+11*R*E+(r-1)*E+l, f(r,i,j, E,link)) = -L;
                A(2*n_server+11*R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+12*R*E+(r-1)*E+l, m(i-n_client, R,E,n_server)) = -1;
                A(2*n_server+12*R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = 1;
                A(2*n_server+13*R*E+(r-1)*E+l, m(i-n_client, R,E,n_server)) = 1;
                A(2*n_server+13*R*E+(r-1)*E+l, f(r,i,j, E,link)) = L;
                A(2*n_server+13*R*E+(r-1)*E+l, delta(r,i,j, R,E,n_server,link)) = -1;
                b(2*n_server+13*R*E+(r-1)*E+l) = L;
            end
        end
    end
end
% lower/upper bounds:
lb = zeros(n_var,1);
lb([R*E+1:R*E+2*n_server]) = 1;
ub = [ones(R*E,1); ones(2*n_server,1)*L; ones(R*E,1)*(L+1); ones(R*E*3,1)*L];
% integer constraints:
intcon = [1:R*E+2*n_server];

% call MILP solver:
tolerance_ILP = 1e-4;
intlinprog_MaxTime = 200; 
options = optimoptions('intlinprog','RelativeGapTolerance',tolerance_ILP,'MaxTime',intlinprog_MaxTime,'Display','iter');
% Matlab's built-in MILP solver:
% soln = intlinprog(cost,intcon,A,b,Aeq,beq,lb,ub,[],options);
% 1) solve the LP relaxation:
[soln,soln_val_LP,~,~] = gurobi_intlinprog(cost,[],A,b,Aeq,beq,lb,ub,[],options); % return value: [x,fval,exitflag,output] 
% convert solution to our matrix form:
soln_f = reshape(soln([1:R*E]), E,R); % soln_f(l,r) = 1 iff (global) request r is routed on link_list(l,:)
soln_a = reshape(soln(R*E+[1:n_server]), n_server,1); % soln_a(j): the first block stored on server j
soln_m = reshape(soln(R*E+n_server+[1:n_server]), n_server,1); % soln_m(j): #blocks stored on server j
% soln_val_LP: lower bound on the minimum total inference time over all the
% R requests (by LP relaxation)

% 2) round soln_a and soln_m to integers:
soln_m = floor(soln_m); 
soln_a = min(round(soln_a), L-soln_m+1);

% 3) compute the corresponding request routing by solving (19):
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

% call MILP solver:
tolerance_ILP = 1e-4;
intlinprog_MaxTime = 200; 
options = optimoptions('intlinprog','RelativeGapTolerance',tolerance_ILP,'MaxTime',intlinprog_MaxTime,'Display','iter');
% Gurobi's MILP solver:
[soln,soln_val,~,~] = gurobi_intlinprog(cost,intcon,A,b,Aeq,beq,lb,ub,[],options); % return value: [x,fval,exitflag,output] 
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

% evaluate the total inference time under this solution:
[soln_val] = total_inference_time(soln_f,soln_a,soln_m, n_client,n_server,n_request_client,lc,RTT,tau,sm,sc,M,R,link_list);

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

%%%%%%%%%%%%%%%%%%%%%%%% auxiliary functions:

function index = f(r,i,j, E,link) % here i,j are global node index; r is global request index
% index of f^r_{ij} in the overall solution vector
    index = (r-1)*E+link(i,j); 
end

function index = a(j, R,E) % here j is a local index among servers
% index of a_j
    index = R*E+j;
end

function index = m(j, R,E,n_server) % j is local index among servers
% index of m_j
    index = R*E+n_server+j;
end

function index = alpha(r,i,j, R,E,n_server,link) % i,j are global node index; r is global request index
% index of alpha^r_{ij}
    index = R*E+2*n_server + (r-1)*E+link(i,j);
end

function index = beta(r,i,j, R,E,n_server,link) % i,j are global node index; r is global request index
% index of beta^r_{ij}
    index = R*E*2+2*n_server + (r-1)*E+link(i,j);
end

function index = gamma(r,i,j, R,E,n_server,link) % i,j are global node index; r is global request index
% index of gamma^r_{ij}
    index = R*E*3+2*n_server + (r-1)*E+link(i,j);
end

function index = delta(r,i,j, R,E,n_server,link) % i,j are global node index; r is global request index
% index of delta^r_{ij}
    index = R*E*4+2*n_server + (r-1)*E+link(i,j);
end
