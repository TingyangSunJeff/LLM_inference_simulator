function [soln_f,soln_a,soln_m,soln_val] = BPRR_MILP_general(n_client,n_server,n_request_client,lc,RTT,tau,L,sm,sc,M,E,V,R,link,link_list)
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
intlinprog_MaxTime = 200; % max running time in seconds
options = optimoptions('intlinprog','RelativeGapTolerance',tolerance_ILP,'MaxTime',intlinprog_MaxTime,'Display','iter');
% Matlab's built-in MILP solver:
% soln = intlinprog(cost,intcon,A,b,Aeq,beq,lb,ub,[],options);
% Gurobi's MILP solver:
[soln,soln_val,~,~] = gurobi_intlinprog(cost,intcon,A,b,Aeq,beq,lb,ub,[],options); % return value: [x,fval,exitflag,output] 

% convert solution to our matrix form:
soln_f = reshape(soln([1:R*E]), E,R); % soln_f(l,r) = 1 iff (global) request r is routed on link_list(l,:)
soln_a = reshape(soln(R*E+[1:n_server]), n_server,1); % soln_a(j): the first block stored on server j
soln_m = reshape(soln(R*E+n_server+[1:n_server]), n_server,1); % soln_m(j): #blocks stored on server j
% soln_val: the total inference time over all the R requests

isdebug = 0; 
if isdebug
    %% debug: check the selecting "routing paths" (as node sequence)
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
