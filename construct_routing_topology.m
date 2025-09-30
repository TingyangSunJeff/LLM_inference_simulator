function [E,V,link,link_list] = construct_routing_topology(n_client,n_server)
% construct logical routing topology:
E = n_server^2 + 2*n_client*n_server; % #links |E| (directed)
V = n_server + 2*n_client; % #nodes |V|; S-clients: 1,...,n_client; Servers: n_client+1,...n_client+n_server; D-clients: n_client+n_server+1,...,n_client+n_server+n_client
link = zeros(V); % if link(i,j)>0, link(i,j) is index of link (i,j) (i,j must be global node indices); if link(i,j)=0, link (i,j) does not exist
link_list = zeros(E,2); % if link_list(l,1)>0, link_list(l,:) is the (global) node indices for l-th link
for i=1:n_client 
    for j=1:n_server
        link(i,j+n_client) = n_server*(i-1) + j; % S-client -> server
        link_list(link(i,j+n_client),:) = [i,j+n_client];
        link(j+n_client,i+n_client+n_server) = n_client*n_server + n_server^2 + n_server*(i-1) + j; % server -> D-client
        link_list(link(j+n_client,i+n_client+n_server),:) = [j+n_client,i+n_client+n_server]; 
    end
end
for i=1:n_server
    for j=1:n_server
        if i~=j
            link(i+n_client,j+n_client) = n_client*n_server + n_server*(j-1)+i; % server -> server
            link_list(link(i+n_client,j+n_client),:) = [i+n_client,j+n_client]; 
        end
    end
end
end