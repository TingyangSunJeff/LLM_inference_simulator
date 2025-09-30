function [soln_val] = total_inference_time(soln_f,soln_a,soln_m, n_client,n_server,n_request_client,lc,RTT,tau,sm,sc,M,R,link_list)
% evaluate the total time to complete all the R inference sessions:
% soln_f: E*R, soln_f(l,r) = 1 iff (global) request r is routed on link_list(l,:)
% soln_a: n_server*1, soln_a(j): the first block stored on server j
% soln_m: n_server*1, soln_m(j): #blocks stored on server j
% soln_val: the total inference time over all the R requests
% Note: link_list(l,:) is the (global) node indices for l-th link, where
% the global node incides are defined as:
% S-clients: 1,...,n_client; Servers: n_client+1,...n_client+n_server; D-clients: n_client+n_server+1,...,n_client+n_server+n_client
EPS = 10^(-4); % precision to handle numerical errors in soln_f

time = zeros(2,R); % time(1,r), time(2,r): start/completion time for each (global) request r
traversing_session = zeros(n_server,R); % traversing_session(j,r) = 1 iff server j is traversed by session r (i.e., for global request r)
memory_left = (M - soln_m*sm) .* ones(n_server, 2*R); % record the remaining memory at each point of change (unit: GiB)
timestamp = inf(n_server, 2*R); % memory_left(j,k) is the memory left right after time timestamp(j,k), i.e., during interval [timestamp(j,k), timestamp(j,k+1))
% memory_left(:,1) = M - soln_m*sm; % initial memory left after storing model parameters
timestamp(:,1) = 0; 
memory_needed = zeros(n_server,R); % memory_needed(j,r): memory needed at server j for the attention cache of session r
% total_time = 0; % total response time over all the R requests
delayed_start = 0; 
for c=1:n_client
    for r1 = 1:n_request_client
        r = (c-1)*n_request_client + r1; % global request index of this request
        time_per_token = 0; % time for each traversal of the path for request r (i.e., for generating each token for this request)        
        for l = find(soln_f(:,r)'>EPS) % for each link traversed by path for request r1
            if link_list(l,2) <= n_client+n_server % if this is not the last hop (the last hop introduces 0 delay)
                j = link_list(l,2)-n_client; % j is a local server index (in 1,...,n_server)
                traversing_session(j,r) = 1; 
                if link_list(l,1) <= n_client % if this is the first hop
                    time_per_token = time_per_token + RTT(c,j) + tau(j)*(soln_a(j)+soln_m(j)-1);
                    memory_needed(j,r) = memory_needed(j,r) + sc*(soln_a(j)+soln_m(j)-1); 
                else % this is not the first hop
                    i = link_list(l,1)-n_client; % i is a local server index (in 1,...,n_server)
                    time_per_token = time_per_token + RTT(c,j) + tau(j)*(soln_a(j)+soln_m(j)-soln_a(i)-soln_m(i));
                    memory_needed(j,r) = memory_needed(j,r) + sc*(soln_a(j)+soln_m(j)-soln_a(i)-soln_m(i));
                end
            end
        end
        time_r = time_per_token*lc; % total time to process request r once the session starts
        % To-do: check when session r can start (all traversed servers have
        % enough memory) to set start/completion time
        startime = 0; % first time that all traversed servers have enough memory
        endtime = inf; % (continuous) last time that all traversed servers have enough memory after 'startime'
        while 1
        for j = find(traversing_session(:,r)') % for each server traversed by session r
            % find the earliest time the available memory at server j >=
            % memory_needed(j,r) 
            k = find(timestamp(j,:) >= startime & memory_left(j,:) >= memory_needed(j,r),1,'first'); 
            if k > 1 && memory_left(j,k-1) >= memory_needed(j,r)
                k = k - 1; 
            end
            startime = max(startime, timestamp(j,k));
            k2 = find(timestamp(j,k+1:end) > startime,1,'first') + k; % index for the earliest possible stopping time due to insufficient resource at server j
            if any(memory_left(j,k2:end) < memory_needed(j,r))
                k1 = find(memory_left(j,k2:end) < memory_needed(j,r),1,'first') + k2-1; % timestamp(j,k1) is the first time after timestamp(j,k) that server j does not have enough memory
                endtime = min(endtime, timestamp(j,k1));
            end % otherwise, endtime = min(endtime, inf) 
        end
        if startime > 0 % the given resource allocation is not feasible (i.e., cannot run all inference sessions concurrently)
%             disp(['total_inference_time: request ' num2str(r) ' postponed by ' num2str(startime) ' ms'])
            delayed_start = 1; 
        end
        if endtime-startime < time_r % cannot run request 'r1' till finish within time window [startime,endtime] 
            % Q: In this case, will the session be partially completed, or will it be postponed? 
            % A: Currently assume it will be postponed (a session must be run continuously). 
%             disp(['total_inference_time: try to start request ' num2str(r1) ' for client ' num2str(c) ' after time ' num2str(endtime/1000) ' s']);
            startime = endtime; % reset earliest start time to previous 'endtime'
            endtime = inf; 
        else
            time(1,r) = startime; % starting time of session r
            time(2,r) = startime + time_r; % ending time of session r
            % update memory_left:
            for j = find(traversing_session(:,r)') 
                k = find(timestamp(j,:) <= startime,1,'last');
                if timestamp(j,k) < startime
                    timestamp(j,k+2:end) = timestamp(j,k+1:end-1); 
                    timestamp(j,k+1) = startime; % insert a new timestamp for the starting of a session
                    memory_left(j,k+2:end) = memory_left(j,k+1:end-1);
                    memory_left(j,k+1) = memory_left(j,k); 
                    k_start = k+1;
                else% timestamp(j,k) == startime
                    k_start = k; % request r starts using memory at server j at timestamp(j,k_start)
                end
                k = find(timestamp(j,:) <= time(2,r),1,'last');
                if timestamp(j,k) < time(2,r)
                    timestamp(j,k+2:end) = timestamp(j,k+1:end-1); 
                    timestamp(j,k+1) = time(2,r); % insert a new timestamp for the ending of a session
                    memory_left(j,k+2:end) = memory_left(j,k+1:end-1); 
                    memory_left(j,k+1) = memory_left(j,k); 
                    k_end = k+1;                    
                else% timestamp(j,k) == time(2,r)
                    k_end = k;
                end% session r consumes memory at server j from timestamp(j,k_start) (inclusive) to timestamp(j,k_end) (exclusive)
                for k=k_start:k_end-1
                    memory_left(j,k) = memory_left(j,k) - memory_needed(j,r); 
                end
            end
            break;
        end          
        end
    end
end

soln_val = sum(time(2,:)); % total time in serving all the requests, including waiting time (assuming all requests are submitted at time 0)

if delayed_start
    disp(['total_inference_time: cannot start all the sessions concurrently'])
end

end