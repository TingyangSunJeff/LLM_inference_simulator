function eff_rps = compute_throughput(throughputs, s, num_blocks)
% COMPUTE_THROUGHPUT  
%   eff_rps = compute_throughput(throughputs, s, num_blocks)
%
%   Inputs:
%     throughputs   n_server×2 matrix
%                    col 1 = GPU-forward rate (requests/sec)
%                    col 2 = network rate   (requests/sec)
%     s             index of the server you’re querying
%     num_blocks    # of model-blocks hosted on server s
%
%   Output:
%     eff_rps       the effective cap in requests/sec, i.e.
%                   min( forward_rps/(avg blocks used),
%                        network_rps )

  % pull out the two caps
  forward_rps = throughputs(s,1);
  network_rps = throughputs(s,2);

  % we assume a request on average uses (1 + num_blocks)/2 blocks
  avg_blocks = (1 + num_blocks) / 2;

  % compute-bound throughput
  cap_compute = forward_rps / avg_blocks;

  % the real cap is the minimum of compute and network
  eff_rps = min(cap_compute, network_rps);
end
