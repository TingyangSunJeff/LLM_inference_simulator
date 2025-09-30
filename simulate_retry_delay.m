function total_wait = simulate_retry_delay(wait_interval)
    % Simulate exponential backoff retries over a wait interval
    min_backoff = 1000;
    max_backoff = 60000;
    attempt_no = 0;
    total_backoff = 0;
    while total_backoff < wait_interval
        delay = min(min_backoff * 2^attempt_no, max_backoff);
        total_backoff = total_backoff + delay;
        attempt_no = attempt_no + 1;
    end
    total_wait = total_backoff; % This is the total time spent waiting (may be >= wait_interval)
end