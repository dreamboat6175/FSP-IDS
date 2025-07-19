function results = initializeResults(config, n_agents)
    %% initializeResults - 初始化仿真结果结构体
    % 输入:
    %   config - 配置结构体
    %   n_agents - 智能体数量
    % 输出:
    %   results - 初始化的结果结构体
    
    % 基本参数
    results.n_agents = n_agents;
    results.n_iterations = config.n_iterations;
    results.n_episodes_per_iter = config.n_episodes_per_iter;
    
    % 主要性能指标矩阵 [n_agents x n_iterations]
    results.radi = zeros(n_agents, config.n_iterations);
    results.resource_efficiency = zeros(n_agents, config.n_iterations);
    results.allocation_balance = zeros(n_agents, config.n_iterations);
    results.convergence_metrics = zeros(n_agents, config.n_iterations);
    
    % 奖励历史 [n_agents x n_iterations]
    results.defender_rewards = zeros(n_agents, config.n_iterations);
    results.attacker_rewards = zeros(1, config.n_iterations);
    
    % 每个episode的详细数据 [n_agents x n_episodes_per_iter x n_iterations]
    results.episode_rewards = zeros(n_agents, config.n_episodes_per_iter, config.n_iterations);
    results.episode_detection_rates = zeros(n_agents, config.n_episodes_per_iter, config.n_iterations);
    results.episode_resource_usage = zeros(n_agents, config.n_episodes_per_iter, config.n_iterations);
    
    % 累积统计
    results.cumulative_rewards = zeros(n_agents, config.n_iterations);
    results.success_rates = zeros(n_agents, config.n_iterations);
    results.false_positive_rates = zeros(n_agents, config.n_iterations);
    results.false_negative_rates = zeros(n_agents, config.n_iterations);
    
    % 策略收敛性指标
    results.policy_changes = zeros(n_agents, config.n_iterations);
    results.strategy_diversity = zeros(n_agents, config.n_iterations);
    
    % 系统级指标
    results.system_security_level = zeros(1, config.n_iterations);
    results.total_resource_consumption = zeros(1, config.n_iterations);
    results.network_coverage = zeros(1, config.n_iterations);
    
    % FSP相关指标
    results.exploitability = zeros(1, config.n_iterations);
    results.nash_conv = zeros(1, config.n_iterations);
    
    % 时间记录
    results.iteration_times = zeros(1, config.n_iterations);
    results.start_time = now;
    
    % 配置记录
    results.config = config;
    
    % 元数据
    results.matlab_version = version;
    results.creation_time = datestr(now);
    results.status = 'initialized';
    
    fprintf('✓ 结果结构体初始化完成 (%d个智能体, %d次迭代)\n', n_agents, config.n_iterations);
end