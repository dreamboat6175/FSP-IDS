%% main_improved_fsp_simulation.m - 改进的FSP仿真主程序
% =========================================================================
% 使用优化的环境和参数设置，提高检测率
% =========================================================================

%% 初始化
clear all; close all; clc;

% 添加所有子文件夹到MATLAB路径
addpath(genpath(pwd));

%% 主程序
try
    % 1. 加载配置
    fprintf('正在加载配置...\n');
    config = ConfigManager.loadConfig('default_config.json');
    config.learning_rate = 0.15;
    config.epsilon = 0.8;
    config.n_iterations = 1000;
    config.n_episodes_per_iter = 50;
    % === 添加RADI配置 ===
    config.radi = struct();
    config.radi.optimal_allocation = [0.2, 0.2, 0.2, 0.2, 0.2];
    config.radi.weight_computation = 0.3;
    config.radi.weight_bandwidth = 0.2;
    config.radi.weight_sensors = 0.2;
    config.radi.weight_scanning = 0.15;
    config.radi.weight_inspection = 0.15;
    config.radi.threshold_excellent = 0.1;
    config.radi.threshold_good = 0.2;
    config.radi.threshold_acceptable = 0.3;
    config.radi.target_radi = 0.15;
    config.reward = struct();
    config.reward.w_radi = 0.4;
    config.reward.w_efficiency = 0.3;
    config.reward.w_balance = 0.3;
    
    % 2. 初始化日志系统
    logger = Logger(config.log_file);
    logger.info('改进版FSP仿真开始');
    
    % 3. 参数验证
    ConfigManager.validateConfig(config);
    
    % 4. 初始化改进的仿真环境
    fprintf('正在初始化改进的仿真环境...\n');
    
    env =  TCSEnvironment(config); 
    
    % 5. 初始化智能体（使用更小的状态空间）
    fprintf('正在初始化智能体...\n');
    defender_agents = cell(1, length(config.algorithms));
    
    for i = 1:length(config.algorithms)
        switch config.algorithms{i}
            case 'Q-Learning'
                defender_agents{i} = QLearningAgent(sprintf('Q-Learning-%d', i), ...
                                                  'defender', config, ...
                                                  env.state_dim, env.action_dim_defender);
                % 优化Q表初始化
                defender_agents{i}.Q_table = defender_agents{i}.Q_table + 1.0;  % 乐观初始化
                
            case 'SARSA'
                defender_agents{i} = SARSAAgent(sprintf('SARSA-%d', i), ...
                                             'defender', config, ...
                                             env.state_dim, env.action_dim_defender);
                defender_agents{i}.Q_table = defender_agents{i}.Q_table + 1.0;
                
            case 'Double Q-Learning'
                defender_agents{i} = DoubleQLearningAgent(sprintf('DoubleQ-%d', i), ...
                                                       'defender', config, ...
                                                       env.state_dim, env.action_dim_defender);
                defender_agents{i}.Q1_table = defender_agents{i}.Q1_table + 1.0;
                defender_agents{i}.Q2_table = defender_agents{i}.Q2_table + 1.0;
        end
    end
    
    % 攻击者智能体
    attacker_agent = QLearningAgent('Attacker', 'attacker', config, ...
                                 env.state_dim, env.action_dim_attacker);
    
    % 6. 初始化性能监控器
    monitor = PerformanceMonitor(config.n_iterations, length(defender_agents), config);
    
    % 7. 关闭并行计算（避免问题）
    config.use_parallel = false;
    
    % 8. 运行FSP仿真
    fprintf('\n开始改进版FSP迭代仿真...\n');
    fprintf('========================================\n');
    fprintf('配置信息:\n');
    fprintf('- 主站数量: %d\n', config.n_stations);
    fprintf('- 总组件数: %d\n', sum(config.n_components_per_station));
    fprintf('- 迭代次数: %d\n', config.n_iterations);
    fprintf('- 学习率: %.3f\n', config.learning_rate);
    fprintf('- 初始探索率: %.3f\n', config.epsilon);
    fprintf('- 算法: %s\n', strjoin(config.algorithms, ', '));
    fprintf('- 目标RADI: %.3f\n', config.radi.target_radi);
    fprintf('- 最优资源分配: [%.2f, %.2f, %.2f, %.2f, %.2f]\n', ...
            config.radi.optimal_allocation);
    fprintf('========================================\n\n');
    
    % 使用改进的训练循环
    [results, trained_agents] = improvedFSPTraining(env, defender_agents, ...
                                                   attacker_agent, config, monitor, logger);
    
    % 9. 结果分析
    fprintf('\n正在生成分析报告...\n');
    try
        EnhancedReportGenerator.generateEnhancedReport(results, config, trained_agents, env);
    catch ME
        fprintf('报告生成出错: %s\n', ME.message);
        fprintf('使用简化报告生成...\n');
        EnhancedReportGenerator.generateSimpleReport(results, config);
    end
    
    % 10. 保存结果
    DataManager.saveResults(results, config, trained_agents);
    
    logger.info('改进版仿真成功完成');
    fprintf('\n✓ 仿真完成！\n');
    
    % 显示最终性能
    displayFinalPerformance(results);
    
catch ME
    % 错误处理
    if exist('logger', 'var')
        logger.error(sprintf('仿真出错: %s', ME.message));
        logger.error(sprintf('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line));
    else
        fprintf('错误: %s\n', ME.message);
    end
    rethrow(ME);
end

%% 清理
if exist('logger', 'var')
    delete(logger);
end

%% 辅助函数

function [results, trained_agents] = improvedFSPTraining(env, defender_agents, attacker_agent, config, monitor, logger)
    % 改进的FSP训练循环 - RADI版本
    n_agents = length(defender_agents);
    n_iterations = config.n_iterations;
    n_episodes = config.n_episodes_per_iter;
    warmup_iterations = 200;
    original_epsilon = config.epsilon;
    for iter = 1:n_iterations
        if iter <= warmup_iterations
            current_epsilon = 0.8;
        else
            current_epsilon = max(config.epsilon_min, ...
                                original_epsilon * (config.epsilon_decay ^ (iter - warmup_iterations)));
        end
        for i = 1:n_agents
            defender_agents{i}.epsilon = current_epsilon;
        end
        attacker_agent.epsilon = current_epsilon * 0.5;
        episode_results = improvedRunEpisodesRADI(env, defender_agents, attacker_agent, n_episodes, config);
        % --- 修正：逐个agent传入最后一集的metrics ---
        for agent_idx = 1:n_agents
            metrics = struct();
            metrics.resource_allocation = squeeze(episode_results.resource_allocations(end, agent_idx, :))';
            metrics.resource_efficiency = episode_results.resource_efficiency(end, agent_idx);
            metrics.allocation_balance = episode_results.allocation_balance(end, agent_idx);
            monitor.update(iter, metrics);
        end
        if mod(iter, 50) == 0
            avg_radi = mean(episode_results.avg_radi);
            avg_efficiency = mean(episode_results.avg_efficiency);
            fprintf('[迭代 %d] 平均RADI: %.3f, 资源效率: %.2f%%, 探索率: %.3f\n', ...
                   iter, avg_radi, avg_efficiency * 100, current_epsilon);
        end
        if mod(iter, 200) == 0
            checkAndAdjustPerformanceRADI(defender_agents, monitor, iter, config);
        end
    end
    results = monitor.getResults();
    % 字段兼容性整理，确保RADI体系主字段
    if isfield(results, 'radi_scores')
        results.radi = results.radi_scores;
    end
    results.n_agents = n_agents;
    results.n_iterations = n_iterations;
    trained_agents.defenders = defender_agents;
    trained_agents.attacker = attacker_agent;
end

function episode_results = improvedRunEpisodesRADI(env, defender_agents, attacker_agent, n_episodes, config)
    n_agents = length(defender_agents);
    episode_results.radi_scores = zeros(n_episodes, n_agents);
    episode_results.resource_efficiency = zeros(n_episodes, n_agents);
    episode_results.allocation_balance = zeros(n_episodes, n_agents);
    episode_results.defender_rewards = zeros(n_episodes, n_agents);
    episode_results.attacker_rewards = zeros(n_episodes, 1);
    episode_results.resource_allocations = zeros(n_episodes, n_agents, 5);
    episode_results.attack_info = cell(n_episodes, 1);
    for ep = 1:n_episodes
        state = env.reset();
        is_attack_episode = rand() < 0.7;
        % 生成攻击者动作向量
        if is_attack_episode
            attacker_action_vec = ones(1, env.n_stations);
            for s = 1:env.n_stations
                attacker_action_vec(s) = randi([2, env.n_attack_types]);
            end
        else
            attacker_action_vec = ones(1, env.n_stations); % 默认安全动作
        end
        for agent_idx = 1:n_agents
            defender = defender_agents{agent_idx};
            % --- 提取每站重要性特征作为state_vec ---
            n_stations = env.n_stations;
            station_features = state(1:n_stations*8);
            state_vec = zeros(1, n_stations);
            for s = 1:n_stations
                state_vec(s) = station_features((s-1)*8 + 1);
            end
            defender_action = defender.selectAction(state_vec);
            % --- Robust shape check for defender_action ---
            if isempty(defender_action) || numel(defender_action) ~= 5
                warning('main_improved_fsp_simulation: defender_action is empty or not length 5, auto-fixing...');
                defender_action = ones(1, 5);
            end
            defender_action = reshape(defender_action, 1, 5);
            % --- Robust shape check for attacker_action_vec ---
            if isempty(attacker_action_vec) || numel(attacker_action_vec) ~= 5
                warning('main_improved_fsp_simulation: attacker_action_vec is empty or not length 5, auto-fixing...');
                attacker_action_vec = ones(1, 5);
            end
            attacker_action_vec = reshape(attacker_action_vec, 1, 5);
            [next_state, reward_def, reward_att, info] = env.step(defender_action, attacker_action_vec);
            % --- 提取next_state_vec ---
            next_station_features = next_state(1:n_stations*8);
            next_state_vec = zeros(1, n_stations);
            for s = 1:n_stations
                next_state_vec(s) = next_station_features((s-1)*8 + 1);
            end
            if isfield(info, 'resource_allocation')
                if size(info.resource_allocation, 1) > 1
                    current_allocation = info.resource_allocation(agent_idx, :);
                else
                    current_allocation = info.resource_allocation;
                end
            else
                current_allocation = ones(1, 5) * 0.2;
            end
            assert(isvector(current_allocation) && numel(current_allocation) == 5, ...
                'current_allocation must be a 1x5 vector, got size %s', mat2str(size(current_allocation)));
            radi = calculateRADI(current_allocation, config.radi.optimal_allocation, config.radi);
            efficiency = calculateResourceEfficiency(current_allocation, info);
            balance = calculateAllocationBalance(current_allocation);
            radi_reward = calculateRADIReward(radi, efficiency, balance, config);
            episode_results.radi_scores(ep, agent_idx) = radi;
            episode_results.resource_efficiency(ep, agent_idx) = efficiency;
            episode_results.allocation_balance(ep, agent_idx) = balance;
            episode_results.defender_rewards(ep, agent_idx) = radi_reward;
            episode_results.resource_allocations(ep, agent_idx, :) = current_allocation;
            if isa(defender, 'SARSAAgent')
                next_action = defender.selectAction(next_state_vec);
                defender.update(state_vec, defender_action, radi_reward, next_state_vec, next_action);
            else
                defender.update(state_vec, defender_action, radi_reward, next_state_vec, []);
            end
            if agent_idx == 1
                episode_results.attack_info{ep} = info;
            end
        end
        % --- Robust shape check for attacker_action_vec before update ---
        if isempty(attacker_action_vec) || numel(attacker_action_vec) ~= 5
            warning('main_improved_fsp_simulation: attacker_action_vec is empty or not length 5 (update), auto-fixing...');
            attacker_action_vec = ones(1, 5);
        end
        attacker_action_vec = reshape(attacker_action_vec, 1, 5);
        % --- 提取attacker_state_vec/next_state_vec ---
        attacker_state_vec = zeros(1, n_stations);
        for s = 1:n_stations
            attacker_state_vec(s) = station_features((s-1)*8 + 1);
        end
        next_attacker_state_vec = zeros(1, n_stations);
        for s = 1:n_stations
            next_attacker_state_vec(s) = next_station_features((s-1)*8 + 1);
        end
        avg_radi = mean(episode_results.radi_scores(ep, :));
        attacker_reward = avg_radi * 10;
        attacker_agent.update(attacker_state_vec, attacker_action_vec, attacker_reward, next_attacker_state_vec, []);
        episode_results.attacker_rewards(ep) = attacker_reward;
    end
    episode_results.avg_radi = mean(episode_results.radi_scores, 1);
    episode_results.avg_efficiency = mean(episode_results.resource_efficiency, 1);
    episode_results.avg_balance = mean(episode_results.allocation_balance, 1);
    episode_results.avg_defender_reward = mean(episode_results.defender_rewards, 1);
    episode_results.avg_attacker_reward = mean(episode_results.attacker_rewards);
    if n_episodes >= 10
        fprintf('\nEpisode批次完成:\n');
        for i = 1:n_agents
            fprintf('  智能体%d - RADI: %.3f, 效率: %.2f%%, 平衡度: %.2f%%\n', ...
                    i, episode_results.avg_radi(i), ...
                    episode_results.avg_efficiency(i) * 100, ...
                    episode_results.avg_balance(i) * 100);
        end
    end
end

function checkAndAdjustPerformanceRADI(defender_agents, monitor, iter, config)
    results = monitor.getResults();
    last_iters = max(1, iter-49):iter;
    if isfield(results, 'radi_scores')
        avg_radi = mean(results.radi_scores(last_iters));
        if avg_radi > 0.4
            for i = 1:length(defender_agents)
                defender_agents{i}.epsilon = min(0.5, defender_agents{i}.epsilon * 1.5);
                defender_agents{i}.learning_rate = min(0.3, defender_agents{i}.learning_rate * 1.2);
                fprintf('  [警告] %s RADI过高(%.3f)，调整参数\n', defender_agents{i}.name, avg_radi);
            end
        elseif avg_radi < 0.1
            for i = 1:length(defender_agents)
                defender_agents{i}.epsilon = max(0.05, defender_agents{i}.epsilon * 0.9);
            end
        end
    end
end

function displayFinalPerformance(results)
    fprintf('\n========================================\n');
    fprintf('最终性能总结（基于RADI）\n');
    fprintf('========================================\n');
    last_iters = max(1, results.n_iterations-99):results.n_iterations;
    agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
    for i = 1:results.n_agents
        avg_radi = mean(results.radi(i, last_iters));
        avg_efficiency = mean(results.resource_efficiency(i, last_iters));
        avg_balance = mean(results.allocation_balance(i, last_iters));
        fprintf('\n%s:\n', agent_names{i});
        fprintf('  - 平均RADI: %.3f\n', avg_radi);
        fprintf('  - 资源效率: %.2f%%\n', avg_efficiency * 100);
        fprintf('  - 分配平衡度: %.2f%%\n', avg_balance * 100);
        if avg_radi <= 0.1
            fprintf('  - 评估: 优秀 ✓\n');
        elseif avg_radi <= 0.2
            fprintf('  - 评估: 良好\n');
        elseif avg_radi <= 0.3
            fprintf('  - 评估: 一般\n');
        else
            fprintf('  - 评估: 需要改进\n');
        end
        if isfield(results, 'final_allocations') && size(results.final_allocations, 1) >= i
            fprintf('  - 最终资源分配: [%.2f, %.2f, %.2f, %.2f, %.2f]\n', ...
                   results.final_allocations(i, :));
        end
    end
    fprintf('\n========================================\n');
end

%% RADI相关辅助函数
function radi = calculateRADI(current_allocation, optimal_allocation, radi_config)
    current_allocation = reshape(current_allocation, 1, []); % 保证为行向量
    optimal_allocation = reshape(optimal_allocation, 1, []);
    weights = [
        radi_config.weight_computation,
        radi_config.weight_bandwidth,
        radi_config.weight_sensors,
        radi_config.weight_scanning,
        radi_config.weight_inspection
    ];
    deviation = abs(current_allocation - optimal_allocation);
    deviation = deviation(:)'; % 强制为1x5行向量
    weights = weights(:)';     % 强制为1x5行向量
    radi = sum(weights .* deviation, 2); % 明确sum为标量
    radi = radi(1); % 保证输出为标量
end
function efficiency = calculateResourceEfficiency(allocation, info)
    % 确保allocation是行向量
    allocation = reshape(allocation, 1, []);
    
    if isfield(info, 'game_result') && isfield(info.game_result, 'resource_efficiency')
        efficiency = info.game_result.resource_efficiency;
    else
        % 计算实际资源利用率
        total_allocated = sum(allocation);
        max_possible = length(allocation); % 假设每个资源最大值为1
        
        if max_possible > 0
            efficiency = total_allocated / max_possible;
        else
            efficiency = 0;
        end
        
        % 确保效率在0-1之间
        efficiency = max(0, min(1, efficiency));
    end
end
function balance = calculateAllocationBalance(allocation)
    if sum(allocation) == 0
        balance = 0;
        return;
    end
    cv = std(allocation) / (mean(allocation) + eps);
    balance = 1 / (1 + cv);
end
function reward = calculateRADIReward(radi, efficiency, balance, config)
    radi_penalty = -radi * 50;
    efficiency_bonus = efficiency * 30;
    balance_bonus = balance * 20;
    reward = config.reward.w_radi * radi_penalty + ...
             config.reward.w_efficiency * efficiency_bonus + ...
             config.reward.w_balance * balance_bonus;
    if radi <= config.radi.threshold_excellent
        reward = reward + 50;
    elseif radi <= config.radi.threshold_good
        reward = reward + 20;
    end
end