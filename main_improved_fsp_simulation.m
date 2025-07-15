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
    
    if i == 1
        % 智能体1：激进探索策略（修复过度集中问题）
        % 使用接近0的初始值，强制探索
        defender_agents{i}.Q_table = randn(env.state_dim, env.action_dim_defender) * 0.01;
        
        % 极高探索率
        defender_agents{i}.epsilon = 0.95;
        defender_agents{i}.epsilon_min = 0.4;
        defender_agents{i}.epsilon_decay = 0.999;  % 非常慢的衰减
        
        % 使用高温度softmax
        defender_agents{i}.use_softmax = true;
        defender_agents{i}.temperature = 10.0;
        defender_agents{i}.temperature_min = 2.0;
        defender_agents{i}.temperature_decay = 0.999;
        
        % 高学习率
        defender_agents{i}.learning_rate = 0.5;
        defender_agents{i}.learning_rate_min = 0.1;
        defender_agents{i}.learning_rate_decay = 0.9999;
        
        fprintf('  智能体1: 激进探索 - epsilon=%.2f, temp=%.1f, lr=%.2f\n', ...
               defender_agents{i}.epsilon, defender_agents{i}.temperature, ...
               defender_agents{i}.learning_rate);
        
    elseif i == 2
        % 智能体2：平衡策略（当前表现尚可，微调）
        defender_agents{i}.Q_table = ones(env.state_dim, env.action_dim_defender) * 0.5 + ...
                                    randn(env.state_dim, env.action_dim_defender) * 0.2;
        
        defender_agents{i}.epsilon = 0.5;
        defender_agents{i}.epsilon_min = 0.15;
        defender_agents{i}.epsilon_decay = 0.997;
        
        defender_agents{i}.use_softmax = false;  % 使用ε-greedy
        defender_agents{i}.learning_rate = 0.2;
        defender_agents{i}.learning_rate_min = 0.05;
        defender_agents{i}.learning_rate_decay = 0.9998;
        
        fprintf('  智能体2: 平衡策略 - epsilon=%.2f, softmax=%d, lr=%.2f\n', ...
               defender_agents{i}.epsilon, defender_agents{i}.use_softmax, ...
               defender_agents{i}.learning_rate);
        
    else
        % 智能体3：需要增加探索（当前完全没有探索）
        defender_agents{i}.Q_table = ones(env.state_dim, env.action_dim_defender) * 1.0 + ...
                                    randn(env.state_dim, env.action_dim_defender) * 0.3;
        
        % 增加探索率，避免过早收敛
        defender_agents{i}.epsilon = 0.3;
        defender_agents{i}.epsilon_min = 0.05;
        defender_agents{i}.epsilon_decay = 0.995;
        
        % 使用温和的softmax
        defender_agents{i}.use_softmax = true;
        defender_agents{i}.temperature = 2.0;
        defender_agents{i}.temperature_min = 0.5;
        defender_agents{i}.temperature_decay = 0.998;
        
        defender_agents{i}.learning_rate = 0.15;
        defender_agents{i}.learning_rate_min = 0.03;
        defender_agents{i}.learning_rate_decay = 0.9997;
        
        fprintf('  智能体3: 改进探索 - epsilon=%.2f, temp=%.1f, lr=%.2f\n', ...
               defender_agents{i}.epsilon, defender_agents{i}.temperature, ...
               defender_agents{i}.learning_rate);
    end
    
    % 为所有智能体添加必要属性
    if ~isprop(defender_agents{i}, 'epsilon_min')
        addprop(defender_agents{i}, 'epsilon_min');
    end
    if ~isprop(defender_agents{i}, 'epsilon_decay')
        addprop(defender_agents{i}, 'epsilon_decay');
    end
    if ~isprop(defender_agents{i}, 'temperature_min')
        addprop(defender_agents{i}, 'temperature_min');
        defender_agents{i}.temperature_min = 0.1;
    end
    if ~isprop(defender_agents{i}, 'temperature_decay')
        addprop(defender_agents{i}, 'temperature_decay');
        defender_agents{i}.temperature_decay = 0.995;
    end
    if ~isprop(defender_agents{i}, 'learning_rate_min')
        addprop(defender_agents{i}, 'learning_rate_min');
    end
    if ~isprop(defender_agents{i}, 'learning_rate_decay')
        addprop(defender_agents{i}, 'learning_rate_decay');
    end


            case 'SARSA'
                defender_agents{i} = SARSAAgent(sprintf('SARSA-%d', i), ...
                                             'defender', config, ...
                                             env.state_dim, env.action_dim_defender);
                
                % SARSA特有的初始化
                base_value = 1.5 + (i-1) * 0.4;
                noise_level = 0.4;
                defender_agents{i}.Q_table = ones(env.state_dim, env.action_dim_defender) * base_value + ...
                                            randn(env.state_dim, env.action_dim_defender) * noise_level;
                
                % SARSA参数设置
                defender_agents{i}.epsilon = 0.3 + (i-1) * 0.15;
                defender_agents{i}.use_softmax = mod(i, 3) == 1;  % 部分使用softmax
                defender_agents{i}.temperature = 0.8 + (i-1) * 0.3;
                
                fprintf('  SARSA智能体%d: epsilon=%.2f, softmax=%d, temp=%.2f\n', ...
                       i, defender_agents{i}.epsilon, defender_agents{i}.use_softmax, ...
                       defender_agents{i}.temperature);
                
            case 'Double Q-Learning'
                defender_agents{i} = DoubleQLearningAgent(sprintf('DoubleQ-%d', i), ...
                                                       'defender', config, ...
                                                       env.state_dim, env.action_dim_defender);
                
                % Double Q-Learning初始化
                base_value1 = 1.8 + (i-1) * 0.3;
                base_value2 = 2.2 + (i-1) * 0.2;
                noise_level = 0.35;
                
                defender_agents{i}.Q1_table = ones(env.state_dim, env.action_dim_defender) * base_value1 + ...
                                             randn(env.state_dim, env.action_dim_defender) * noise_level;
                defender_agents{i}.Q2_table = ones(env.state_dim, env.action_dim_defender) * base_value2 + ...
                                             randn(env.state_dim, env.action_dim_defender) * noise_level;
                
                % Double Q参数设置
                defender_agents{i}.epsilon = 0.35 + (i-1) * 0.12;
                defender_agents{i}.use_softmax = i <= 2;  % 前两个使用softmax
                defender_agents{i}.temperature = 1.1 + (i-1) * 0.25;
                
                fprintf('  DoubleQ智能体%d: epsilon=%.2f, softmax=%d, temp=%.2f\n', ...
                       i, defender_agents{i}.epsilon, defender_agents{i}.use_softmax, ...
                       defender_agents{i}.temperature);
        end
        
        % === 为所有智能体设置共同属性 ===
        if isprop(defender_agents{i}, 'learning_rate')
            defender_agents{i}.learning_rate = 0.1 + (i-1) * 0.02;  % 不同学习率
        end
        
        % 确保每个智能体都有必要的属性
        if ~isprop(defender_agents{i}, 'use_softmax')
            defender_agents{i}.use_softmax = false;
        end
        if ~isprop(defender_agents{i}, 'temperature')
            defender_agents{i}.temperature = 1.0;
        end
    end

    % =========================================================================
    % === 代码修正：添加攻击者智能体的初始化 ===
    % =========================================================================
    fprintf('正在初始化攻击者智能体...\n');
    % 假设攻击者也使用Q-Learning。攻击者的状态空间与防御者相同，
    % 但动作空间（action_dim_attacker）由环境定义。
    if ~isprop(env, 'action_dim_attacker')
        % 如果环境中没有定义 `action_dim_attacker`，则根据逻辑推断。
        % 在 `improvedRunEpisodesRADI` 函数中，攻击动作是 `randi([2, env.n_attack_types])`
        % 这意味着动作空间维度就是攻击类型的数量。
        env.action_dim_attacker = env.n_attack_types;
        fprintf('  [警告] 环境中未定义 action_dim_attacker, 已根据逻辑推断为 %d\n', env.action_dim_attacker);
    end
    
    attacker_agent = QLearningAgent('Attacker', 'attacker', config, ...
                                  env.state_dim, env.action_dim_attacker);
    
    % 为攻击者设置独立的学习参数
    attacker_agent.learning_rate = config.learning_rate * 0.9; % 攻击者学习率可以略有不同
    attacker_agent.epsilon = config.epsilon;                   % 攻击者使用基础探索率
    fprintf('  攻击者智能体初始化完成。\n');
    % =========================================================================
    
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
            % 攻防动态输出
            fprintf('\n[攻防动态] 迭代 %d\n', iter);
            if isprop(env, 'attacker_strategy') && isprop(env, 'optimal_defender_strategy')
                fprintf('当前攻击策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', env.attacker_strategy);
                fprintf('最优防守策略: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', env.optimal_defender_strategy);
            end
            results = monitor.getResults();
            if isfield(results, 'resource_allocations')
                fields = {'computation', 'bandwidth', 'sensors', 'scanning_freq', 'inspection_depth'};
                for i = 1:min(3, n_agents)
                    recent_alloc = zeros(1, 5);
                    for j = 1:5
                        field_data = results.resource_allocations.(fields{j});
                        n = size(field_data, 1);
                        recent_range = max(1, n-49):n;
                        % 兼容一维和二维
                        if isvector(field_data)
                            agent_data = field_data(recent_range);
                        elseif size(field_data, 2) >= i
                            agent_data = field_data(recent_range, i);
                        else
                            agent_data = field_data(recent_range, 1);
                        end
                        recent_alloc(j) = mean(agent_data);
                    end
                    fprintf('智能体%d实际分配: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', i, recent_alloc);
                end
            end
        end
        if mod(iter, 200) == 0
            checkAndAdjustPerformanceRADI(defender_agents, monitor, iter, config);
        end
    end
    results = monitor.getResults();
    % 字段兼容性整理，确保RADI体系主字段
    if isfield(results, 'radi_scores')
        % 保证为[n_agents, n_iterations]结构
        if size(results.radi_scores, 1) == n_iterations && size(results.radi_scores, 2) == n_agents
            results.radi = results.radi_scores';
        elseif size(results.radi_scores, 2) == n_iterations && size(results.radi_scores, 1) == n_agents
            results.radi = results.radi_scores;
        elseif size(results.radi_scores, 1) == n_agents && size(results.radi_scores, 2) ~= n_iterations
            % 补齐或截断
            pad = n_iterations - size(results.radi_scores, 2);
            if pad > 0
                results.radi = [results.radi_scores, nan(n_agents, pad)];
            else
                results.radi = results.radi_scores(:, 1:n_iterations);
            end
        else
            % 其他情况，强制转置
            results.radi = results.radi_scores';
        end
    end
    if isfield(results, 'resource_efficiency')
        if size(results.resource_efficiency, 1) == n_iterations && size(results.resource_efficiency, 2) == n_agents
            results.resource_efficiency = results.resource_efficiency';
        elseif size(results.resource_efficiency, 2) == n_iterations && size(results.resource_efficiency, 1) == n_agents
            % ok
        elseif size(results.resource_efficiency, 1) == n_agents && size(results.resource_efficiency, 2) ~= n_iterations
            pad = n_iterations - size(results.resource_efficiency, 2);
            if pad > 0
                results.resource_efficiency = [results.resource_efficiency, nan(n_agents, pad)];
            else
                results.resource_efficiency = results.resource_efficiency(:, 1:n_iterations);
            end
        else
            results.resource_efficiency = results.resource_efficiency';
        end
    end
    if isfield(results, 'allocation_balance')
        if size(results.allocation_balance, 1) == n_iterations && size(results.allocation_balance, 2) == n_agents
            results.allocation_balance = results.allocation_balance';
        elseif size(results.allocation_balance, 2) == n_iterations && size(results.allocation_balance, 1) == n_agents
            % ok
        elseif size(results.allocation_balance, 1) == n_agents && size(results.allocation_balance, 2) ~= n_iterations
            pad = n_iterations - size(results.allocation_balance, 2);
            if pad > 0
                results.allocation_balance = [results.allocation_balance, nan(n_agents, pad)];
            else
                results.allocation_balance = results.allocation_balance(:, 1:n_iterations);
            end
        else
            results.allocation_balance = results.allocation_balance';
        end
    end
    results.n_agents = n_agents;
    results.n_iterations = n_iterations;
    trained_agents.defenders = defender_agents;
    trained_agents.attacker = attacker_agent;
end
function episode_results = improvedRunEpisodesRADI(env, defender_agents, attacker_agent, n_episodes, config)
    n_agents = length(defender_agents);
    n_stations = env.n_stations;
    n_resource_types = env.n_resource_types;
    
    % 初始化结果存储
    episode_results.radi_scores = zeros(n_episodes, n_agents);
    episode_results.resource_efficiency = zeros(n_episodes, n_agents);
    episode_results.allocation_balance = zeros(n_episodes, n_agents);
    episode_results.defender_rewards = zeros(n_episodes, n_agents);
    episode_results.attacker_rewards = zeros(n_episodes, 1);
    episode_results.resource_allocations = zeros(n_episodes, n_agents, n_resource_types);
    episode_results.attack_info = cell(n_episodes, 1);
    
    for ep = 1:n_episodes
        state = env.reset();
        is_attack_episode = rand() < 0.7;
        
        % 生成攻击者动作
        if is_attack_episode
            attacker_action_vec = ones(1, n_stations);
            for s = 1:n_stations
                attacker_action_vec(s) = randi([2, env.n_attack_types]);
            end
        else
            attacker_action_vec = ones(1, n_stations);
        end
        
        for agent_idx = 1:n_agents
            defender = defender_agents{agent_idx};
            
            % === 关键修改：直接处理智能体动作，不使用parseDefenderAction ===
            if isa(defender, 'QLearningAgent') || isa(defender, 'SARSAAgent') || isa(defender, 'DoubleQLearningAgent')
                % 强化学习智能体 - 直接获取动作
                raw_action = defender.selectAction(state);
                
                % 确保是行向量
                if iscolumn(raw_action)
                    raw_action = raw_action';
                end
                
                % 将Q表动作转换为资源分配
                allocation = convertQLearningActionToAllocation(raw_action, n_resource_types, defender);
                
            elseif isa(defender, 'FictitiousPlayAgent')
                % 虚拟博弈智能体
                allocation = defender.selectAction(state, env.attacker_strategy);
                
            else
                % 其他智能体类型 - 默认行为
                allocation = defender.selectAction(state);
            end
            
            % 验证和归一化分配
            allocation = validateAndNormalizeAllocation(allocation, n_resource_types, agent_idx, ep);
            
            % 存储分配 - 这里是关键！不使用env.parseDefenderAction
            episode_results.resource_allocations(ep, agent_idx, :) = allocation;
            
            % 计算性能指标
            optimal_allocation = config.radi.optimal_allocation;
            radi = env.calculateRADIScore(allocation, optimal_allocation);
            episode_results.radi_scores(ep, agent_idx) = radi;
            
            % 计算效率和平衡度
            efficiency = calculateResourceEfficiency(allocation, struct());
            balance = calculateAllocationBalance(allocation);
            episode_results.resource_efficiency(ep, agent_idx) = efficiency;
            episode_results.allocation_balance(ep, agent_idx) = balance;
            
            % 计算奖励
            defender_reward = calculateRADIReward(radi, efficiency, balance, config);
            episode_results.defender_rewards(ep, agent_idx) = defender_reward;
            
            % 更新智能体 - 使用原始动作
            if ismethod(defender, 'update')
                next_state = state; % 简化处理
                defender.update(state, raw_action, defender_reward, next_state, []);
            end
        end
        
        % 攻击者奖励
        avg_radi = mean(episode_results.radi_scores(ep, :));
        attacker_reward = avg_radi * 10;
        if ismethod(attacker_agent, 'update')
            attacker_agent.update(state, attacker_action_vec, attacker_reward, state, []);
        end
        episode_results.attacker_rewards(ep) = attacker_reward;
    end
    
    % 计算平均值
    episode_results.avg_radi = mean(episode_results.radi_scores, 1);
    episode_results.avg_efficiency = mean(episode_results.resource_efficiency, 1);
    episode_results.avg_balance = mean(episode_results.allocation_balance, 1);
    episode_results.avg_defender_reward = mean(episode_results.defender_rewards, 1);
    episode_results.avg_attacker_reward = mean(episode_results.attacker_rewards);
    
    % 输出结果
    if n_episodes >= 10
        fprintf('\nEpisode批次完成:\n');
        for i = 1:n_agents
            fprintf('  智能体%d - RADI: %.6f, 效率: %.2f%%, 平衡度: %.2f%%\n', ...
                    i, episode_results.avg_radi(i), ...
                    episode_results.avg_efficiency(i) * 100, ...
                    episode_results.avg_balance(i) * 100);
            
            % 显示平均分配
            avg_allocation = squeeze(mean(episode_results.resource_allocations(:, i, :), 1));
            fprintf('    平均分配: [%.3f, %.3f, %.3f, %.3f, %.3f] (总和=%.3f)\n', ...
                    avg_allocation, sum(avg_allocation));
            
            % 显示分配标准差 - 这里可以看出是否有变化
            std_allocation = squeeze(std(episode_results.resource_allocations(:, i, :), 0, 1));
            fprintf('    分配标准差: [%.3f, %.3f, %.3f, %.3f, %.3f]\n', std_allocation);
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
    % 计算资源利用效率
    if isfield(info, 'game_result') && isfield(info.game_result, 'resource_efficiency')
        % 如果有游戏结果中的效率，使用它
        efficiency = info.game_result.resource_efficiency;
    else
        % 否则基于分配计算
        % 考虑分配的均衡性和总量
        total_allocation = sum(allocation);
        
        % 效率 = 使用的资源比例 * 分配均衡度
        if total_allocation > 0
            utilization = min(1, total_allocation);
            balance = 1 - std(allocation) / (mean(allocation) + eps);
            efficiency = utilization * balance;
        else
            efficiency = 0;
        end
    end
    
    % 确保在[0,1]范围内
    efficiency = max(0, min(1, efficiency));
end
function balance = calculateAllocationBalance(allocation)
    % 计算分配平衡度
    if sum(allocation) == 0
        balance = 0;
        return;
    end
    
    % 使用基尼系数的补数作为平衡度
    n = length(allocation);
    sorted_alloc = sort(allocation);
    
    % 基尼系数
    gini = 0;
    for i = 1:n
        gini = gini + (2*i - n - 1) * sorted_alloc(i);
    end
    gini = gini / (n * sum(allocation));
    
    % 平衡度是基尼系数的补数
    balance = 1 - abs(gini);
    balance = max(0, min(1, balance));
end
function reward = calculateRADIReward(radi, efficiency, balance, config)
    % 基于RADI的奖励函数，针对不同RADI范围使用不同策略
    
    % 基础奖励
    base_reward = 0;
    
    % RADI奖励/惩罚（分段处理）
    if radi <= 0.05  % 优秀
        radi_component = 100;
    elseif radi <= 0.1  % 良好
        radi_component = 80 - (radi - 0.05) * 400;
    elseif radi <= 0.5  % 可接受
        radi_component = 60 - (radi - 0.1) * 150;
    elseif radi <= 1.0  % 较差
        radi_component = -50 - (radi - 0.5) * 100;
    else  % 很差（如智能体1的情况）
        radi_component = -100 - (radi - 1.0) * 200;
    end
    
    % 效率奖励
    if efficiency > 0.8
        efficiency_component = 50 + (efficiency - 0.8) * 250;
    elseif efficiency > 0.5
        efficiency_component = (efficiency - 0.5) * 166.67;
    else
        efficiency_component = efficiency * 100 - 50;
    end
    
    % 平衡度奖励
    balance_component = balance * 50;
    
    % 综合奖励
    reward = radi_component * config.reward.w_radi + ...
             efficiency_component * config.reward.w_efficiency + ...
             balance_component * config.reward.w_balance;
    
    % 特殊情况处理
    if radi > 1.5 && efficiency < 0.1
        % 严重问题（如智能体1），给予额外惩罚
        reward = reward - 100;
    end
    
    if radi < 0.01 && balance > 0.95
        %接近完美，给予额外奖励
        reward = reward + 100;
    end
    
    % 限制范围
    reward = max(-500, min(500, reward));
end


function validateResourceAllocation(allocation, agent_name, episode)
    % 验证资源分配的正确性
    tolerance = 1e-6;
    
    % 检查总和
    total = sum(allocation);
    if abs(total - 1) > tolerance
        error('Agent %s, Episode %d: 资源分配总和错误: %.6f (应该为1)', ...
              agent_name, episode, total);
    end
    
    % 检查非负性
    if any(allocation < -tolerance)
        error('Agent %s, Episode %d: 资源分配包含负值: %s', ...
              agent_name, episode, mat2str(allocation));
    end
    
    % 检查合理性（每个分配应该在[0,1]之间）
    if any(allocation > 1 + tolerance)
        error('Agent %s, Episode %d: 资源分配超过1: %s', ...
              agent_name, episode, mat2str(allocation));
    end
end
function allocation = convertQLearningActionToAllocation(raw_action, n_resource_types, agent)
    % 将Q-Learning动作转换为资源分配，确保多样性
    
    % 获取智能体编号（从名称中提取）
    agent_num = 0;
    if contains(agent.name, '1')
        agent_num = 1;
    elseif contains(agent.name, '2')
        agent_num = 2;
    elseif contains(agent.name, '3')
        agent_num = 3;
    end
    
    if isscalar(raw_action)
        % 单一动作索引
        action_idx = max(1, min(n_resource_types, round(raw_action)));
        
        % 根据智能体和探索率决定分配策略
        if agent_num == 1
            % 智能体1：强制多样化
            if agent.epsilon > 0.7
                % 高探索：使用更均匀的分配
                allocation = ones(1, n_resource_types) * 0.15;
                allocation(action_idx) = 0.25;
                % 随机增强其他资源
                boost_indices = randperm(n_resource_types, min(3, n_resource_types));
                for idx = boost_indices
                    allocation(idx) = allocation(idx) + rand() * 0.1;
                end
            else
                % 中等探索：避免过度集中
                allocation = ones(1, n_resource_types) * 0.1;
                allocation(action_idx) = 0.4;
                % 相邻资源也获得分配
                if action_idx > 1
                    allocation(action_idx-1) = 0.2;
                end
                if action_idx < n_resource_types
                    allocation(action_idx+1) = 0.2;
                end
            end
            
        elseif agent_num == 3
            % 智能体3：增加随机性
            if rand() < 0.3  % 30%概率使用随机策略
                allocation = rand(1, n_resource_types);
            else
                % 正常分配但添加噪声
                allocation = ones(1, n_resource_types) * 0.1;
                allocation(action_idx) = 0.5;
                noise = randn(1, n_resource_types) * 0.1;
                allocation = allocation + noise;
            end
            
        else
            % 智能体2和其他：标准策略
            allocation = ones(1, n_resource_types) * 0.05;
            allocation(action_idx) = 0.6;
            % 添加适度噪声
            if agent.epsilon > 0.1
                noise = randn(1, n_resource_types) * agent.epsilon * 0.2;
                allocation = allocation + noise;
            end
        end
        
    else
        % 向量输入
        allocation = raw_action;
        
        % 为智能体1和3添加额外噪声
        if agent_num == 1
            noise_level = 0.3;
            allocation = allocation + randn(1, n_resource_types) * noise_level;
        elseif agent_num == 3
            noise_level = 0.15;
            allocation = allocation + randn(1, n_resource_types) * noise_level;
        end
    end
    
    % 确保非负
    allocation = max(0, allocation);
    
    % 归一化
    total = sum(allocation);
    if total > 1e-10
        allocation = allocation / total;
    else
        % 使用带扰动的均匀分配
        allocation = ones(1, n_resource_types) / n_resource_types;
        allocation = allocation + randn(1, n_resource_types) * 0.05;
        allocation = max(0, allocation);
        allocation = allocation / sum(allocation);
    end
    
    % 最终检查：确保没有极端分配
    max_allocation = max(allocation);
    if max_allocation > 0.7 && agent_num == 1
        % 智能体1不允许超过70%集中在一个资源上
        allocation(allocation == max_allocation) = 0.7;
        excess = max_allocation - 0.7;
        other_indices = find(allocation ~= 0.7);
        if ~isempty(other_indices)
            allocation(other_indices) = allocation(other_indices) + excess / length(other_indices);
        end
        allocation = allocation / sum(allocation);
    end
end
function allocation = validateAndNormalizeAllocation(allocation, n_resource_types, agent_idx, episode)
    % 验证和归一化资源分配
    
    % 确保维度正确
    if length(allocation) ~= n_resource_types
        if length(allocation) < n_resource_types
            allocation = [allocation, zeros(1, n_resource_types - length(allocation))];
        else
            allocation = allocation(1:n_resource_types);
        end
    end
    
    % 确保是行向量
    if iscolumn(allocation)
        allocation = allocation';
    end
    
    % 确保非负
    allocation = max(0, allocation);
    
    % 归一化
    total = sum(allocation);
    if total > 1e-10
        allocation = allocation / total;
    else
        % 如果总和为0，使用随机分配避免均匀分配
        allocation = rand(1, n_resource_types);
        allocation = allocation / sum(allocation);
        fprintf('警告：智能体%d在Episode %d分配总和为0，使用随机分配\n', agent_idx, episode);
    end
    
    % 最终验证
    if abs(sum(allocation) - 1) > 1e-6
        error('智能体%d Episode %d: 分配归一化失败: 总和=%.6f', agent_idx, episode, sum(allocation));
    end
end