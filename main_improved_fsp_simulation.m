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
                % SARSA智能体已在构造函数中进行了优化的Q表初始化
                
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
            
            % 智能体选择动作（每个站点选择一种资源类型）
            if isa(defender, 'QLearningAgent') || isa(defender, 'SARSAAgent') || isa(defender, 'DoubleQLearningAgent')
                % 使用智能体的selectAction方法直接选择动作
                defender_action = defender.selectAction(state);
            else
                % 其他类型智能体的默认行为
                defender_action = ones(1, n_stations);
            end
            
            % 执行动作
            [next_state, reward_def, ~, info] = env.step(defender_action, attacker_action_vec);
            
            % 获取资源分配（已经归一化）
            if isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
                current_allocation = info.resource_allocation;
            else
                % 备用方案：手动计算资源分配
                current_allocation = env.calculateResourceAllocationFromActions(defender_action);
            end
            
            % 验证资源分配
            agent_name = sprintf('Agent-%d', agent_idx);
            validateResourceAllocation(current_allocation, agent_name, ep);
            
            % 计算指标
            radi = calculateRADI(current_allocation, config.radi.optimal_allocation, config.radi);
            efficiency = calculateResourceEfficiency(current_allocation, info);
            balance = calculateAllocationBalance(current_allocation);
            radi_reward = calculateRADIReward(radi, efficiency, balance, config);
            
            % 记录结果
            episode_results.radi_scores(ep, agent_idx) = radi;
            episode_results.resource_efficiency(ep, agent_idx) = efficiency;
            episode_results.allocation_balance(ep, agent_idx) = balance;
            episode_results.defender_rewards(ep, agent_idx) = radi_reward;
            episode_results.resource_allocations(ep, agent_idx, :) = current_allocation;
            
            % 更新智能体 - 传递状态向量而不是状态索引
            defender.update(state, defender_action, radi_reward, next_state, []);
            
            if agent_idx == 1
                episode_results.attack_info{ep} = info;
            end
        end
        
        % 更新攻击者 - 传递状态向量而不是状态索引
        avg_radi = mean(episode_results.radi_scores(ep, :));
        attacker_reward = avg_radi * 10;
        attacker_agent.update(state, attacker_action_vec, attacker_reward, next_state, []);
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
            fprintf('  智能体%d - RADI: %.3f, 效率: %.2f%%, 平衡度: %.2f%%\n', ...
                    i, episode_results.avg_radi(i), ...
                    episode_results.avg_efficiency(i) * 100, ...
                    episode_results.avg_balance(i) * 100);
            
            % 显示平均分配
            avg_allocation = squeeze(mean(episode_results.resource_allocations(:, i, :), 1));
            fprintf('    平均分配: [%.3f, %.3f, %.3f, %.3f, %.3f] (总和=%.3f)\n', ...
                    avg_allocation, sum(avg_allocation));
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
    
    % 去除零值计算变异系数
    non_zero_alloc = allocation(allocation > 0);
    if isempty(non_zero_alloc)
        balance = 0;
    else
        % 使用变异系数的倒数作为平衡度
        cv = std(non_zero_alloc) / mean(non_zero_alloc);
        balance = 1 / (1 + cv);
    end
    
    % 确保在[0,1]范围内
    balance = max(0, min(1, balance));
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