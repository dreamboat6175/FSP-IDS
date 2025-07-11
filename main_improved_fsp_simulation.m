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
    
    % --- 您可以在此调整关键参数 ---
    config.learning_rate = 0.15;  % 适中的学习率
    config.epsilon = 0.8;        % 高初始探索率
    config.n_iterations = 1000;  % 大幅增加迭代次数以保证收敛
    config.n_episodes_per_iter = 50;  % 减少每轮episode数以加快训练
    
    % 2. 初始化日志系统
    logger = Logger(config.log_file);
    logger.info('改进版FSP仿真开始');
    
    % 3. 参数验证
    ConfigManager.validateConfig(config);
    
    % 4. 初始化改进的仿真环境
    fprintf('正在初始化改进的仿真环境...\n');
    % ===== 修改开始 =====
    % 将调用的环境从 TCSEnvironment 修正为您最新的 CyberBattleTCSEnvironment
    env = CyberBattleTCSEnvironment(config); 
    % ===== 修改结束 =====
    
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
    monitor = PerformanceMonitor(config.n_iterations, length(defender_agents));
    
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
    % 改进的FSP训练循环
    
    n_agents = length(defender_agents);
    n_iterations = config.n_iterations;
    n_episodes = config.n_episodes_per_iter;
    
    % 预热阶段（前200次迭代使用更高探索率）
    warmup_iterations = 200;
    original_epsilon = config.epsilon;
    
    for iter = 1:n_iterations
        % 调整探索率
        if iter <= warmup_iterations
            current_epsilon = 0.8;  % 预热阶段高探索
        else
            current_epsilon = max(config.epsilon_min, ...
                                original_epsilon * (config.epsilon_decay ^ (iter - warmup_iterations)));
        end
        
        % 更新所有智能体的探索率
        for i = 1:n_agents
            defender_agents{i}.epsilon = current_epsilon;
        end
        attacker_agent.epsilon = current_epsilon * 0.5;  % 攻击者探索率较低
        
        % 运行episodes
        episode_results = improvedRunEpisodes(env, defender_agents, attacker_agent, n_episodes);
        
        % 更新监控
        monitor.update(iter, episode_results, defender_agents, attacker_agent, env);
        
        % 显示进度
        if mod(iter, 50) == 0
            avg_detection = mean(episode_results.avg_detection_rate);
            fprintf('[迭代 %d] 平均检测率: %.2f%%, 探索率: %.3f\n', ...
                   iter, avg_detection * 100, current_epsilon);
        end
        
        % 中期性能检查
        if mod(iter, 200) == 0
            checkAndAdjustPerformance(defender_agents, monitor, iter);
        end
    end
    
    % 返回结果
    results = monitor.getResults();
    trained_agents.defenders = defender_agents;
    trained_agents.attacker = attacker_agent;
end

function episode_results = improvedRunEpisodes(env, defender_agents, attacker_agent, n_episodes)
    % 改进的episodes运行函数，确保正确跟踪检测指标
    
    n_agents = length(defender_agents);
    
    % 初始化结果矩阵
    episode_results.detections = zeros(n_episodes, n_agents);
    episode_results.defender_rewards = zeros(n_episodes, n_agents);
    episode_results.attacker_rewards = zeros(n_episodes, 1);
    episode_results.false_positives = zeros(n_episodes, n_agents);
    episode_results.attack_info = cell(n_episodes, 1);
    
    % 初始化每个智能体的统计
    agent_stats = struct();
    for i = 1:n_agents
        agent_stats(i).tp = 0;  % True Positives
        agent_stats(i).tn = 0;  % True Negatives
        agent_stats(i).fp = 0;  % False Positives
        agent_stats(i).fn = 0;  % False Negatives
        agent_stats(i).total_detections = 0;
        agent_stats(i).total_attacks = 0;
        agent_stats(i).total_no_attacks = 0;
    end
    
    % 运行episodes
    for ep = 1:n_episodes
        % 重置环境
        state = env.reset();
        
        % 重置环境的连续正确计数
        env.consecutive_correct = 0;
        
        % 决定这个episode是否包含攻击（70%概率有攻击）
        is_attack_episode = rand() < 0.7;
        
        % 为每个防御智能体运行
        for agent_idx = 1:n_agents
            defender = defender_agents{agent_idx};
            
            % 防御者选择动作
            defender_action = defender.selectAction(state);
            
            % 攻击者策略
            if is_attack_episode
                % 智能攻击：优先攻击重要组件
                important_components = find(env.component_importance > 0.6);
                if ~isempty(important_components)
                    target_component = important_components(randi(length(important_components)));
                    attack_type = randi([2, min(env.n_attack_types, 5)]); % 随机选择攻击类型
                    
                    % 计算攻击者动作
                    attacker_action = 1 + (attack_type - 2) * env.total_components + target_component;
                else
                    % 随机攻击
                    target_component = randi(env.total_components);
                    attack_type = randi([2, min(env.n_attack_types, 5)]);
                    attacker_action = 1 + (attack_type - 2) * env.total_components + target_component;
                end
            else
                % 无攻击
                attacker_action = 1;
            end
            
            % 确保动作在有效范围内
            defender_action = max(1, min(defender_action, env.action_dim_defender));
            attacker_action = max(1, min(attacker_action, env.action_dim_attacker));
            
            % 环境交互
            [next_state, reward_def, reward_att, info] = env.step(defender_action, attacker_action);
            
            % 记录检测结果
            episode_results.detections(ep, agent_idx) = info.detected;
            episode_results.defender_rewards(ep, agent_idx) = reward_def;
            
            % 更新统计
            switch info.detection_category
                case 'TP'
                    agent_stats(agent_idx).tp = agent_stats(agent_idx).tp + 1;
                    agent_stats(agent_idx).total_detections = agent_stats(agent_idx).total_detections + 1;
                case 'TN'
                    agent_stats(agent_idx).tn = agent_stats(agent_idx).tn + 1;
                case 'FP'
                    agent_stats(agent_idx).fp = agent_stats(agent_idx).fp + 1;
                    episode_results.false_positives(ep, agent_idx) = 1;
                case 'FN'
                    agent_stats(agent_idx).fn = agent_stats(agent_idx).fn + 1;
            end
            
            % 记录攻击信息
            if info.is_attack
                agent_stats(agent_idx).total_attacks = agent_stats(agent_idx).total_attacks + 1;
            else
                agent_stats(agent_idx).total_no_attacks = agent_stats(agent_idx).total_no_attacks + 1;
            end
            
            % 更新智能体
            if isa(defender, 'SARSAAgent')
                % SARSA需要下一个动作
                next_action = defender.selectAction(next_state);
                defender.update(state, defender_action, reward_def, next_state, next_action);
            else
                % Q-Learning和Double Q-Learning
                defender.update(state, defender_action, reward_def, next_state, []);
            end
            
            % 记录攻击信息（仅在第一个智能体时）
            if agent_idx == 1
                episode_results.attack_info{ep} = info;
            end
        end
        
        % 更新攻击者（使用平均防御奖励的负值）
        avg_defender_reward = mean(episode_results.defender_rewards(ep, :));
        attacker_reward = -avg_defender_reward * 0.8;
        attacker_agent.update(state, attacker_action, attacker_reward, next_state, []);
        episode_results.attacker_rewards(ep) = attacker_reward;
        
        % 每10个episode进行经验回放
        if mod(ep, 10) == 0
            for agent_idx = 1:n_agents
                % 这里可以添加经验回放逻辑
            end
        end
    end
    
    % 计算平均统计
    for i = 1:n_agents
        % 检测率（TPR）
        if agent_stats(i).total_attacks > 0
            episode_results.avg_detection_rate(i) = agent_stats(i).tp / agent_stats(i).total_attacks;
        else
            episode_results.avg_detection_rate(i) = 0;
        end
        
        % 误报率（FPR）
        if agent_stats(i).total_no_attacks > 0
            episode_results.avg_false_positive_rate(i) = agent_stats(i).fp / agent_stats(i).total_no_attacks;
        else
            episode_results.avg_false_positive_rate(i) = 0;
        end
    end
    
    % 添加统计信息到结果
    episode_results.tp_count = sum([agent_stats.tp]);
    episode_results.tn_count = sum([agent_stats.tn]);
    episode_results.fp_count = sum([agent_stats.fp]);
    episode_results.fn_count = sum([agent_stats.fn]);
    
    % 平均奖励
    episode_results.avg_defender_reward = mean(episode_results.defender_rewards, 1);
    episode_results.avg_attacker_reward = mean(episode_results.attacker_rewards);
    
    % 显示episode摘要
    if n_episodes >= 10
        fprintf('\nEpisode批次完成:\n');
        for i = 1:n_agents
            fprintf('  智能体%d - TPR: %.2f%%, FPR: %.2f%%\n', ...
                    i, episode_results.avg_detection_rate(i) * 100, ...
                    episode_results.avg_false_positive_rate(i) * 100);
        end
    end
end

function checkAndAdjustPerformance(defender_agents, monitor, iter)
    % 检查并调整性能
    
    results = monitor.getResults();
    last_iters = max(1, iter-49):iter;
    
    for i = 1:length(defender_agents)
        avg_detection = mean(results.detection_rates(i, last_iters));
        
        if avg_detection < 0.2
            % 性能太差，增加探索
            defender_agents{i}.epsilon = min(0.5, defender_agents{i}.epsilon * 1.5);
            defender_agents{i}.learning_rate = min(0.3, defender_agents{i}.learning_rate * 1.2);
            fprintf('  [警告] %s 检测率过低(%.1f%%)，调整参数\n', ...
                   defender_agents{i}.name, avg_detection * 100);
        end
    end
end

function displayFinalPerformance(results)
    % 显示最终性能总结
    
    fprintf('\n========================================\n');
    fprintf('最终性能总结\n');
    fprintf('========================================\n');
    
    last_iters = max(1, results.n_iterations-99):results.n_iterations;
    
    agent_names = {'Q-Learning', 'SARSA', 'Double Q-Learning'};
    
    for i = 1:results.n_agents
        avg_detection = mean(results.detection_rates(i, last_iters));
        avg_resource = mean(results.resource_utilization(i, last_iters));
        avg_convergence = mean(results.convergence_metrics(i, last_iters));
        
        fprintf('\n%s:\n', agent_names{i});
        fprintf('  - 平均检测率: %.2f%%\n', avg_detection * 100);
        fprintf('  - 资源利用率: %.2f%%\n', avg_resource * 100);
        fprintf('  - 收敛性指标: %.4f\n', avg_convergence);
        
        % 性能评估
        if avg_detection > 0.7
            fprintf('  - 评估: 优秀 ✓\n');
        elseif avg_detection > 0.5
            fprintf('  - 评估: 良好\n');
        elseif avg_detection > 0.3
            fprintf('  - 评估: 一般\n');
        else
            fprintf('  - 评估: 需要改进\n');
        end
    end
    
    fprintf('\n========================================\n');
end