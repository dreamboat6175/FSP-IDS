好的，我理解您的要求。我们不修改整体框架，只针对您更新 CyberBattleTCSEnvironment.m 后出现的不兼容和错误进行必要的、小范围的修改。

根据对您提供的文件和日志的分析，核心问题在于 FSPSimulator.m 的主循环未能正确处理新环境返回的数据，以及 EnhancedReportGenerator.m 中方法调用方式的错误。以下是确保您的项目能顺利运行的针对性修改：

1. 修正 FSPSimulator.m
此文件是仿真的核心，需要更新它与新版 CyberBattleTCSEnvironment 的交互方式，特别是 runIteration 方法，以正确处理 step 函数返回的复杂 info 结构体，并为 SARSA 智能体提供必要的 next_action。

Matlab

%% FSPSimulator.m - Fictitious Self-Play仿真器
% =========================================================================
% 描述: 实现FSP框架的核心仿真逻辑
% =========================================================================

classdef FSPSimulator
    
    methods (Static)
        function [results, trained_agents] = run(env, defender_agents, attacker_agent, config, monitor, logger)
            % (此部分代码无变化)
            % ...
        end
        
        function episode_results = runIteration(env, defender_agents, attacker_agent, config, ~)
            % ===== 修改开始 =====
            % 修正了与新环境的交互逻辑
            
            n_agents = length(defender_agents);
            n_episodes = config.n_episodes_per_iter;
            
            % 初始化结果记录
            episode_results.detections = zeros(n_episodes, n_agents);
            episode_results.defender_rewards = zeros(n_episodes, n_agents);
            episode_results.attacker_rewards = zeros(n_episodes, 1);
            episode_results.attack_info = cell(n_episodes, 1);
            
            for ep = 1:n_episodes
                state = env.reset();
                
                % 每个防御智能体分别与攻击者交互并更新
                for agent_idx = 1:n_agents
                    defender = defender_agents{agent_idx};
                    
                    % 1. 选择动作
                    defender_action = defender.selectAction(state);
                    attacker_action = attacker_agent.selectAction(state);
                    
                    % 2. 环境交互
                    [next_state, reward, ~, info] = env.step(defender_action, attacker_action);
                    
                    % 3. 记录该智能体的结果
                    episode_results.detections(ep, agent_idx) = info.detected;
                    episode_results.defender_rewards(ep, agent_idx) = reward; % 直接使用环境返回的复合奖励
                    episode_results.attack_info{ep} = info; % 记录详细信息
                    
                    % 4. 更新智能体
                    if isa(defender, 'SARSAAgent')
                        % SARSA算法需要获取下一个状态的动作
                        next_action = defender.selectAction(next_state);
                        defender.update(state, defender_action, reward, next_state, next_action);
                    else
                        % Q-Learning 和 Double Q-Learning
                        defender.update(state, defender_action, reward, next_state, []);
                    end
                end
                
                % 5. 单独更新攻击者
                % 攻击者的奖励是防御者奖励的负值（简化处理）
                avg_attacker_reward = -mean(episode_results.defender_rewards(ep, :));
                attacker_agent.update(state, attacker_action, avg_attacker_reward, next_state, []);
                episode_results.attacker_rewards(ep) = avg_attacker_reward;
            end
            
            % 计算本轮迭代的平均统计信息
            episode_results.avg_detection_rate = mean(episode_results.detections, 1);
            episode_results.avg_defender_reward = mean(episode_results.defender_rewards, 1);
            episode_results.avg_attacker_reward = mean(episode_results.attacker_rewards);
            % ===== 修改结束 =====
        end

        
        function results = runParallelEpisodes(env, defender_agents, attacker_agent, config)
            % 并行执行episodes（简化版本，避免复杂对象传递）
            n_episodes = config.n_episodes_per_iter;
            n_agents = length(defender_agents);
            
            % 准备并行数据
            detections = zeros(n_episodes, n_agents);
            defender_rewards = zeros(n_episodes, n_agents);
            attacker_rewards = zeros(n_episodes, 1);
            
            % 提取必要的环境参数（避免传递整个对象）
            env_params = struct();
            env_params.state_dim = env.state_dim;
            env_params.action_dim_defender = env.action_dim_defender;
            env_params.action_dim_attacker = env.action_dim_attacker;
            env_params.n_stations = env.n_stations;
            env_params.n_components = env.n_components;
            env_params.total_components = env.total_components;
            env_params.component_importance = env.component_importance;
            env_params.attack_types = env.attack_types;
            env_params.attack_severity = env.attack_severity;
            env_params.attack_detection_difficulty = env.attack_detection_difficulty;
            env_params.resource_types = env.resource_types;
            env_params.resource_effectiveness = env.resource_effectiveness;
            env_params.total_resources = env.total_resources;
            
            % 提取智能体Q表（避免传递整个对象）
            defender_q_tables = cell(1, n_agents);
            for i = 1:n_agents
                if isa(defender_agents{i}, 'DoubleQLearningAgent')
                    % Double Q-Learning使用组合Q表
                    defender_q_tables{i} = (defender_agents{i}.Q1_table + defender_agents{i}.Q2_table) / 2;
                else
                    % Q-Learning和SARSA使用Q_table
                    defender_q_tables{i} = defender_agents{i}.Q_table;
                end
            end
            
            if isa(attacker_agent, 'DoubleQLearningAgent')
                attacker_q_table = (attacker_agent.Q1_table + attacker_agent.Q2_table) / 2;
            else
                attacker_q_table = attacker_agent.Q_table;
            end
            
            % 使用parfor并行计算
            parfor ep = 1:n_episodes
                % 初始化episode结果
                ep_detections = zeros(1, n_agents);
                ep_def_rewards = zeros(1, n_agents);
                ep_att_reward = 0;
                
                % 随机初始状态
                state = randi(env_params.state_dim);
                
                for agent_idx = 1:n_agents
                    % 简化的动作选择（ε-贪婪）
                    if rand() < 0.1  % 固定探索率
                        def_action = randi(env_params.action_dim_defender);
                        att_action = randi(env_params.action_dim_attacker);
                    else
                        [~, def_action] = max(defender_q_tables{agent_idx}(state, :));
                        [~, att_action] = max(attacker_q_table(state, :));
                    end
                    
                    % 简化的环境交互
                    [reward_def, reward_att, detected] = ...
                        simulateInteractionParallel(env_params, def_action, att_action);
                    
                    ep_detections(agent_idx) = detected;
                    ep_def_rewards(agent_idx) = reward_def;
                    ep_att_reward = ep_att_reward + reward_att / n_agents;
                end
                
                detections(ep, :) = ep_detections;
                defender_rewards(ep, :) = ep_def_rewards;
                attacker_rewards(ep) = ep_att_reward;
            end
            
            % 整合结果
            results.detections = detections;
            results.defender_rewards = defender_rewards;
            results.attacker_rewards = attacker_rewards;
            results.avg_detection_rate = mean(detections, 1);
            results.avg_defender_reward = mean(defender_rewards, 1);
            results.avg_attacker_reward = mean(attacker_rewards);
        end
        
        function adaptiveParameterUpdate(defender_agents, attacker_agent, config, iter)
            % 自适应更新学习参数
            
            % 更新所有防御智能体的参数
            for i = 1:length(defender_agents)
                defender_agents{i}.updateEpsilon();
                
                % 动态调整学习率
                decay_factor = 1 / (1 + 0.0001 * iter);
                defender_agents{i}.learning_rate = defender_agents{i}.learning_rate * decay_factor;
                
                % 调整温度参数
                if isprop(defender_agents{i}, 'temperature')
                    defender_agents{i}.temperature = max(0.1, defender_agents{i}.temperature * 0.995);
                end
            end
            
            % 更新攻击者参数
            attacker_agent.updateEpsilon();
            attacker_agent.learning_rate = attacker_agent.learning_rate * decay_factor;
        end
        
        function updateStrategyPools(defender_agents, attacker_agent)
            % 更新所有智能体的策略池
            
            for i = 1:length(defender_agents)
                defender_agents{i}.updateStrategyPool();
            end
            
            attacker_agent.updateStrategyPool();
        end
        
        function saveCheckpoint(defender_agents, attacker_agent, monitor, iter)
            % 保存检查点
            
            checkpoint_dir = sprintf('checkpoints/iter_%d', iter);
            if ~exist(checkpoint_dir, 'dir')
                mkdir(checkpoint_dir);
            end
            
            % 保存防御智能体
            for i = 1:length(defender_agents)
                filename = fullfile(checkpoint_dir, sprintf('defender_%d.mat', i));
                defender_agents{i}.save(filename);
            end
            
            % 保存攻击智能体
            filename = fullfile(checkpoint_dir, 'attacker.mat');
            attacker_agent.save(filename);
            
            % 保存监控数据
            monitor_data = monitor.getResults();
            save(fullfile(checkpoint_dir, 'monitor_data.mat'), 'monitor_data');
            
            fprintf('✓ 检查点已保存: %s\n', checkpoint_dir);
        end
        
        function displayProgress(iter, total_iter, start_time)
            % 显示进度信息
            
            elapsed = toc(start_time);
            avg_time_per_iter = elapsed / iter;
            eta = avg_time_per_iter * (total_iter - iter);
            
            progress_percent = iter / total_iter * 100;
            
            fprintf('\n=== 进度更新 ===\n');
            fprintf('迭代: %d / %d (%.1f%%)\n', iter, total_iter, progress_percent);
            fprintf('已用时间: %s\n', FSPSimulator.formatTime(elapsed));
            fprintf('预计剩余: %s\n', FSPSimulator.formatTime(eta));
            fprintf('平均速度: %.2f 迭代/分钟\n', 60 / avg_time_per_iter);
            fprintf('================\n\n');
        end
        
        function str = formatTime(seconds)
            % 格式化时间显示
            if seconds < 60
                str = sprintf('%.1f秒', seconds);
            elseif seconds < 3600
                str = sprintf('%.1f分钟', seconds/60);
            else
                hours = floor(seconds/3600);
                minutes = floor(mod(seconds, 3600)/60);
                str = sprintf('%d小时%d分钟', hours, minutes);
            end
        end
    end
end

%% 辅助函数
function action = selectActionParallel(agent, state)
    % 并行环境中的动作选择（简化版本）
    % 注意：在parfor中不能直接访问对象方法
    
    % 这里使用简化的ε-贪婪策略
    if rand() < 0.1  % 固定探索率
        action = randi(agent.action_dim);
    else
        % 简化的贪婪选择
        q_values = agent.Q_table(state, :);
        [~, action] = max(q_values);
    end
end

function [reward_def, reward_att, detected] = simulateInteractionParallel(env_params, defender_action, attacker_action)
    % 并行环境中的简化交互模拟
    
    % 解析动作
    n_resource_types = length(env_params.resource_types);
    defense_allocation = zeros(env_params.total_components, n_resource_types);
    def_component = mod(defender_action - 1, env_params.total_components) + 1;
    def_resource = floor((defender_action - 1) / env_params.total_components) + 1;
    
    if def_resource <= n_resource_types
        defense_allocation(def_component, def_resource) = 1;
    end
    
    % 归一化资源分配
    total_alloc = sum(defense_allocation(:));
    if total_alloc > 0
        defense_allocation = defense_allocation / total_alloc * env_params.total_resources;
    end
    
    % 解析攻击动作
    attack_target = mod(attacker_action - 1, env_params.total_components) + 1;
    attack_type_idx = floor((attacker_action - 1) / env_params.total_components) + 1;
    attack_type_idx = min(attack_type_idx, length(env_params.attack_types));
    
    % 计算检测概率
    target_defense_strength = sum(defense_allocation(attack_target, :) .* env_params.resource_effectiveness');
    base_detection_prob = 1 - exp(-target_defense_strength / 30);
    final_detection_prob = base_detection_prob * (1 - env_params.attack_detection_difficulty(attack_type_idx));
    
    % 随机确定是否检测到
    detected = rand() < final_detection_prob;
    
    % 计算奖励
    attack_impact = env_params.component_importance(attack_target) * env_params.attack_severity(attack_type_idx);
    
    if detected
        reward_def = 10 * attack_impact;
        reward_att = -5;
    else
        reward_def = -20 * attack_impact;
        reward_att = 10 * attack_impact;
    end
end