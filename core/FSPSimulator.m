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
    n_agents = length(defender_agents);
    n_episodes = config.n_episodes_per_iter;
    experience_buffer = cell(n_agents, 1);
    buffer_size = 1000;
    for i = 1:n_agents
        experience_buffer{i} = [];
    end
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
        for agent_idx = 1:n_agents
            defender = defender_agents{agent_idx};
            defender_action = defender.selectAction(state);
            if is_attack_episode
                current_allocation = env.getCurrentResourceAllocation();
                weak_areas = find(sum(current_allocation, 2) < mean(sum(current_allocation, 2)));
                if ~isempty(weak_areas)
                    target_component = weak_areas(randi(length(weak_areas)));
                    attack_type = randi([2, env.n_attack_types]);
                    attacker_action = (attack_type - 1) * env.total_components + target_component;
                else
                    attacker_action = attacker_agent.selectAction(state);
                end
            else
                attacker_action = 1;
            end
            [next_state, reward, ~, info] = env.step(defender_action, attacker_action);
            current_allocation = info.resource_allocation;
            radi = calculateRADI(current_allocation, config.radi.optimal_allocation, config.radi);
            efficiency = calculateResourceEfficiency(current_allocation, info);
            balance = calculateAllocationBalance(current_allocation);
            radi_reward = calculateRADIReward(radi, efficiency, balance, config);
            episode_results.radi_scores(ep, agent_idx) = radi;
            episode_results.resource_efficiency(ep, agent_idx) = efficiency;
            episode_results.allocation_balance(ep, agent_idx) = balance;
            episode_results.defender_rewards(ep, agent_idx) = radi_reward;
            episode_results.resource_allocations(ep, agent_idx, :) = current_allocation;
            experience = struct('state', state, 'action', defender_action, ...
                              'reward', radi_reward, 'next_state', next_state, ...
                              'info', info, 'radi', radi);
            experience_buffer{agent_idx}(end+1) = experience;
            if length(experience_buffer{agent_idx}) > buffer_size
                experience_buffer{agent_idx}(1) = [];
            end
            if isa(defender, 'SARSAAgent')
                next_action = defender.selectAction(next_state);
                defender.update(state, defender_action, radi_reward, next_state, next_action);
            else
                defender.update(state, defender_action, radi_reward, next_state, []);
            end
            if mod(ep, 10) == 0 && length(experience_buffer{agent_idx}) > 50
                radi_values = arrayfun(@(x) x.radi, experience_buffer{agent_idx});
                [~, sorted_idx] = sort(radi_values, 'descend');
                for replay = 1:5
                    if rand() < 0.7 && replay <= length(sorted_idx)/2
                        idx = sorted_idx(randi(ceil(length(sorted_idx)/2)));
                    else
                        idx = randi(length(experience_buffer{agent_idx}));
                    end
                    exp = experience_buffer{agent_idx}(idx);
                    if isa(defender, 'SARSAAgent')
                        next_action = defender.selectAction(exp.next_state);
                        defender.update(exp.state, exp.action, exp.reward, exp.next_state, next_action);
                    else
                        defender.update(exp.state, exp.action, exp.reward, exp.next_state, []);
                    end
                end
            end
        end
        avg_radi = mean(episode_results.radi_scores(ep, :));
        attacker_reward = avg_radi * 10;
        attacker_agent.update(state, attacker_action, attacker_reward, next_state, []);
        episode_results.attacker_rewards(ep) = attacker_reward;
        episode_results.attack_info{ep} = info;
    end
    episode_results.avg_radi = mean(episode_results.radi_scores, 1);
    episode_results.avg_efficiency = mean(episode_results.resource_efficiency, 1);
    episode_results.avg_balance = mean(episode_results.allocation_balance, 1);
    episode_results.avg_defender_reward = mean(episode_results.defender_rewards, 1);
    episode_results.avg_attacker_reward = mean(episode_results.attacker_rewards);
    episode_results.avg_resource_allocation = squeeze(mean(episode_results.resource_allocations, 1));
end

        
        function results = runParallelEpisodes(env, defender_agents, attacker_agent, config)
            n_episodes = config.n_episodes_per_iter;
            n_agents = length(defender_agents);
            radi_scores = zeros(n_episodes, n_agents);
            resource_efficiency = zeros(n_episodes, n_agents);
            allocation_balance = zeros(n_episodes, n_agents);
            defender_rewards = zeros(n_episodes, n_agents);
            attacker_rewards = zeros(n_episodes, 1);
            resource_allocations = zeros(n_episodes, n_agents, 5);
            env_params = struct();
            env_params.state_dim = env.state_dim;
            env_params.action_dim_defender = env.action_dim_defender;
            env_params.action_dim_attacker = env.action_dim_attacker;
            env_params.n_stations = env.n_stations;
            env_params.n_components = env.n_components;
            env_params.total_components = env.total_components;
            env_params.resource_types = env.resource_types;
            env_params.resource_effectiveness = env.resource_effectiveness;
            env_params.total_resources = env.total_resources;
            env_params.radi_config = config.radi;
            env_params.reward_config = config.reward;
            defender_q_tables = cell(1, n_agents);
            for i = 1:n_agents
                if isa(defender_agents{i}, 'DoubleQLearningAgent')
                    defender_q_tables{i} = (defender_agents{i}.Q1_table + defender_agents{i}.Q2_table) / 2;
                else
                    defender_q_tables{i} = defender_agents{i}.Q_table;
                end
            end
            if isa(attacker_agent, 'DoubleQLearningAgent')
                attacker_q_table = (attacker_agent.Q1_table + attacker_agent.Q2_table) / 2;
            else
                attacker_q_table = attacker_agent.Q_table;
            end
            parfor ep = 1:n_episodes
                ep_radi = zeros(1, n_agents);
                ep_efficiency = zeros(1, n_agents);
                ep_balance = zeros(1, n_agents);
                ep_def_rewards = zeros(1, n_agents);
                ep_allocations = zeros(n_agents, 5);
                ep_att_reward = 0;
                state = randi(env_params.state_dim);
                for agent_idx = 1:n_agents
                    if rand() < 0.1
                        def_action = randi(env_params.action_dim_defender);
                        att_action = randi(env_params.action_dim_attacker);
                    else
                        [~, def_action] = max(defender_q_tables{agent_idx}(state, :));
                        [~, att_action] = max(attacker_q_table(state, :));
                    end
                    [reward_def, reward_att, radi, efficiency, balance, allocation] = ...
                        simulateRADIInteractionParallel(env_params, def_action, att_action);
                    ep_radi(agent_idx) = radi;
                    ep_efficiency(agent_idx) = efficiency;
                    ep_balance(agent_idx) = balance;
                    ep_def_rewards(agent_idx) = reward_def;
                    ep_allocations(agent_idx, :) = allocation;
                    ep_att_reward = ep_att_reward + reward_att / n_agents;
                end
                radi_scores(ep, :) = ep_radi;
                resource_efficiency(ep, :) = ep_efficiency;
                allocation_balance(ep, :) = ep_balance;
                defender_rewards(ep, :) = ep_def_rewards;
                resource_allocations(ep, :, :) = ep_allocations;
                attacker_rewards(ep) = ep_att_reward;
            end
            results.radi_scores = radi_scores;
            results.resource_efficiency = resource_efficiency;
            results.allocation_balance = allocation_balance;
            results.defender_rewards = defender_rewards;
            results.attacker_rewards = attacker_rewards;
            results.resource_allocations = resource_allocations;
            results.avg_radi = mean(radi_scores, 1);
            results.avg_efficiency = mean(resource_efficiency, 1);
            results.avg_balance = mean(allocation_balance, 1);
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
            checkpoint_dir = sprintf('checkpoints/iter_%d', iter);
            if ~exist(checkpoint_dir, 'dir')
                mkdir(checkpoint_dir);
            end
            for i = 1:length(defender_agents)
                filename = fullfile(checkpoint_dir, sprintf('defender_%d.mat', i));
                defender_agents{i}.save(filename);
            end
            filename = fullfile(checkpoint_dir, 'attacker.mat');
            attacker_agent.save(filename);
            monitor_data = monitor.getResults();
            if isfield(monitor_data, 'metrics_history')
                monitor_data.radi_summary = struct();
                monitor_data.radi_summary.final_radi = mean(monitor_data.metrics_history.radi_scores(end-99:end));
                monitor_data.radi_summary.best_radi = min(monitor_data.metrics_history.radi_scores);
                monitor_data.radi_summary.radi_improvement = ...
                    monitor_data.metrics_history.radi_scores(1) - monitor_data.metrics_history.radi_scores(end);
            end
            save(fullfile(checkpoint_dir, 'monitor_data.mat'), 'monitor_data');
            fprintf('✓ 检查点已保存: %s\n', checkpoint_dir);
            if exist('monitor_data', 'var') && isfield(monitor_data, 'radi_summary')
                fprintf('  当前RADI: %.3f\n', monitor_data.radi_summary.final_radi);
                fprintf('  最佳RADI: %.3f\n', monitor_data.radi_summary.best_radi);
            end
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

function [reward_def, reward_att, radi, efficiency, balance, allocation] = ...
    simulateRADIInteractionParallel(env_params, defender_action, attacker_action)
    n_resource_types = length(env_params.resource_types);
    allocation = zeros(1, n_resource_types);
    allocation_idx = mod(defender_action - 1, 5) + 1;
    allocation_strategies = [
        0.3, 0.2, 0.2, 0.15, 0.15;
        0.5, 0.2, 0.1, 0.1, 0.1;
        0.2, 0.4, 0.2, 0.1, 0.1;
        0.2, 0.2, 0.3, 0.15, 0.15;
        0.2, 0.2, 0.2, 0.2, 0.2;
    ];
    if allocation_idx <= 5
        allocation = allocation_strategies(allocation_idx, :);
    else
        allocation = rand(1, 5);
        allocation = allocation / sum(allocation);
    end
    optimal_allocation = env_params.radi_config.optimal_allocation;
    weights = [
        env_params.radi_config.weight_computation,
        env_params.radi_config.weight_bandwidth,
        env_params.radi_config.weight_sensors,
        env_params.radi_config.weight_scanning,
        env_params.radi_config.weight_inspection
    ];
    deviation = abs(allocation - optimal_allocation);
    radi = sum(weights .* deviation);
    utilization = sum(allocation .* env_params.resource_effectiveness) / n_resource_types;
    efficiency = min(1, utilization);
    cv = std(allocation) / (mean(allocation) + eps);
    balance = 1 / (1 + cv);
    radi_penalty = -radi * 50;
    efficiency_bonus = efficiency * 30;
    balance_bonus = balance * 20;
    reward_def = env_params.reward_config.w_radi * radi_penalty + ...
                 env_params.reward_config.w_efficiency * efficiency_bonus + ...
                 env_params.reward_config.w_balance * balance_bonus;
    reward_att = radi * 10;
    if radi <= env_params.radi_config.threshold_excellent
        reward_def = reward_def + 50;
        reward_att = reward_att - 20;
    end
end

%% RADI相关计算函数
function radi = calculateRADI(current_allocation, optimal_allocation, radi_config)
    weights = [
        radi_config.weight_computation,
        radi_config.weight_bandwidth,
        radi_config.weight_sensors,
        radi_config.weight_scanning,
        radi_config.weight_inspection
    ];
    deviation = abs(current_allocation - optimal_allocation);
    radi = sum(weights .* deviation);
end
function efficiency = calculateResourceEfficiency(allocation, info)
    if isfield(info, 'defended_successfully')
        total_resources = sum(allocation);
        efficiency = info.defended_successfully / (1 + total_resources/100);
    else
        utilization = sum(allocation) / 100;
        efficiency = min(1, utilization);
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