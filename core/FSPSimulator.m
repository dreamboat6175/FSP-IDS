% FSPSimulator.m - FSP-TCS智能防御系统仿真核心逻辑
% =========================================================================
% 描述: 负责运行FSP仿真，包括智能体交互、状态更新和奖励计算。
% =========================================================================

classdef FSPSimulator
    methods(Static)
        function results = run(env, defenders, attacker, config, monitor)
            % run - 运行FSP仿真
            %   env: TCSEnvironment实例
            %   defenders: 防御者智能体(cell array of RLAgent objects)
            %   attacker: 攻击者智能体(RLAgent object)
            %   config: 配置结构体
            %   monitor: PerformanceMonitor实例

            % 将所有智能体组织成一个cell数组，以便ResultsCollector处理
            % ResultsCollector 期望第一个是攻击者，后面是防御者
            all_agents_list = [{attacker}, defenders]; 

            % 初始化结果收集器
            results_collector_obj = ResultsCollector(all_agents_list, config); 

            % 获取站点数量
            n_stations = config.n_stations;

            % 获取每个episode的步数，优先使用max_steps_per_episode，其次是max_episode_steps，最后是默认值
            if isfield(config, 'max_steps_per_episode')
                num_steps_per_episode = config.max_steps_per_episode;
            elseif isfield(config, 'max_episode_steps')
                num_steps_per_episode = config.max_episode_steps;
            else
                num_steps_per_episode = 50; % Default value if not specified in config
                fprintf('[WARNING] Config missing max_steps_per_episode or max_episode_steps. Using default: %d\n', num_steps_per_episode);
            end

            % 主仿真循环
            for iter = 1:config.n_iterations
                tic; % Start timer for the iteration
                fprintf('⏳ 执行第 %d/%d 次迭代...\n', iter, config.n_iterations);
                
                % 重置环境和智能体状态 (注意：QLearningAgent.reset() 已修改为不重置 update_count)
                env.reset(); % 重置环境
                for i = 1:length(defenders)
                    defenders{i}.reset(); % 重置防御者智能体 (重置 episode 相关的状态，不重置总更新次数)
                end
                attacker.reset(); % 重置攻击者智能体 (重置 episode 相关的状态，不重置总更新次数)

                % 初始化当前状态（所有站点初始状态相同，例如，都处于安全状态）
                % 状态向量可以简化为每个站点的安全级别或威胁等级
                % 这里假设状态是单一标量，表示系统整体状态或简化状态
                current_state_defender = env.current_state; 
                current_state_attacker = env.current_state; 

                % 为每个Episode运行多个步骤
                % Note: The actual episode running and agent updating logic is now primarily in runEpisodes.
                % The inner loop here is for the steps within a single episode.
                % The monitor.recordIterationResults was incorrectly placed here as it expects aggregated data.
                
                % Collect data for the current episode to pass to updateIterationData
                episode_defender_rewards_sum = zeros(1, length(defenders));
                episode_attacker_reward_sum = 0;
                episode_detection_rates_sum = zeros(1, length(defenders));
                episode_efficiency_sum = zeros(1, length(defenders)); % Assuming efficiency is also per defender

                for step = 1:num_steps_per_episode 
                    % === 1. 智能体选择动作 ===
                    % 防御者选择资源分配策略 (动作是资源分配向量)
                    defender_actions = cell(1, length(defenders));
                    for i = 1:length(defenders)
                        defender_actions{i} = defenders{i}.selectAction(current_state_defender);
                        % 确保防御者动作是归一化的资源分配
                        if sum(defender_actions{i}) > 0
                            defender_actions{i} = defender_actions{i} / sum(defender_actions{i});
                        else
                            defender_actions{i} = ones(size(defender_actions{i})) / length(defender_actions{i}); % Avoid division by zero
                        end
                        % 调试信息：打印防御者资源分配
                        if mod(defenders{i}.update_count, 100) == 0 || defenders{i}.update_count < 5
                           fprintf('[%s] 防御者 %s (更新次数 %d): 资源分配=%s\n', ...
                               class(defenders{i}), defenders{i}.name, defenders{i}.update_count, mat2str(defender_actions{i}, 3));
                        end
                    end

                    % 攻击者选择目标站点 (动作是站点索引)
                    attacker_action = attacker.selectAction(current_state_attacker);
                    % 确保攻击者动作是有效的站点索引
                    attacker_action = max(1, min(n_stations, round(attacker_action)));
                    % 调试信息：打印攻击者目标站点
                    if mod(attacker.update_count, 100) == 0 || attacker.update_count < 5
                        fprintf('[%s] 攻击者 %s (更新次数 %d): 选择目标站点=%d, 站点数=%d\n', ...
                                class(attacker), attacker.name, attacker.update_count, attacker_action, n_stations);
                    end
                    
                    % === 2. 环境执行动作并计算奖励和下一个状态 ===
                    % 这里需要根据实际的防御者数量和攻击者动作来计算奖励和下一个状态
                    % 简化处理：假设只有一个防御者或所有防御者协同
                    % 如果有多个防御者，需要聚合他们的资源分配
                    aggregated_defender_action = zeros(1, n_stations);
                    for i = 1:length(defenders)
                        aggregated_defender_action = aggregated_defender_action + defender_actions{i};
                    end
                    aggregated_defender_action = aggregated_defender_action / length(defenders); % 平均分配

                    [reward_attacker_step, reward_defenders_step, next_state_env] = ...
                        env.step(aggregated_defender_action, attacker_action);

                    % === 3. 智能体更新Q值表 ===
                    % 获取下一个状态的表示
                    next_state_defender = next_state_env;
                    next_state_attacker = next_state_env;

                    % 获取下一个动作 (用于SARSA，Q-learning只关心max Q)
                    % 这里为了兼容RLAgent的update接口，我们传递一个占位符
                    next_defender_action_for_update = 0; % 占位符
                    next_attacker_action_for_update = 0; % 占位符

                    for i = 1:length(defenders)
                        % !!! 关键：调用防御者智能体的更新方法 !!!
                        % 确保传递的是单个防御者的奖励
                        if isstruct(reward_defenders_step) && isfield(reward_defenders_step, 'defender')
                           current_defender_reward = reward_defenders_step.defender(i);
                        elseif isvector(reward_defenders_step) && length(reward_defenders_step) >= i
                           current_defender_reward = reward_defenders_step(i);
                        else
                           current_defender_reward = 0; % 默认值
                        end
                        defenders{i}.update(current_state_defender, defender_actions{i}, current_defender_reward, next_state_defender, next_defender_action_for_update);
                        episode_defender_rewards_sum(i) = episode_defender_rewards_sum(i) + current_defender_reward;
                    end
                    % !!! 关键：调用攻击者智能体的更新方法 !!!
                    % 确保传递的是单个攻击者的奖励
                    if isstruct(reward_attacker_step) && isfield(reward_attacker_step, 'attacker')
                        current_attacker_reward = reward_attacker_step.attacker;
                    elseif isscalar(reward_attacker_step)
                        current_attacker_reward = reward_attacker_step;
                    else
                        current_attacker_reward = 0; % 默认值
                    end
                    attacker.update(current_state_attacker, attacker_action, current_attacker_reward, next_state_attacker, next_attacker_action_for_update);
                    episode_attacker_reward_sum = episode_attacker_reward_sum + current_attacker_reward;

                    % Collect detection rates and efficiency for episode_results
                    % Assuming getEnvironmentInfo() gives a single scalar detection rate
                    % or needs to be adapted for per-defender detection rates if applicable
                    env_info_step = env.getEnvironmentInfo(); % Get info for current step
                    if isfield(env_info_step, 'recent_detection_rate')
                        episode_detection_rates_sum = episode_detection_rates_sum + repmat(env_info_step.recent_detection_rate, 1, length(defenders));
                    else
                        episode_detection_rates_sum = episode_detection_rates_sum + zeros(1, length(defenders)); % Fallback
                    end
                    
                    if isfield(env_info_step, 'recent_efficiency') % Assuming 'recent_efficiency' exists in env_info_step
                         episode_efficiency_sum = episode_efficiency_sum + repmat(env_info_step.recent_efficiency, 1, length(defenders));
                    else
                         episode_efficiency_sum = episode_efficiency_sum + zeros(1, length(defenders)); % Fallback
                    end


                    % === 4. 更新当前状态 ===
                    current_state_defender = next_state_env; 
                    current_state_attacker = next_state_env; 

                    % === 5. 记录和监控结果 (Removed from here, as updateIterationData expects aggregated episode data) ===
                    % The monitor.recordIterationResults was incorrectly placed here.
                    % It should either be an internal function of PerformanceMonitor
                    % that collects step-by-step data, or this line should be removed
                    % if only episode-level data is updated.
                    % Given the structure of PerformanceMonitor.updateIterationData,
                    % we assume only episode-level data is updated.

                end % end step loop

                % Aggregate episode results for updateIterationData
                num_steps = num_steps_per_episode;
                if num_steps == 0, num_steps = 1; end % Avoid division by zero if no steps ran

                episode_results_aggregated = struct();
                episode_results_aggregated.avg_defender_reward = episode_defender_rewards_sum / num_steps;
                episode_results_aggregated.avg_attacker_reward = episode_attacker_reward_sum / num_steps;
                episode_results_aggregated.avg_detection_rate = episode_detection_rates_sum / num_steps;
                episode_results_aggregated.avg_efficiency = episode_efficiency_sum / num_steps; % Added efficiency

                % 迭代结束后的性能更新
                monitor.updateIterationData(iter, episode_results_aggregated); 
                
                % 记录智能体参数历史（在QLearningAgent.m的recordPerformance中完成）
                iter_time = toc; % End timer for the iteration
                fprintf('[%s] 迭代 %d 完成，用时 %.2f秒\n', datestr(now,'yyyy-mm-dd HH:MM:SS'), iter, iter_time); % Corrected: Use iter_time
            end % end iter loop

            % 在仿真结束后收集所有智能体的数据
            results_collector_obj.collectFromAgents();
            results = results_collector_obj.getResults(); % 获取最终收集到的结果
        end
    end
end
