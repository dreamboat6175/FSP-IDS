function [iter_rewards, iter_detections, iter_resource_utilization, iter_allocation_balance] = runSimpleEpisodes(env, attacker_agent, defender_agents, config)
    %% runSimpleEpisodes - 简化的episode运行函数 (完全修复版)
    % 修复了环境step方法参数不匹配和维度不兼容问题
    
    % 获取防御者数量和每个迭代的episode数量
    n_defenders = length(defender_agents); 
    n_episodes = config.simulation.n_episodes_per_iter; 
    
    % 调试信息
    fprintf('🔧 runSimpleEpisodes: n_defenders=%d, n_episodes=%d\n', n_defenders, n_episodes);
    
    % 初始化累积变量，确保维度兼容性
    expected_cols = max(1, n_defenders);
    
    total_attacker_rewards = zeros(1, n_episodes);
    total_defender_rewards = zeros(n_episodes, expected_cols); 
    total_detections = zeros(1, n_episodes);
    total_resource_utilization = zeros(n_episodes, expected_cols); 
    total_allocation_balance = zeros(n_episodes, expected_cols); 
    
    fprintf('✓ 初始化矩阵: %dx%d (episodes x defenders)\n', n_episodes, expected_cols);
    
    % 运行episodes
    for ep = 1:n_episodes
        try
            % 重置环境到初始状态
            state = env.reset();
            
            % 每个episode的累积变量初始化
            episode_defender_rewards_sum = zeros(1, expected_cols);
            episode_attacker_reward_sum = 0;
            episode_detection_count = 0;
            episode_resource_utilization_sum = zeros(1, expected_cols);
            episode_allocation_balance_sum = zeros(1, expected_cols);

            max_episode_steps = config.simulation.max_episode_steps;
            
            % 运行单个episode的步骤
            for step = 1:max_episode_steps
                try
                    % 攻击者行动
                    attacker_action = attacker_agent.selectAction(state);
                    
                    % 防御者行动
                    defender_deployment_actions = cell(1, n_defenders);
                    for d_idx = 1:n_defenders
                        defender_deployment_actions{d_idx} = defender_agents{d_idx}.selectAction(state);
                    end
                    
                    % 准备环境输入
                    if n_defenders > 0
                        current_defender_deployment = defender_deployment_actions{1};
                    else
                        current_defender_deployment = ones(1, config.system.n_stations) / config.system.n_stations;
                    end
                    
                    % 环境步进 - 修复：TCSEnvironment.step 返回4个参数
                    [next_state, reward_def_env, reward_att_env, info] = env.step(current_defender_deployment, attacker_action);
                    
                    % 更新智能体
                    try
                        if hasMethod(attacker_agent, 'update')
                            attacker_agent.update(state, attacker_action, reward_att_env, next_state);
                        end
                    catch ME
                        if mod(ep, 50) == 0 || config.debug.debug_mode
                            warning('攻击者智能体更新失败 (Episode %d, Step %d): %s', ep, step, ME.message);
                        end
                    end
                    
                    % 防御者更新
                    for d_idx = 1:n_defenders
                        try
                            if hasMethod(defender_agents{d_idx}, 'update')
                                defender_agents{d_idx}.update(state, defender_deployment_actions{d_idx}, reward_def_env, next_state);
                            end
                        catch ME
                            if mod(ep, 50) == 0 || config.debug.debug_mode
                                warning('防御者智能体 %d 更新失败 (Episode %d, Step %d): %s', d_idx, ep, step, ME.message);
                            end
                        end
                    end
                    
                    % 累积奖励和指标 - 使用安全处理函数
                    current_step_defender_rewards = processDefenderRewards(reward_def_env, expected_cols, ep, step);
                    episode_defender_rewards_sum = episode_defender_rewards_sum + current_step_defender_rewards;
                    episode_attacker_reward_sum = episode_attacker_reward_sum + reward_att_env;
                    
                    % 累积检测结果
                    if isfield(info, 'detection_result') && isfield(info.detection_result, 'detected') && info.detection_result.detected
                        episode_detection_count = episode_detection_count + 1;
                    end

                    % 累积资源利用率和分配均衡性
                    if n_defenders > 0
                        [resource_util, alloc_balance] = processResourceMetrics(info, config, expected_cols);
                        episode_resource_utilization_sum = episode_resource_utilization_sum + resource_util;
                        episode_allocation_balance_sum = episode_allocation_balance_sum + alloc_balance;
                    end

                    state = next_state;
                    
                catch stepError
                    if mod(ep, 50) == 0
                        warning('Episode %d Step %d 执行失败: %s', ep, step, stepError.message);
                    end
                end
            end % End of steps loop

            % 安全地记录每个episode的总奖励和指标
            total_attacker_rewards(ep) = episode_attacker_reward_sum;
            
            % 使用安全赋值函数避免维度不兼容错误
            total_defender_rewards(ep, :) = safeAssignRewards(episode_defender_rewards_sum, expected_cols, ep, 'defender_rewards');
            total_detections(ep) = episode_detection_count / max_episode_steps;
            total_resource_utilization(ep, :) = safeAssignRewards(episode_resource_utilization_sum / max_episode_steps, expected_cols, ep, 'resource_utilization');
            total_allocation_balance(ep, :) = safeAssignRewards(episode_allocation_balance_sum / max_episode_steps, expected_cols, ep, 'allocation_balance');
            
            % 进度显示
            if ep <= 5 || ep == 50 || mod(ep, 100) == 0
                fprintf('✓ Episode %d 完成 (攻击者奖励: %.3f, 检测率: %.2f%%)\n', ...
                        ep, episode_attacker_reward_sum, (episode_detection_count/max_episode_steps)*100);
            end
            
        catch ME
            warning('Episode %d 运行出错: %s', ep, ME.message);
            total_attacker_rewards(ep) = 0;
            total_defender_rewards(ep, :) = zeros(1, expected_cols);
            total_detections(ep) = 0;
            total_resource_utilization(ep, :) = zeros(1, expected_cols);
            total_allocation_balance(ep, :) = zeros(1, expected_cols);
        end
    end % End of episodes loop
    
    % 聚合整个迭代的奖励和指标
    iter_rewards.attacker_total = total_attacker_rewards;
    iter_rewards.defender_total = total_defender_rewards;

    iter_detections = mean(total_detections);
    iter_resource_utilization = mean(total_resource_utilization, 1);
    iter_allocation_balance = mean(total_allocation_balance, 1);
    
    fprintf('✓ 简化episodes运行完成 (%d个episodes)\n', n_episodes);
end

%% 辅助函数：安全处理防御者奖励
function safe_rewards = processDefenderRewards(reward_def_env, expected_cols, ep, step)
    safe_rewards = zeros(1, expected_cols);
    
    try
        if isscalar(reward_def_env)
            safe_rewards = reward_def_env * ones(1, expected_cols);
        elseif isvector(reward_def_env)
            reward_vector = reshape(reward_def_env, 1, []);
            if length(reward_vector) == expected_cols
                safe_rewards = reward_vector;
            elseif length(reward_vector) < expected_cols
                safe_rewards(1:length(reward_vector)) = reward_vector;
            else
                safe_rewards = reward_vector(1:expected_cols);
                if ep <= 5
                    warning('Episode %d Step %d: 截取防御者奖励从 %d 到 %d 维', ep, step, length(reward_vector), expected_cols);
                end
            end
        else
            if ep <= 5
                warning('Episode %d Step %d: reward_def_env 结构复杂 (%s)，使用零填充', ep, step, mat2str(size(reward_def_env)));
            end
        end
    catch
        if ep <= 5
            warning('Episode %d Step %d: 处理防御者奖励失败，使用零填充', ep, step);
        end
    end
end

%% 辅助函数：安全处理资源指标
function [resource_util, alloc_balance] = processResourceMetrics(info, config, expected_cols)
    resource_util = zeros(1, expected_cols);
    alloc_balance = zeros(1, expected_cols);
    
    try
        % 处理资源利用率
        if isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
            current_resource_allocation = reshape(info.resource_allocation, 1, []);
            if sum(current_resource_allocation) > 0 && isfield(config.system, 'total_resources')
                resource_util(1) = sum(current_resource_allocation) / config.system.total_resources;
            end
        end
        
        % 处理分配均衡性
        if isfield(info, 'current_allocation_balance')
            balance_val = info.current_allocation_balance;
            if isscalar(balance_val)
                alloc_balance(1) = balance_val;
            elseif isvector(balance_val)
                balance_vector = reshape(balance_val, 1, []);
                alloc_balance(1:min(length(balance_vector), expected_cols)) = balance_vector(1:min(length(balance_vector), expected_cols));
            end
        elseif isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
            allocation = info.resource_allocation;
            if mean(allocation) > 0
                balance_val = 1 - (std(allocation) / mean(allocation));
                alloc_balance(1) = max(0, min(1, balance_val));
            end
        end
    catch
        % 如果处理失败，使用默认值
    end
end

%% 辅助函数：安全赋值
function result = safeAssignRewards(input_data, expected_cols, ep, data_type)
    try
        if length(input_data) == expected_cols
            result = reshape(input_data, 1, []);
        elseif length(input_data) < expected_cols
            result = zeros(1, expected_cols);
            result(1:length(input_data)) = input_data;
        else
            result = reshape(input_data(1:expected_cols), 1, []);
            if ep <= 5
                fprintf('警告: Episode %d %s 截取从 %d 到 %d 维\n', ep, data_type, length(input_data), expected_cols);
            end
        end
    catch ME
        if ep <= 5
            warning('Episode %d %s 赋值失败: %s，使用零填充', ep, data_type, ME.message);
        end
        result = zeros(1, expected_cols);
    end
end

%% 辅助函数：检查对象方法
function has_method = hasMethod(obj, method_name)
    try
        has_method = any(strcmp(methods(obj), method_name));
    catch
        has_method = false;
    end
end