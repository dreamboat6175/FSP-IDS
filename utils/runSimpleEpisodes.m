function [iter_rewards, iter_detections, iter_resource_utilization, iter_allocation_balance] = runSimpleEpisodes(env, attacker_agent, defender_agents, config)
    %% runSimpleEpisodes - 简化的episode运行函数
    % 当FSPSimulator不存在时使用的备用函数
    % 输入:
    %   env - TCS环境对象
    %   attacker_agent - 攻击者智能体
    %   defender_agents - 防御者智能体数组
    %   config - 配置结构体
    % 输出:
    %   iter_rewards - 本次迭代的总奖励 (结构体包含 attacker_total 和 defender_total)
    %   iter_detections - 本次迭代的平均检测率
    %   iter_resource_utilization - 本次迭代的平均资源利用率 (每个防御者)
    %   iter_allocation_balance - 本次迭代的平均分配均衡性 (每个防御者)
    
    % 获取防御者数量和每个迭代的episode数量
    n_defenders = length(defender_agents); 
    n_episodes = config.simulation.n_episodes_per_iter; 
    
    % 初始化累积变量，用于存储每个episode的结果
    % 这些变量将作为最终的输出
    % total_defender_rewards 确保即使没有防御者 (n_defenders=0) 也能正确初始化为 n_episodes x 1
    total_attacker_rewards = zeros(1, n_episodes);
    total_defender_rewards = zeros(n_episodes, max(1, n_defenders)); 

    total_detections = zeros(1, n_episodes);
    % total_resource_utilization 和 total_allocation_balance 也确保即使没有防御者也能正确初始化
    total_resource_utilization = zeros(n_episodes, max(1, n_defenders)); 
    total_allocation_balance = zeros(n_episodes, max(1, n_defenders)); 
    
    % 运行episodes
    for ep = 1:n_episodes
        try
            % 重置环境到初始状态
            state = env.reset();
            
            % 每个episode的累积奖励和指标初始化
            % 确保这些累积变量在没有防御者时也至少是 1x1 数组，以避免维度不兼容错误
            episode_defender_rewards_sum = zeros(1, max(1, n_defenders));
            episode_attacker_reward_sum = 0;
            episode_detection_count = 0;
            episode_resource_utilization_sum = zeros(1, max(1, n_defenders));
            episode_allocation_balance_sum = zeros(1, max(1, n_defenders));

            % 假设每个episode有 max_episode_steps 步
            max_episode_steps = config.simulation.max_episode_steps;
            for step = 1:max_episode_steps
                % 智能体选择动作
                % 攻击者选择目标站点
                attacker_target_action = attacker_agent.selectAction(state);
                
                % 防御者选择资源部署
                defender_deployment_actions = cell(1, n_defenders);
                for d_idx = 1:n_defenders
                    defender_deployment_actions{d_idx} = defender_agents{d_idx}.selectAction(state);
                end
                
                % 环境交互
                % 这里为了匹配 TCSEnvironment.step 的单一部署输入，我们假设使用第一个防御者的部署
                % 如果有多个防御者，并且环境需要聚合它们的部署，您可能需要修改此逻辑
                if ~isempty(defender_deployment_actions) && n_defenders > 0
                    current_defender_deployment = defender_deployment_actions{1};
                else
                    % 如果没有防御者或部署，使用默认值（例如，平均分配资源）
                    current_defender_deployment = ones(1, config.system.n_stations) * (config.system.total_resources / config.system.n_stations);
                end

                % 执行环境步骤，获取下一个状态、奖励和信息
                [next_state, reward_def_env, reward_att_env, info] = env.step(current_defender_deployment, attacker_target_action);
                
                % 更新智能体
                % 攻击者更新策略
                try
                    if hasMethod(attacker_agent, 'update')
                        attacker_agent.update(state, attacker_target_action, reward_att_env, next_state);
                    end
                catch ME
                    % 仅在调试模式或特定episode打印警告，避免过多输出
                    if mod(ep, 50) == 0 || config.debug.debug_mode
                        warning('攻击者智能体更新失败 (Episode %d, Step %d): %s', ep, step, ME.message);
                    end
                end

                % 防御者更新策略
                for d_idx = 1:n_defenders
                    try
                        if hasMethod(defender_agents{d_idx}, 'update')
                            % 假设每个防御者接收其自身的奖励（这里简化为环境返回的reward_def_env）
                            % 在更复杂的博弈中，每个防御者可能有独立的奖励
                            defender_agents{d_idx}.update(state, defender_deployment_actions{d_idx}, reward_def_env, next_state);
                        end
                    catch ME
                        if mod(ep, 50) == 0 || config.debug.debug_mode
                            warning('防御者智能体 %d 更新失败 (Episode %d, Step %d): %s', d_idx, ep, step, ME.message);
                        end
                    end
                end
                
                % 累积本episode的奖励和指标
                % 关键修复：确保 reward_def_env 的维度与 episode_defender_rewards_sum 兼容
                current_step_defender_rewards = zeros(1, max(1, n_defenders)); % 初始化为正确大小
                if isscalar(reward_def_env)
                    % 如果是标量，将其广播到所有防御者
                    current_step_defender_rewards = reward_def_env * ones(1, max(1, n_defenders));
                elseif isvector(reward_def_env) && length(reward_def_env) == max(1, n_defenders)
                    % 如果是正确长度的向量，确保它是行向量
                    current_step_defender_rewards = reshape(reward_def_env, 1, []);
                else
                    % 如果维度不匹配，记录警告并使用零填充，防止后续错误
                    warning('runSimpleEpisodes:IncompatibleRewardDefEnv', ...
                            'Episode %d, Step %d: reward_def_env 维度不兼容 (%s)，期望大小为 1x%d。将使用零填充该步的防御者奖励。', ...
                            ep, step, mat2str(size(reward_def_env)), max(1, n_defenders));
                    % current_step_defender_rewards 已经初始化为零，无需额外操作
                end
                episode_defender_rewards_sum = episode_defender_rewards_sum + current_step_defender_rewards;
                
                episode_attacker_reward_sum = episode_attacker_reward_sum + reward_att_env;
                
                % 累积检测结果
                if isfield(info, 'detection_result') && isfield(info.detection_result, 'detected') && info.detection_result.detected
                    episode_detection_count = episode_detection_count + 1;
                end

                % 累积资源利用率和分配均衡性 (这里简化为只记录第一个防御者的)
                if n_defenders > 0 % 确保有防御者才尝试访问其信息
                    if isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
                        % 确保 resource_allocation 是行向量
                        current_resource_allocation = reshape(info.resource_allocation, 1, []);
                        if sum(current_resource_allocation) > 0
                            % 假设资源利用率是总分配资源与总资源的比值
                            episode_resource_utilization_sum(1) = episode_resource_utilization_sum(1) + sum(current_resource_allocation) / config.system.total_resources;
                        end
                    end
                    if isfield(info, 'current_allocation_balance')
                        % 确保 info.current_allocation_balance 始终是行向量，以避免维度不兼容问题
                        episode_allocation_balance_sum(1) = episode_allocation_balance_sum(1) + reshape(info.current_allocation_balance, 1, []);
                    elseif isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
                        % 如果没有直接的 balance 字段，从 resource_allocation 计算一个简化的平衡性指标
                        % 越接近1表示越均衡
                        if mean(info.resource_allocation) > 0
                             balance_val = 1 - (std(info.resource_allocation) / mean(info.resource_allocation));
                        else % 避免除以零
                            balance_val = 0; 
                        end
                        % 确保 balance_val 始终是行向量，以避免维度不兼容问题
                        episode_allocation_balance_sum(1) = episode_allocation_balance_sum(1) + reshape(balance_val, 1, []);
                    end
                end

                state = next_state; % 更新状态以进行下一步
            end % End of steps loop

            % 记录每个episode的总奖励和指标到累积变量中
            total_attacker_rewards(ep) = episode_attacker_reward_sum;
            % 修复：在赋值前确保 episode_defender_rewards_sum 是一个行向量且维度正确
            total_defender_rewards(ep, :) = reshape(episode_defender_rewards_sum, 1, max(1, n_defenders)); 
            total_detections(ep) = episode_detection_count / max_episode_steps; % 平均检测率
            total_resource_utilization(ep, :) = episode_resource_utilization_sum / max_episode_steps; % 平均资源利用率
            total_allocation_balance(ep, :) = episode_allocation_balance_sum / max_episode_steps; % 平均分配均衡性
            
        catch ME
            % 如果episode运行出错，记录警告并用零填充该episode的结果，以避免中断仿真
            warning('Episode %d 运行出错: %s', ep, ME.message);
            total_attacker_rewards(ep) = 0;
            total_defender_rewards(ep, :) = zeros(1, max(1, n_defenders)); 
            total_detections(ep) = 0;
            total_resource_utilization(ep, :) = zeros(1, max(1, n_defenders)); 
            total_allocation_balance(ep, :) = zeros(1, max(1, n_defenders)); 
        end
    end % End of episodes loop
    
    % 聚合整个迭代的奖励和指标 (平均值)
    iter_rewards.attacker_total = total_attacker_rewards;
    iter_rewards.defender_total = total_defender_rewards; % 矩阵: episodes x defenders

    iter_detections = mean(total_detections); % 整个迭代的平均检测率
    iter_resource_utilization = mean(total_resource_utilization, 1); % 整个迭代每个防御者的平均资源利用率
    iter_allocation_balance = mean(total_allocation_balance, 1); % 整个迭代每个防御者的平均分配均衡性
    
    fprintf('✓ 简化episodes运行完成 (%d个episodes)\n', n_episodes);
end

function has_method = hasMethod(obj, method_name)
    % 检查对象是否有指定方法
    try
        has_method = any(strcmp(methods(obj), method_name));
    catch
        has_method = false;
    end
end
