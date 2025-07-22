function [iter_rewards, iter_detections, iter_resource_utilization, iter_allocation_balance] = runSimpleEpisodes(env, attacker_agent, defender_agents, config)
    %% runSimpleEpisodes - 简化的episode运行函数
    % 当FSPSimulator不存在时使用的备用函数
    % 输入:
    %   env - TCS环境对象
    %   attacker_agent - 攻击者智能体 (修正: 调整顺序以匹配main_fsp.m的调用)
    %   defender_agents - 防御者智能体数组 (修正: 调整顺序以匹配main_fsp.m的调用)
    %   config - 配置结构体
    % 输出:
    %   iter_rewards - 本次迭代的总奖励 (修正: 增加为输出参数)
    %   iter_detections - 本次迭代的总检测结果 (修正: 增加为输出参数)
    %   iter_resource_utilization - 本次迭代的总资源利用率 (修正: 增加为输出参数)
    %   iter_allocation_balance - 本次迭代的总分配均衡性 (修正: 增加为输出参数)
    
    n_defenders = length(defender_agents); % 修正: 使用n_defenders来表示防御者数量
    n_episodes = config.simulation.n_episodes_per_iter; % 修正: 从config.simulation中获取
    
    % 初始化累积变量，用于存储每个episode的结果
    % 这些变量将作为最终的输出
    total_attacker_rewards = zeros(1, n_episodes);
    total_defender_rewards = zeros(n_episodes, max(1, n_defenders)); % 修正: 确保至少有一列，避免n_defenders=0时维度问题

    total_detections = zeros(1, n_episodes);
    total_resource_utilization = zeros(n_episodes, max(1, n_defenders)); % 修正: 确保至少有一列
    total_allocation_balance = zeros(n_episodes, max(1, n_defenders)); % 修正: 确保至少有一列
    
    % 运行episodes
    for ep = 1:n_episodes
        try
            % 重置环境
            state = env.reset();
            
            % 每个episode的累积奖励和指标
            % 修复: 确保这些累积变量在没有防御者时也至少是 1x1 数组，以避免维度不兼容错误
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
                
                % 环境交互 (这里简化为只考虑第一个防御者的部署对环境的影响)
                % 您可能需要根据实际仿真逻辑，决定哪个防御者的部署影响环境，或者如何聚合多个防御者的部署
                % 这里为了匹配 TCSEnvironment.step 的单一部署输入，我们假设使用第一个防御者的部署
                if ~isempty(defender_deployment_actions) && n_defenders > 0
                    current_defender_deployment = defender_deployment_actions{1};
                else
                    % 如果没有防御者或部署，使用默认值
                    current_defender_deployment = ones(1, config.system.n_stations) * (config.system.total_resources / config.system.n_stations);
                end

                [next_state, reward_def_env, reward_att_env, info] = env.step(current_defender_deployment, attacker_target_action);
                
                % 更新智能体
                % 攻击者更新
                try
                    if hasMethod(attacker_agent, 'update')
                        attacker_agent.update(state, attacker_target_action, reward_att_env, next_state);
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
                % 修复: 如果 reward_def_env 是标量，将其扩展为与防御者数量匹配的行向量
                % 这样可以避免在有多个防御者时，标量与行向量相加的维度不兼容问题
                if isscalar(reward_def_env)
                    episode_defender_rewards_sum = episode_defender_rewards_sum + (reward_def_env * ones(1, max(1, n_defenders)));
                else
                    % 如果 reward_def_env 已经是向量，确保它是行向量并直接相加
                    episode_defender_rewards_sum = episode_defender_rewards_sum + reshape(reward_def_env, 1, []);
                end
                episode_attacker_reward_sum = episode_attacker_reward_sum + reward_att_env;
                
                if isfield(info, 'detection_result') && isfield(info.detection_result, 'detected') && info.detection_result.detected
                    episode_detection_count = episode_detection_count + 1;
                end

                % 累积资源利用率和分配均衡性 (这里简化为只记录第一个防御者的)
                if n_defenders > 0 % 确保有防御者才尝试访问其信息
                    if isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
                        % 确保 resource_allocation 是行向量
                        current_resource_allocation = reshape(info.resource_allocation, 1, []);
                        if sum(current_resource_allocation) > 0
                            episode_resource_utilization_sum(1) = episode_resource_utilization_sum(1) + sum(current_resource_allocation) / config.system.total_resources;
                        end
                    end
                    if isfield(info, 'current_allocation_balance')
                        % 修复: 确保 info.current_allocation_balance 始终是行向量，以避免维度不兼容问题
                        episode_allocation_balance_sum(1) = episode_allocation_balance_sum(1) + reshape(info.current_allocation_balance, 1, []);
                    elseif isfield(info, 'resource_allocation') && ~isempty(info.resource_allocation)
                        % 如果没有直接的 balance 字段，从 resource_allocation 计算一个简化的
                        if std(info.resource_allocation) > 0
                            balance_val = 1 - (std(info.resource_allocation) / mean(info.resource_allocation));
                        else
                            balance_val = 1.0;
                        end
                        % 修复: 确保 balance_val 始终是行向量，以避免维度不兼容问题
                        episode_allocation_balance_sum(1) = episode_allocation_balance_sum(1) + reshape(balance_val, 1, []);
                    end
                end

                state = next_state; % 更新状态
            end % End of steps loop

            % 记录每个episode的总奖励和指标
            total_attacker_rewards(ep) = episode_attacker_reward_sum;
            total_defender_rewards(ep, :) = episode_defender_rewards_sum; % 修正: 赋值整个行向量
            total_detections(ep) = episode_detection_count / max_episode_steps; % 平均检测率
            total_resource_utilization(ep, :) = episode_resource_utilization_sum / max_episode_steps; % 平均资源利用率
            total_allocation_balance(ep, :) = episode_allocation_balance_sum / max_episode_steps; % 平均分配均衡性
            
        catch ME
            warning('Episode %d 运行出错: %s', ep, ME.message);
            % 如果episode出错，用零填充该episode的结果，以避免中断仿真
            total_attacker_rewards(ep) = 0;
            total_defender_rewards(ep, :) = zeros(1, max(1, n_defenders)); % 修复: 确保在错误时也正确处理维度
            total_detections(ep) = 0;
            total_resource_utilization(ep, :) = zeros(1, max(1, n_defenders)); % 修复: 确保在错误时也正确处理维度
            total_allocation_balance(ep, :) = zeros(1, max(1, n_defenders)); % 修复: 确保在错误时也正确处理维度
        end
    end % End of episodes loop
    
    % 聚合整个迭代的奖励和指标
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
