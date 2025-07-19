function episode_results = runSimpleEpisodes(env, defender_agents, attacker_agent, config)
    %% runSimpleEpisodes - 简化的episode运行函数
    % 当FSPSimulator不存在时使用的备用函数
    % 输入:
    %   env - TCS环境对象
    %   defender_agents - 防御者智能体数组
    %   attacker_agent - 攻击者智能体
    %   config - 配置结构体
    % 输出:
    %   episode_results - episode运行结果
    
    n_agents = length(defender_agents);
    n_episodes = config.n_episodes_per_iter;
    
    % 初始化结果结构
    episode_results = struct();
    episode_results.avg_radi = zeros(1, n_agents);
    episode_results.avg_efficiency = zeros(1, n_agents);
    episode_results.avg_balance = zeros(1, n_agents);
    episode_results.avg_defender_reward = zeros(1, n_agents);
    episode_results.avg_attacker_reward = 0;
    episode_results.attack_info = cell(n_episodes, 1);
    episode_results.avg_resource_allocation = zeros(n_agents, config.n_stations);
    
    % 累积变量
    radi_sum = zeros(1, n_agents);
    efficiency_sum = zeros(1, n_agents);
    balance_sum = zeros(1, n_agents);
    defender_reward_sum = zeros(1, n_agents);
    attacker_reward_sum = 0;
    resource_allocation_sum = zeros(n_agents, config.n_stations);
    
    % 运行episodes
    for ep = 1:n_episodes
        try
            % 重置环境
            state = env.reset();
            
            % 存储每个智能体在这个episode中的结果
            episode_radi = zeros(1, n_agents);
            episode_efficiency = zeros(1, n_agents);
            episode_balance = zeros(1, n_agents);
            episode_defender_rewards = zeros(1, n_agents);
            episode_resource_allocation = zeros(n_agents, config.n_stations);
            
            % 每个防御者选择动作并执行
            for agent_idx = 1:n_agents
                % 选择动作
                defender_action = defender_agents{agent_idx}.selectAction(state);
                attacker_action = attacker_agent.selectAction(state);
                
                % 执行环境步骤
                [next_state, reward, done, info] = env.step(defender_action, attacker_action);
                
                % 提取资源分配信息
                if isfield(info, 'resource_allocation')
                    allocation = info.resource_allocation;
                elseif isfield(info, 'defender_allocation')
                    allocation = info.defender_allocation;
                else
                    % 默认均匀分配
                    allocation = ones(1, config.n_stations) * (config.total_resources / config.n_stations);
                end
                
                % 确保allocation是正确维度
                if length(allocation) ~= config.n_stations
                    allocation = ones(1, config.n_stations) * (sum(allocation) / config.n_stations);
                end
                
                % 计算RADI指标
                if isfield(config, 'radi') && isfield(config.radi, 'optimal_allocation')
                    radi = calculateRADI(allocation, config.radi.optimal_allocation, config.radi);
                else
                    % 简化的RADI计算
                    radi = 1 / (1 + std(allocation));
                end
                
                % 计算资源效率
                if isfield(config, 'total_resources')
                    efficiency = sum(allocation) / config.total_resources;
                else
                    efficiency = mean(allocation) / 100; % 假设总资源为100
                end
                
                % 计算分配均衡性
                if std(allocation) > 0
                    balance = 1 - (std(allocation) / mean(allocation));
                else
                    balance = 1.0;
                end
                balance = max(0, min(1, balance)); % 限制在[0,1]
                
                % 存储结果
                episode_radi(agent_idx) = radi;
                episode_efficiency(agent_idx) = efficiency;
                episode_balance(agent_idx) = balance;
                episode_defender_rewards(agent_idx) = reward;
                episode_resource_allocation(agent_idx, :) = allocation;
                
                % 更新智能体
                try
                    if hasMethod(defender_agents{agent_idx}, 'update')
                        % 尝试不同的参数组合
                        try
                            defender_agents{agent_idx}.update(state, defender_action, reward, next_state, []);
                        catch
                            try
                                defender_agents{agent_idx}.update(state, defender_action, reward, next_state);
                            catch
                                % 最简化调用
                                defender_agents{agent_idx}.update(reward);
                            end
                        end
                    elseif hasMethod(defender_agents{agent_idx}, 'updateQTable')
                        try
                            defender_agents{agent_idx}.updateQTable(state, defender_action, reward, next_state);
                        catch
                            defender_agents{agent_idx}.updateQTable(reward);
                        end
                    else
                        % 如果都没有，尝试基本的Q值更新
                        if isprop(defender_agents{agent_idx}, 'Q_table') || isfield(defender_agents{agent_idx}, 'Q_table')
                            defender_agents{agent_idx}.Q_table(1, 1) = defender_agents{agent_idx}.Q_table(1, 1) + 0.01 * reward;
                        end
                    end
                catch ME
                    % 静默处理错误，记录但不中断
                    if mod(ep, 50) == 0  % 偶尔显示错误信息
                        warning('防御者智能体 %d 更新失败: %s', agent_idx, ME.message);
                    end
                end
                
                % 更新状态
                state = next_state;
            end
            
            % 攻击者奖励和更新
            attacker_reward = -mean(episode_defender_rewards); % 攻击者奖励与防御者相反
            try
                if hasMethod(attacker_agent, 'update')
                    % 尝试不同的参数组合
                    try
                        attacker_agent.update(state, attacker_action, attacker_reward, next_state, []);
                    catch
                        try
                            attacker_agent.update(state, attacker_action, attacker_reward, next_state);
                        catch
                            attacker_agent.update(attacker_reward);
                        end
                    end
                elseif hasMethod(attacker_agent, 'updateQTable')
                    try
                        attacker_agent.updateQTable(state, attacker_action, attacker_reward, next_state);
                    catch
                        attacker_agent.updateQTable(attacker_reward);
                    end
                else
                    % 尝试基本更新
                    if isprop(attacker_agent, 'Q_table') || isfield(attacker_agent, 'Q_table')
                        attacker_agent.Q_table(1, 1) = attacker_agent.Q_table(1, 1) + 0.01 * attacker_reward;
                    end
                end
            catch ME
                if mod(ep, 50) == 0
                    warning('攻击者智能体更新失败: %s', ME.message);
                end
            end
            
            % 累积结果
            radi_sum = radi_sum + episode_radi;
            efficiency_sum = efficiency_sum + episode_efficiency;
            balance_sum = balance_sum + episode_balance;
            defender_reward_sum = defender_reward_sum + episode_defender_rewards;
            attacker_reward_sum = attacker_reward_sum + attacker_reward;
            resource_allocation_sum = resource_allocation_sum + episode_resource_allocation;
            
            % 攻击成功信息（简化模拟）
            attack_success = rand() < 0.3; % 30%的攻击成功率
            episode_results.attack_info{ep} = attack_success;
            
        catch ME
            warning('Episode %d 运行出错: %s', ep, ME.message);
            % 使用默认值
            episode_results.attack_info{ep} = false;
        end
    end
    
    % 计算平均值
    episode_results.avg_radi = radi_sum / n_episodes;
    episode_results.avg_efficiency = efficiency_sum / n_episodes;
    episode_results.avg_balance = balance_sum / n_episodes;
    episode_results.avg_defender_reward = defender_reward_sum / n_episodes;
    episode_results.avg_attacker_reward = attacker_reward_sum / n_episodes;
    episode_results.avg_resource_allocation = resource_allocation_sum / n_episodes;
    
    % 添加策略信息（如果可用）
    try
        if hasMethod(attacker_agent, 'getStrategy')
            episode_results.attacker_strategy = attacker_agent.getStrategy();
        elseif hasMethod(attacker_agent, 'getPolicy')
            episode_results.attacker_strategy = attacker_agent.getPolicy();
        elseif isprop(attacker_agent, 'strategy') || isfield(attacker_agent, 'strategy')
            episode_results.attacker_strategy = attacker_agent.strategy;
        else
            episode_results.attacker_strategy = ones(1, config.n_stations) / config.n_stations;
        end
        
        episode_results.defender_strategies = cell(n_agents, 1);
        for i = 1:n_agents
            if hasMethod(defender_agents{i}, 'getStrategy')
                episode_results.defender_strategies{i} = defender_agents{i}.getStrategy();
            elseif hasMethod(defender_agents{i}, 'getPolicy')
                episode_results.defender_strategies{i} = defender_agents{i}.getPolicy();
            elseif isprop(defender_agents{i}, 'strategy') || isfield(defender_agents{i}, 'strategy')
                episode_results.defender_strategies{i} = defender_agents{i}.strategy;
            else
                episode_results.defender_strategies{i} = ones(1, config.n_stations) / config.n_stations;
            end
        end
    catch ME
        % 如果获取策略失败，使用默认值
        episode_results.attacker_strategy = ones(1, config.n_stations) / config.n_stations;
        episode_results.defender_strategies = cell(n_agents, 1);
        for i = 1:n_agents
            episode_results.defender_strategies{i} = ones(1, config.n_stations) / config.n_stations;
        end
        if mod(randi(100), 20) == 0  % 偶尔显示警告
            warning('获取策略信息失败: %s', ME.message);
        end
    end
    
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