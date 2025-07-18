function updateAgentParameters(defender_agents, attacker_agent, config)
    %% updateAgentParameters - 更新智能体学习参数
    % 输入:
    %   defender_agents - 防御者智能体数组
    %   attacker_agent - 攻击者智能体
    %   config - 配置结构体（包含更新后的参数）
    
    try
        % 更新防御者智能体参数
        updateDefenderAgents(defender_agents, config);
        
        % 更新攻击者智能体参数
        updateAttackerAgent(attacker_agent, config);
        
        fprintf('✓ 智能体参数已更新\n');
        
    catch ME
        warning('更新智能体参数时出错: %s', ME.message);
    end
end

function updateDefenderAgents(defender_agents, config)
    %% updateDefenderAgents - 更新防御者智能体参数
    
    for i = 1:length(defender_agents)
        agent = defender_agents{i};
        
        try
            % 更新学习率
            if isfield(config, 'learning_rate')
                updateAgentProperty(agent, 'learning_rate', config.learning_rate);
            end
            
            % 更新探索率(epsilon)
            if isfield(config, 'epsilon')
                updateAgentProperty(agent, 'epsilon', config.epsilon);
            end
            
            % 更新折扣因子
            if isfield(config, 'discount_factor')
                updateAgentProperty(agent, 'discount_factor', config.discount_factor);
            end
            
            % 更新温度参数(用于softmax)
            if isfield(config, 'temperature')
                updateAgentProperty(agent, 'temperature', config.temperature);
            end
            
            % 更新探索衰减率
            if isfield(config, 'epsilon_decay')
                updateAgentProperty(agent, 'epsilon_decay', config.epsilon_decay);
            end
            
            % 更新最小探索率
            if isfield(config, 'epsilon_min')
                updateAgentProperty(agent, 'epsilon_min', config.epsilon_min);
            end
            
            % 更新学习率衰减
            if isfield(config, 'learning_rate_decay')
                updateAgentProperty(agent, 'learning_rate_decay', config.learning_rate_decay);
            end
            
            % 更新最小学习率
            if isfield(config, 'learning_rate_min')
                updateAgentProperty(agent, 'learning_rate_min', config.learning_rate_min);
            end
            
            % 更新经验回放相关参数
            if isfield(config, 'replay_buffer_size')
                updateAgentProperty(agent, 'replay_buffer_size', config.replay_buffer_size);
            end
            
            if isfield(config, 'batch_size')
                updateAgentProperty(agent, 'batch_size', config.batch_size);
            end
            
            % 更新目标网络更新频率（如果适用）
            if isfield(config, 'target_update_frequency')
                updateAgentProperty(agent, 'target_update_frequency', config.target_update_frequency);
            end
            
            % SARSA特有参数
            if isfield(config, 'trace_decay') && isa(agent, 'SARSAAgent')
                updateAgentProperty(agent, 'trace_decay', config.trace_decay);
            end
            
            % Double Q-Learning特有参数
            if isfield(config, 'update_probability') && contains(class(agent), 'DoubleQ')
                updateAgentProperty(agent, 'update_probability', config.update_probability);
            end
            
        catch ME
            warning('更新防御者智能体 %d 参数时出错: %s', i, ME.message);
        end
    end
end

function updateAttackerAgent(attacker_agent, config)
    %% updateAttackerAgent - 更新攻击者智能体参数
    
    if isempty(attacker_agent)
        return;
    end
    
    try
        % 更新学习率
        if isfield(config, 'learning_rate')
            updateAgentProperty(attacker_agent, 'learning_rate', config.learning_rate);
        end
        
        % 更新探索率
        if isfield(config, 'epsilon')
            updateAgentProperty(attacker_agent, 'epsilon', config.epsilon);
        end
        
        % 更新折扣因子
        if isfield(config, 'discount_factor')
            updateAgentProperty(attacker_agent, 'discount_factor', config.discount_factor);
        end
        
        % 更新温度参数
        if isfield(config, 'temperature')
            updateAgentProperty(attacker_agent, 'temperature', config.temperature);
        end
        
        % 更新探索衰减率
        if isfield(config, 'epsilon_decay')
            updateAgentProperty(attacker_agent, 'epsilon_decay', config.epsilon_decay);
        end
        
        % 更新最小探索率
        if isfield(config, 'epsilon_min')
            updateAgentProperty(attacker_agent, 'epsilon_min', config.epsilon_min);
        end
        
        % 更新学习率衰减
        if isfield(config, 'learning_rate_decay')
            updateAgentProperty(attacker_agent, 'learning_rate_decay', config.learning_rate_decay);
        end
        
        % 更新最小学习率
        if isfield(config, 'learning_rate_min')
            updateAgentProperty(attacker_agent, 'learning_rate_min', config.learning_rate_min);
        end
        
        % 攻击者特有参数
        if isfield(config, 'attack_power')
            updateAgentProperty(attacker_agent, 'attack_power', config.attack_power);
        end
        
        if isfield(config, 'attack_frequency')
            updateAgentProperty(attacker_agent, 'attack_frequency', config.attack_frequency);
        end
        
        if isfield(config, 'stealth_factor')
            updateAgentProperty(attacker_agent, 'stealth_factor', config.stealth_factor);
        end
        
    catch ME
        warning('更新攻击者智能体参数时出错: %s', ME.message);
    end
end

function updateAgentProperty(agent, property_name, new_value)
    %% updateAgentProperty - 安全地更新智能体属性
    % 输入:
    %   agent - 智能体对象
    %   property_name - 属性名称
    %   new_value - 新值
    
    try
        % 检查是否为对象属性
        if isprop(agent, property_name)
            old_value = agent.(property_name);
            agent.(property_name) = new_value;
            if mod(randi(100), 50) == 0  % 偶尔显示更新信息
                fprintf('  更新 %s.%s: %.4f -> %.4f\n', class(agent), property_name, old_value, new_value);
            end
            return;
        end
        
        % 检查是否为结构体字段
        if isstruct(agent) && isfield(agent, property_name)
            old_value = agent.(property_name);
            agent.(property_name) = new_value;
            if mod(randi(100), 50) == 0
                fprintf('  更新 %s: %.4f -> %.4f\n', property_name, old_value, new_value);
            end
            return;
        end
        
        % 尝试通过setter方法更新
        setter_name = sprintf('set%s', property_name);
        if hasMethod(agent, setter_name)
            agent.(setter_name)(new_value);
            return;
        end
        
        % 尝试直接字段访问（适用于一些特殊情况）
        try
            eval(sprintf('agent.%s = new_value;', property_name));
        catch
            % 如果都失败了，记录警告但不中断程序
            if mod(randi(100), 20) == 0  % 偶尔显示警告
                warning('无法更新智能体属性: %s', property_name);
            end
        end
        
    catch ME
        % 静默处理错误，避免中断主程序
        if mod(randi(100), 20) == 0
            warning('更新属性 %s 时出错: %s', property_name, ME.message);
        end
    end
end

function has_method = hasMethod(obj, method_name)
    %% hasMethod - 检查对象是否有指定方法
    try
        if isobject(obj)
            has_method = any(strcmp(methods(obj), method_name));
        else
            has_method = false;
        end
    catch
        has_method = false;
    end
end