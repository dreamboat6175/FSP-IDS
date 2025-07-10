%% RLAgent.m - 强化学习智能体基类
% =========================================================================
% 描述: 定义所有强化学习智能体的基类接口
% =========================================================================

classdef (Abstract) RLAgent < handle
    
    properties
        % 基本属性
        name                % 智能体名称
        agent_type         % 智能体类型（defender/attacker）
        
        % 学习参数
        learning_rate      % 学习率
        discount_factor    % 折扣因子
        epsilon           % 探索率
        epsilon_min       % 最小探索率
        epsilon_decay     % 探索率衰减
        
        % 策略相关
        strategy_pool     % 策略池
        pool_size_limit   % 策略池大小限制
        current_policy    % 当前策略
        
        % 状态和动作空间
        state_dim         % 状态维度
        action_dim        % 动作维度
        
        % 性能跟踪
        update_count      % 更新次数
        total_reward      % 累计奖励
        episode_rewards   % 每轮奖励记录
        action_history    % 动作历史
        
        % 其他参数
        temperature       % Boltzmann温度参数
        use_softmax      % 是否使用softmax策略
    end
    
    methods (Abstract)
        % 必须实现的抽象方法
        action = selectAction(obj, state)              % 选择动作
        update(obj, state, action, reward, next_state) % 更新策略
        policy = getPolicy(obj)                         % 获取当前策略
        save(obj, filename)                             % 保存模型
        load(obj, filename)                             % 加载模型
    end
    
    methods
        function obj = RLAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            
            obj.name = name;
            obj.agent_type = agent_type;
            
            % 学习参数初始化
            obj.learning_rate = config.learning_rate;
            obj.discount_factor = config.discount_factor;
            obj.epsilon = config.epsilon;
            obj.epsilon_min = config.epsilon_min;
            obj.epsilon_decay = config.epsilon_decay;
            
            % 状态和动作空间
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            
            % 策略池初始化
            obj.strategy_pool = {};
            obj.pool_size_limit = config.pool_size_limit;
            
            % 性能跟踪初始化
            obj.update_count = 0;
            obj.total_reward = 0;
            obj.episode_rewards = [];
            obj.action_history = [];
            
            % 其他参数
            if isfield(config, 'temperature')
                obj.temperature = config.temperature;
            else
                obj.temperature = 1.0;
            end
            obj.use_softmax = false;
        end
        
        function updateEpsilon(obj)
            % 更新探索率
            obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
        end
        
        function updateStrategyPool(obj)
            % 更新策略池
            current_policy = obj.getPolicy();
            obj.strategy_pool{end+1} = current_policy;
            
            % 限制策略池大小
            if length(obj.strategy_pool) > obj.pool_size_limit
                obj.strategy_pool(1) = [];
            end
        end
        
        function recordReward(obj, reward)
            % 记录奖励
            obj.total_reward = obj.total_reward + reward;
            obj.episode_rewards(end+1) = reward;
        end
        
        function recordAction(obj, state, action)
            % 记录动作历史
            obj.action_history(end+1, :) = [state, action];
        end
        
        function stats = getStatistics(obj)
            % 获取智能体统计信息
            stats.name = obj.name;
            stats.update_count = obj.update_count;
            stats.total_reward = obj.total_reward;
            stats.avg_reward = mean(obj.episode_rewards);
            stats.current_epsilon = obj.epsilon;
            stats.strategy_pool_size = length(obj.strategy_pool);
            
            if ~isempty(obj.episode_rewards)
                stats.recent_avg_reward = mean(obj.episode_rewards(max(1,end-99):end));
            else
                stats.recent_avg_reward = 0;
            end
        end
        
        function reset(obj)
            % 重置智能体状态（用于新的episode）
            % 保留学习的知识，只重置临时状态
        end
        
        function action = epsilonGreedyAction(obj, state, q_values)
            % ε-贪婪动作选择
            if rand() < obj.epsilon
                action = randi(obj.action_dim);
            else
                [~, action] = max(q_values);
            end
        end
        
        function action = boltzmannAction(obj, state, q_values)
            % Boltzmann动作选择（softmax）
            exp_q = exp(q_values / obj.temperature);
            probs = exp_q / sum(exp_q);
            
            % 处理数值问题
            if any(isnan(probs))
                probs = ones(size(probs)) / length(probs);
            end
            
            % 根据概率选择动作
            cumsum_probs = cumsum(probs);
            r = rand();
            action = find(cumsum_probs >= r, 1);
            
            if isempty(action)
                action = randi(obj.action_dim);
            end
        end
    end
end