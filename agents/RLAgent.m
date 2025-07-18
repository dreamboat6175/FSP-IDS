%% RLAgent.m - 强化学习智能体基类 (修复版)
% =========================================================================
% 描述: 所有RL智能体的抽象基类，添加了缺失的encodeState方法
% =========================================================================

classdef (Abstract) RLAgent < handle
    
    properties (Access = public)
        name                % 智能体名称
        agent_type          % 智能体类型
        state_dim           % 状态空间维度
        action_dim          % 动作空间维度
        config              % 配置参数
        
        % 学习参数
        learning_rate       % 学习率
        discount_factor     % 折扣因子
        epsilon             % 探索率
        epsilon_min         % 最小探索率
        epsilon_decay       % 探索率衰减
        
        % 附加参数
        temperature         % 温度参数（用于softmax）
        temperature_decay   % 温度衰减
        temperature_min     % 最小温度
        learning_rate_min   % 最小学习率
        learning_rate_decay % 学习率衰减
        
        % 性能跟踪
        strategy_pool       % 策略池
        pool_size_limit     % 策略池大小限制
        update_count        % 更新计数
        total_reward        % 总奖励
        episode_rewards     % episode奖励历史
        action_history      % 动作历史
        use_softmax         % 是否使用softmax策略
    end
    
    methods (Abstract)
        % 抽象方法 - 子类必须实现
        action = selectAction(obj, state)
        update(obj, state, action, reward, next_state, next_action)
    end
    
    methods
        function obj = RLAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            
            obj.name = name;
            obj.agent_type = agent_type;
            obj.config = config;
            obj.state_dim = state_dim;
            obj.action_dim = action_dim;
            
            % 从配置中提取参数
            if isfield(config, 'learning_rate')
                obj.learning_rate = config.learning_rate;
            else
                obj.learning_rate = 0.1;
            end
            
            if isfield(config, 'discount_factor')
                obj.discount_factor = config.discount_factor;
            else
                obj.discount_factor = 0.95;
            end
            
            if isfield(config, 'epsilon')
                obj.epsilon = config.epsilon;
            else
                obj.epsilon = 0.3;
            end
            
            if isfield(config, 'epsilon_min')
                obj.epsilon_min = config.epsilon_min;
            else
                obj.epsilon_min = 0.01;
            end
            
            if isfield(config, 'epsilon_decay')
                obj.epsilon_decay = config.epsilon_decay;
            else
                obj.epsilon_decay = 0.995;
            end
            
            % 初始化附加参数
            obj.temperature = 1.0;
            obj.temperature_decay = 0.995;
            obj.temperature_min = 0.1;
            obj.learning_rate_min = 0.001;
            obj.learning_rate_decay = 0.9995;
            
            % 初始化性能跟踪
            obj.strategy_pool = {};
            obj.pool_size_limit = 50;
            obj.update_count = 0;
            obj.total_reward = 0;
            obj.episode_rewards = [];
            obj.action_history = [];
            obj.use_softmax = false;
        end
        
        function state_idx = encodeState(obj, state_vec)
            % 状态编码方法 - 将状态向量映射到状态索引
            % 输入: state_vec - 状态向量
            % 输出: state_idx - 状态索引 (1到state_dim之间)
            
            % 健壮性检查
            if isempty(state_vec)
                state_idx = 1;
                return;
            end
            
            % 确保state_vec是数值向量
            if ~isnumeric(state_vec)
                state_idx = 1;
                return;
            end
            
            % 将状态向量转换为标量特征
            if length(state_vec) == 1
                % 如果已经是标量
                state_feature = state_vec;
            else
                % 计算状态向量的特征
                state_feature = obj.computeStateFeature(state_vec);
            end
            
            % 将特征映射到状态索引
            % 使用简单的模运算确保索引在有效范围内
            if isnan(state_feature) || isinf(state_feature)
                state_idx = 1;
            else
                % 归一化特征值并映射到状态空间
                normalized_feature = mod(abs(state_feature), 1000); % 限制特征值范围
                state_idx = floor(normalized_feature * obj.state_dim / 1000) + 1;
                
                % 确保索引在有效范围内
                state_idx = max(1, min(obj.state_dim, state_idx));
            end
        end
        
        function feature = computeStateFeature(obj, state_vec)
            % 计算状态向量的特征值
            % 可以被子类覆盖以实现更复杂的特征提取
            
            try
                % 方法1：使用向量的加权和
                weights = linspace(1, 0.1, length(state_vec));
                if length(weights) ~= length(state_vec)
                    weights = ones(size(state_vec));
                end
                feature = sum(state_vec .* weights);
                
                % 方法2：如果上述方法失败，使用简单的平均值
                if isnan(feature) || isinf(feature)
                    feature = mean(state_vec);
                end
                
                % 方法3：最后的备用方案
                if isnan(feature) || isinf(feature)
                    feature = sum(state_vec);
                end
                
                % 最终备用方案
                if isnan(feature) || isinf(feature)
                    feature = 1.0;
                end
                
            catch
                % 如果所有方法都失败，返回默认值
                feature = 1.0;
            end
        end
        
        function updateStateIndex(obj, state_vec, state_idx)
            % 可选的状态索引更新方法
            % 可以用于实现更复杂的状态管理
            % 默认实现：不做任何操作
        end
        
        function state_idx = getStateIndex(obj, state_vec)
            % 改进的状态索引映射，减少冲突
            % 这是encodeState的别名，保持向后兼容性
            state_idx = obj.encodeState(state_vec);
        end

        function features = extractKeyFeatures(obj, state_vec)
            % 提取关键特征，降低维度
            features = [];
            
            try
                % 统计特征
                features(1) = mean(state_vec);
                features(2) = std(state_vec);
                features(3) = max(state_vec);
                features(4) = min(state_vec);
                features(5) = median(state_vec);

                % 分段特征（将状态向量分成几段）
                n_segments = min(5, floor(length(state_vec)/10));
                if n_segments > 0
                    segment_size = floor(length(state_vec)/n_segments);
                    for i = 1:n_segments
                        start_idx = (i-1)*segment_size + 1;
                        end_idx = min(i*segment_size, length(state_vec));
                        features(5+i) = mean(state_vec(start_idx:end_idx));
                    end
                end

                % 归一化
                if ~isempty(features) && max(abs(features)) > 0
                    features = features / (max(abs(features)) + 1e-6);
                end
                
            catch
                % 如果特征提取失败，返回基本特征
                features = [mean(state_vec), std(state_vec)];
            end
        end

        function updateStrategyPool(obj)
            % 更新策略池
            try
                current_policy = obj.getPolicy();
                obj.strategy_pool{end+1} = current_policy;

                % 限制策略池大小
                if length(obj.strategy_pool) > obj.pool_size_limit
                    obj.strategy_pool(1) = [];
                end
            catch
                % 如果获取策略失败，跳过更新
            end
        end

        function recordReward(obj, reward)
            % 记录奖励
            if isnumeric(reward) && ~isnan(reward) && ~isinf(reward)
                obj.total_reward = obj.total_reward + reward;
                obj.episode_rewards(end+1) = reward;
            end
        end

        function recordAction(obj, state_idx, action)
            % 记录动作
            try
                obj.action_history(end+1, :) = [state_idx, action];
            catch
                % 如果记录失败，跳过
            end
        end
        
        function reset(obj)
            % 重置智能体（可被子类覆盖）
            % 默认实现：不做任何操作
        end
        
        function info = getInfo(obj)
            % 获取智能体信息
            info = struct();
            info.name = obj.name;
            info.type = obj.agent_type;
            info.learning_rate = obj.learning_rate;
            info.epsilon = obj.epsilon;
            info.state_dim = obj.state_dim;
            info.action_dim = obj.action_dim;
        end
        
        function updateParameters(obj)
            % 更新学习参数（衰减等）
            
            % 更新探索率
            obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
            
            % 更新学习率
            obj.learning_rate = max(obj.learning_rate_min, ...
                                   obj.learning_rate * obj.learning_rate_decay);
            
            % 更新温度
            obj.temperature = max(obj.temperature_min, ...
                                 obj.temperature * obj.temperature_decay);
        end
        
        function decay(obj)
            % 参数衰减方法（与updateParameters相同，提供别名）
            obj.updateParameters();
        end
        
        function policy = getPolicy(obj)
            % 获取当前策略 - 默认实现
            % 子类应该覆盖此方法以提供具体的策略
            policy = ones(1, obj.action_dim) / obj.action_dim;
        end
        
        function strategy = getStrategy(obj)
            % 获取当前策略分布（与getPolicy相同）
            strategy = obj.getPolicy();
        end
        
        function updateExperience(obj, state, action, reward, next_state, done)
            % 更新经验的默认实现
            % 子类应该覆盖此方法以实现具体的学习更新
            
            % 记录奖励
            obj.recordReward(reward);
            
            % 调用抽象的update方法
            try
                obj.update(state, action, reward, next_state, []);
            catch
                % 如果update方法失败，至少记录奖励
            end
        end
        
        function stats = getStatistics(obj)
            % 获取智能体统计信息 - 默认实现
            % 子类可以覆盖此方法以提供更详细的统计信息
            
            stats = struct();
            stats.name = obj.name;
            stats.agent_type = obj.agent_type;
            stats.update_count = obj.update_count;
            stats.total_reward = obj.total_reward;
            stats.current_learning_rate = obj.learning_rate;
            stats.current_epsilon = obj.epsilon;
            stats.total_episodes = length(obj.episode_rewards);
            
            if ~isempty(obj.episode_rewards)
                stats.avg_episode_reward = mean(obj.episode_rewards);
                stats.best_episode_reward = max(obj.episode_rewards);
                stats.worst_episode_reward = min(obj.episode_rewards);
            else
                stats.avg_episode_reward = 0;
                stats.best_episode_reward = 0;
                stats.worst_episode_reward = 0;
            end
        end
    end
end