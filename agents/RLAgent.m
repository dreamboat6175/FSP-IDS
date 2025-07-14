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

        function state_idx = getStateIndex(obj, state_vec)
            % 改进的状态索引映射，减少冲突

            % 特征提取
            features = obj.extractKeyFeatures(state_vec);

            % 使用改进的哈希函数
            % 将特征向量转换为字符串
            feature_str = sprintf('%.4f,', features);

            % 使用Java的hashCode获得更好的分布
            hash_val = java.lang.String(feature_str).hashCode();

            % 映射到有效范围
            state_idx = mod(abs(hash_val), obj.state_dim) + 1;

            % 二次哈希处理冲突
            if obj.checkCollision(state_idx, features)
                hash_val2 = java.lang.String(fliplr(feature_str)).hashCode();
                state_idx = mod(abs(hash_val2), obj.state_dim) + 1;
            end
        end

        function features = extractKeyFeatures(obj, state_vec)
            % 提取关键特征，降低维度
            features = [];

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
            if ~isempty(features)
                features = features / (max(abs(features)) + 1e-6);
            end
        end

        function collision = checkCollision(obj, state_idx, features)
            % 简单的冲突检测
            persistent feature_cache;
            if isempty(feature_cache)
                feature_cache = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            end

            collision = false;
            if isKey(feature_cache, state_idx)
                cached_features = feature_cache(state_idx);
                if norm(cached_features - features) > 0.1
                    collision = true;
                end
            else
                feature_cache(state_idx) = features;
            end
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
        
        function action_vec = convertToStationActions(obj, q_values, n_stations)
            % 将Q值转换为站点级动作向量
            action_vec = zeros(1, n_stations);
            
            % 计算每个站点的资源类型数量
            n_resource_types = round(obj.action_dim / n_stations);
            
            % 确保资源类型数量是有效的正整数
            if n_resource_types <= 0 || isinf(n_resource_types) || isnan(n_resource_types)
                n_resource_types = 5; % 默认值
            end
            
            % 为每个站点选择最佳资源类型
            for station = 1:n_stations
                start_idx = (station-1) * n_resource_types + 1;
                end_idx = min(station * n_resource_types, length(q_values));
                if start_idx > end_idx || start_idx > length(q_values)
                    station_q_values = [];
                else
                    station_q_values = q_values(start_idx:end_idx);
                end
                
                % 确保Q值有效
                if any(isnan(station_q_values)) || any(isinf(station_q_values))
                    station_q_values = ones(size(station_q_values)) * 1.0;
                end
                
                % ε-贪婪选择
                if rand() < obj.epsilon
                    action_vec(station) = randi([1, max(1, n_resource_types)]);  % 随机选择资源类型
                else
                    if isempty(station_q_values)
                        best_action = 1;
                    else
                        [~, best_action] = max(station_q_values);
                        if ~isscalar(best_action)
                            best_action = best_action(1);
                        end
                    end
                    action_vec(station) = best_action;
                end
                
                % 确保动作在有效范围内
                action_vec(station) = max(1, min(max(1, n_resource_types), round(action_vec(station))));
            end
        end
    end
end
