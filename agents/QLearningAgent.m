%% QLearningAgent.m - Q-Learning智能体实现
% =========================================================================
% 描述: 实现标准Q-Learning算法的智能体
% =========================================================================

classdef QLearningAgent < RLAgent
    
    properties
        Q_table          % Q值表
        visit_count      % 状态-动作访问计数
        lr_scheduler     % 学习率调度器
    end
    
    methods
        function obj = QLearningAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 初始化Q表
            obj.Q_table = zeros(state_dim, action_dim);
            obj.visit_count = zeros(state_dim, action_dim);
            
            % 初始化Q表（乐观初始化）
            initial_value = 5.0;
            noise_level = 0.5;
            obj.Q_table = ones(state_dim, action_dim) * initial_value + ...
                          randn(state_dim, action_dim) * noise_level;
            
            % 初始化学习率调度器
            obj.lr_scheduler = struct();
            obj.lr_scheduler.initial_lr = config.learning_rate;
            obj.lr_scheduler.min_lr = 0.001;
            obj.lr_scheduler.decay_steps = 1000;
            obj.lr_scheduler.current_lr = config.learning_rate;
            obj.lr_scheduler.step_count = 0;
            obj.lr_scheduler.decay_rate = 0.99;
        end
        
        function action = selectAction(obj, state)
            % 选择动作 - 使用epsilon-greedy策略
            
            % 确保状态是有效的
            if isempty(state) || ~isnumeric(state)
                action = randi(obj.action_dim);
                return;
            end
            
            % 获取状态索引
            state_idx = obj.encodeState(state);
            
            % 调试信息
            if isnan(state_idx) || isinf(state_idx) || state_idx < 1 || state_idx > obj.state_dim
                warning('QLearningAgent: Invalid state_idx = %g, state_dim = %d', state_idx, obj.state_dim);
                state_idx = 1;
            end
            
            % Epsilon-greedy动作选择
            if rand() < obj.epsilon
                % 探索：随机选择
                action = randi(obj.action_dim);
            else
                % 利用：选择最优动作
                q_values = obj.Q_table(state_idx, :);
                [~, action] = max(q_values);
            end
        end
        
        function update(obj, state, action, reward, next_state, ~)
            % Q值更新 - 标准Q-learning算法
            
            % 输入验证
            if isempty(state) || isempty(action) || isempty(next_state)
                return;
            end
            
            % 确保action是标量
            if numel(action) > 1
                action = action(1);
            end
            
            % 编码状态
            state_idx = obj.encodeState(state);
            next_state_idx = obj.encodeState(next_state);
            
            % 获取当前Q值
            current_q = obj.Q_table(state_idx, action);
            
            % 计算目标值
            max_next_q = max(obj.Q_table(next_state_idx, :));
            target = reward + obj.discount_factor * max_next_q;
            
            % 计算TD误差
            td_error = target - current_q;
            
            % 获取自适应学习率
            lr = obj.getCurrentLearningRate(state_idx, action);
            
            % 更新Q值
            obj.Q_table(state_idx, action) = current_q + lr * td_error;
            
            % 更新访问计数
            obj.visit_count(state_idx, action) = obj.visit_count(state_idx, action) + 1;
            
            % 更新学习率调度器
            obj.updateLearningRateScheduler();
            
            % 更新探索率
            obj.updateEpsilon();
        end
        
        function reset(obj)
            % 重置智能体状态（保留学习的知识）
            obj.epsilon = obj.config.epsilon; % 可选：重置探索率
        end
        
        function saveModel(obj, filename)
            % 保存模型
            agent_data = struct();
            agent_data.Q_table = obj.Q_table;
            agent_data.visit_count = obj.visit_count;
            agent_data.config = obj.config;
            agent_data.lr_scheduler = obj.lr_scheduler;
            save(filename, 'agent_data');
        end
        
        function loadModel(obj, filename)
            % 加载模型
            if exist(filename, 'file')
                loaded = load(filename);
                obj.Q_table = loaded.agent_data.Q_table;
                obj.visit_count = loaded.agent_data.visit_count;
                obj.lr_scheduler = loaded.agent_data.lr_scheduler;
            end
        end
        
        function strategy = getStrategy(obj)
    % 获取当前策略分布
    
    % 对于防御者，返回资源分配策略
    if contains(obj.name, 'defender')
        % 计算每个站点的平均Q值
        n_stations = 10;  % 或从配置获取
        n_resources = 5;
        strategy = zeros(1, n_stations);
        
        for station = 1:n_stations
            station_q_values = [];
            for resource = 1:n_resources
                action_idx = (station - 1) * n_resources + resource;
                if action_idx <= obj.action_dim
                    avg_q = mean(obj.Q_table(:, action_idx));
                    station_q_values(end+1) = avg_q;
                end
            end
            if ~isempty(station_q_values)
                strategy(station) = mean(station_q_values);
            end
        end
        
        % 归一化
        if sum(strategy) > 0
            strategy = strategy / sum(strategy);
        else
            strategy = ones(1, n_stations) / n_stations;
        end
        
    else  % 攻击者
        % 返回攻击概率分布
        avg_q = mean(obj.Q_table, 1);
        if any(avg_q)
            strategy = exp(avg_q) / sum(exp(avg_q));  % softmax
        else
            strategy = ones(1, obj.action_dim) / obj.action_dim;
        end
    end
    
    % 确保是行向量
    strategy = strategy(:)';
end

    end
    
    methods (Access = private)
        function state_idx = encodeState(obj, state)
            % 将状态向量编码为索引
            
            % 确保state是向量
            if isempty(state) || ~isnumeric(state)
                state_idx = 1;
                return;
            end
            
            % 将state转换为列向量并确保是数值
            state = double(state(:));
            
            % 处理NaN和Inf值
            state(isnan(state)) = 0;
            state(isinf(state)) = 0;
            
            % 使用更简单的哈希函数
            if length(state) == 1
                % 如果只有一个元素，直接使用
                hash_value = abs(state(1));
            else
                % 使用前几个元素的和作为哈希值
                hash_value = sum(abs(state(1:min(5, length(state)))));
            end
            
            % 确保哈希值是有限的正数
            if isnan(hash_value) || isinf(hash_value) || hash_value < 0
                hash_value = 1;
            end
            
            % 将哈希值映射到状态空间
            state_idx = mod(floor(hash_value), obj.state_dim) + 1;
            
            % 确保索引有效
            state_idx = max(1, min(state_idx, obj.state_dim));
            
            % 最终检查
            if isnan(state_idx) || isinf(state_idx) || state_idx < 1 || state_idx > obj.state_dim
                state_idx = 1;
            end
        end
        
        function lr = getCurrentLearningRate(obj, state_idx, action_idx)
            % 获取当前学习率（可以基于访问次数自适应）
            
            visit_count = obj.visit_count(state_idx, action_idx);
            if visit_count > 0
                % 基于访问次数的自适应学习率
                lr = obj.lr_scheduler.current_lr / (1 + visit_count * 0.01);
            else
                lr = obj.lr_scheduler.current_lr;
            end
            
            lr = max(lr, obj.lr_scheduler.min_lr);
        end
        
        function updateLearningRateScheduler(obj)
            % 更新学习率调度器
            
            obj.lr_scheduler.step_count = obj.lr_scheduler.step_count + 1;
            
            if mod(obj.lr_scheduler.step_count, obj.lr_scheduler.decay_steps) == 0
                obj.lr_scheduler.current_lr = max(...
                    obj.lr_scheduler.current_lr * obj.lr_scheduler.decay_rate, ...
                    obj.lr_scheduler.min_lr);
            end
        end
        
        function updateEpsilon(obj)
            % 更新探索率
            obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
        end
    end
end