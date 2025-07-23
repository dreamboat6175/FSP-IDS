classdef DoubleQLearningAgent < RLAgent
    properties
        Q_table_A        % 第一个Q值表
        Q_table_B        % 第二个Q值表
        Q_table          % 兼容性Q表（两个Q表的平均）
        visit_count      % 状态-动作访问计数 
        lr_scheduler     % 学习率调度器
        strategy_history     % 策略历史记录
        performance_history  % 性能历史记录
        parameter_history    % 参数历史记录
    end
    
    methods
        function obj = DoubleQLearningAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 初始化两个Q表
            initial_value = 5.0;
            noise_level = 0.5;
            obj.Q_table_A = ones(state_dim, action_dim) * initial_value + ...
                           randn(state_dim, action_dim) * noise_level;
            obj.Q_table_B = ones(state_dim, action_dim) * initial_value + ...
                           randn(state_dim, action_dim) * noise_level;
            
            % 初始化兼容性Q表
            obj.Q_table = (obj.Q_table_A + obj.Q_table_B) / 2;
            
            % 初始化访问计数
            obj.visit_count = zeros(state_dim, action_dim);
            
            % 初始化学习率调度器
            obj.lr_scheduler = struct();
            if isfield(config, 'learning_rate')
                obj.lr_scheduler.initial_lr = config.learning_rate;
                obj.lr_scheduler.current_lr = config.learning_rate;
            else
                obj.lr_scheduler.initial_lr = 0.15;
                obj.lr_scheduler.current_lr = 0.15;
            end
            obj.lr_scheduler.min_lr = 0.001;
            obj.lr_scheduler.decay_steps = 1000;
            obj.lr_scheduler.step_count = 0;
            obj.lr_scheduler.decay_rate = 0.99;
            
            % ===== 重要修复：移除 use_softmax 属性定义 =====
            % 现在使用基类的 exploration_strategy 属性
            % obj.use_softmax = false; % 删除这行，改用基类属性
            
            % 确保基类属性有默认值
            obj.strategy_history = [];
            obj.performance_history = struct();
            obj.parameter_history = struct();
            obj.parameter_history.learning_rate = [];
            obj.parameter_history.epsilon = [];
            obj.parameter_history.q_values = [];
        end
        
        function action_vec = selectAction(obj, state_vec)
            % 动作选择
            
            % 更新兼容性Q表
            obj.updateQTableProperty();
            
            % 健壮性检查
            if isempty(state_vec)
                state_vec = ones(1, obj.state_dim);
            end
            state_vec = reshape(state_vec, 1, []);
            
            % 获取状态索引
            state_idx = obj.encodeState(mean(state_vec));
            state_idx = max(1, min(state_idx, size(obj.Q_table, 1)));
            
            % 获取组合Q值
            q_values = obj.Q_table(state_idx, :);
            
            % 确保Q值有效
            if any(isnan(q_values)) || any(isinf(q_values))
                q_values = randn(size(q_values)) * 0.1;
            end
            
            % 动态调整参数
            if obj.epsilon_decay < 1
                obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
            end
            
            % ===== 修复：使用 exploration_strategy 而不是 use_softmax =====
            if strcmp(obj.exploration_strategy, 'softmax') && obj.temperature_decay < 1
                obj.temperature = max(0.1, obj.temperature * obj.temperature_decay);
            end
            
            % ===== 区分防御者和攻击者的动作生成 =====
            if contains(obj.agent_type, 'attacker') || contains(obj.name, 'attacker')
                % ===== 攻击者：返回单个站点索引 =====
                
                % 确定站点数量
                if isprop(obj, 'config') && isfield(obj.config, 'n_stations')
                    n_stations = obj.config.n_stations;
                else
                    n_stations = min(obj.action_dim, 10);
                end
                
                % ===== 修复：使用 exploration_strategy 判断 =====
                if strcmp(obj.exploration_strategy, 'softmax')
                    % Softmax选择
                    temperature = max(0.1, obj.temperature);
                    q_normalized = q_values(1:min(n_stations, length(q_values))) - max(q_values(1:min(n_stations, length(q_values))));
                    exp_values = exp(q_normalized / temperature);
                    probabilities = exp_values / sum(exp_values);
                    
                    % 基于概率选择动作
                    cumsum_probs = cumsum(probabilities);
                    rand_val = rand();
                    action_vec = find(cumsum_probs >= rand_val, 1);
                    if isempty(action_vec)
                        action_vec = 1;
                    end
                else
                    % Epsilon-贪婪选择
                    if rand() < obj.epsilon
                        % 探索：随机选择站点
                        action_vec = randi(n_stations);
                    else
                        % 利用：选择Q值最高的站点
                        valid_q_values = q_values(1:min(n_stations, length(q_values)));
                        [~, action_vec] = max(valid_q_values);
                    end
                end
                
                % 确保攻击者动作在有效范围内
                action_vec = max(1, min(n_stations, round(action_vec)));
                
            else
                % ===== 防御者：返回资源分配向量 =====
                
                % 确定站点数量
                if isprop(obj, 'config') && isfield(obj.config, 'n_stations')
                    n_stations = obj.config.n_stations;
                else
                    n_stations = min(obj.action_dim, 10);
                end
                
                % 生成资源分配向量
                action_vec = zeros(1, n_stations);
                
                % ===== 修复：使用 exploration_strategy 判断 =====
                if strcmp(obj.exploration_strategy, 'softmax')
                    % Softmax策略选择
                    temperature = max(0.1, obj.temperature);
                    q_normalized = q_values - max(q_values);
                    exp_values = exp(q_normalized / temperature);
                    probabilities = exp_values / sum(exp_values);
                    
                    % 转换为站点级资源分配
                    for i = 1:n_stations
                        q_start = (i-1) * obj.action_dim / n_stations + 1;
                        q_end = i * obj.action_dim / n_stations;
                        q_start = max(1, round(q_start));
                        q_end = min(obj.action_dim, round(q_end));
                        
                        if q_start <= q_end
                            station_probs = probabilities(q_start:q_end);
                            action_vec(i) = sum(station_probs);
                        end
                    end
                    
                    % 归一化
                    action_vec = action_vec / max(sum(action_vec), 1e-6);
                    
                else
                    % Epsilon-贪婪策略选择
                    if rand() < obj.epsilon
                        % 探索：随机分配资源
                        action_vec = rand(1, n_stations);
                        action_vec = action_vec / sum(action_vec);
                    else
                        % 利用：基于Q值分配资源
                        for i = 1:n_stations
                            q_start = (i-1) * obj.action_dim / n_stations + 1;
                            q_end = i * obj.action_dim / n_stations;
                            q_start = max(1, round(q_start));
                            q_end = min(obj.action_dim, round(q_end));
                            
                            if q_start <= q_end
                                action_vec(i) = mean(q_values(q_start:q_end));
                            end
                        end
                        
                        % 将Q值转换为资源分配概率
                        action_vec = action_vec - min(action_vec) + 0.1;
                        action_vec = action_vec / sum(action_vec);
                    end
                end
            end
        end
        
        function update(obj, state_vec, action, reward, next_state_vec, next_action)
            % Double Q-Learning算法更新
            
            % 健壮性检查
            if isempty(state_vec) || isempty(next_state_vec)
                return;
            end
            
            % 获取状态索引
            state_idx = obj.encodeState(mean(state_vec));
            next_state_idx = obj.encodeState(mean(next_state_vec));
            
            % 处理动作索引
            if isvector(action) && length(action) > 1
                action_idx = obj.encodeAction(action);
            else
                action_idx = round(action);
            end
            
            % 边界检查
            action_idx = max(1, min(obj.action_dim, action_idx));
            
            % 更新访问计数
            obj.visit_count(state_idx, action_idx) = obj.visit_count(state_idx, action_idx) + 1;
            
            % 动态学习率
            visit_count = obj.visit_count(state_idx, action_idx);
            adaptive_lr = obj.lr_scheduler.current_lr / (1 + visit_count * 0.01);
            
            % Double Q-Learning更新：随机选择更新哪个Q表
            if rand() < 0.5
                % 更新Q_table_A，使用Q_table_B来选择动作
                [~, best_action] = max(obj.Q_table_A(next_state_idx, :));
                best_action = max(1, min(best_action, size(obj.Q_table_B, 2)));
                target = reward + obj.discount_factor * obj.Q_table_B(next_state_idx, best_action);
                td_error = target - obj.Q_table_A(state_idx, action_idx);
                obj.Q_table_A(state_idx, action_idx) = obj.Q_table_A(state_idx, action_idx) + adaptive_lr * td_error;
            else
                % 更新Q_table_B，使用Q_table_A来选择动作
                [~, best_action] = max(obj.Q_table_B(next_state_idx, :));
                best_action = max(1, min(best_action, size(obj.Q_table_A, 2)));
                target = reward + obj.discount_factor * obj.Q_table_A(next_state_idx, best_action);
                td_error = target - obj.Q_table_B(state_idx, action_idx);
                obj.Q_table_B(state_idx, action_idx) = obj.Q_table_B(state_idx, action_idx) + adaptive_lr * td_error;
            end
            
            % 更新兼容性Q表
            obj.updateQTableProperty();
            
            % 更新学习率
            obj.updateLearningRate();
            
            % 增加更新计数
            obj.update_count = obj.update_count + 1;
            obj.total_reward = obj.total_reward + reward;
            
            % 记录性能历史
            obj.recordPerformance(reward, td_error);
        end
        
        function updateQTableProperty(obj)
            % 更新兼容性Q_table属性
            try
                if ~isempty(obj.Q_table_A) && ~isempty(obj.Q_table_B)
                    obj.Q_table = (obj.Q_table_A + obj.Q_table_B) / 2;
                elseif ~isempty(obj.Q_table_A)
                    obj.Q_table = obj.Q_table_A;
                elseif ~isempty(obj.Q_table_B)
                    obj.Q_table = obj.Q_table_B;
                else
                    obj.Q_table = zeros(obj.state_dim, obj.action_dim);
                end
            catch
                obj.Q_table = zeros(obj.state_dim, obj.action_dim);
            end
        end
        
        function state_idx = encodeState(obj, state_scalar)
            % 状态编码方法
            try
                if isempty(state_scalar) || ~isnumeric(state_scalar)
                    state_idx = 1;
                    return;
                end
                state_scalar = double(state_scalar);
                if isnan(state_scalar) || isinf(state_scalar)
                    state_scalar = 0;
                end
                state_idx = max(1, min(obj.state_dim, round(state_scalar * obj.state_dim)));
                if state_idx <= 0
                    state_idx = 1;
                end
            catch
                state_idx = 1;
            end
        end
        
        function action_idx = encodeAction(obj, action_vec)
            % 动作编码方法（将动作向量转换为索引）
            if length(action_vec) == 1
                action_idx = round(action_vec);
            else
                % 使用加权求和方式编码
                weights = (1:length(action_vec)) / length(action_vec);
                action_idx = round(sum(action_vec .* weights) * obj.action_dim);
            end
            action_idx = max(1, min(obj.action_dim, action_idx));
        end
        
        function updateLearningRate(obj)
            % 更新学习率
            obj.lr_scheduler.step_count = obj.lr_scheduler.step_count + 1;
            
            if mod(obj.lr_scheduler.step_count, obj.lr_scheduler.decay_steps) == 0
                obj.lr_scheduler.current_lr = max(obj.lr_scheduler.min_lr, ...
                    obj.lr_scheduler.current_lr * obj.lr_scheduler.decay_rate);
            end
        end
        
        function recordPerformance(obj, reward, td_error)
            %% 记录性能数据 - 修复版
            if mod(obj.update_count, 100) == 0
                % 初始化 parameter_history 字段
                if ~isfield(obj.parameter_history, 'learning_rate')
                    obj.parameter_history.learning_rate = [];
                end
                if ~isfield(obj.parameter_history, 'epsilon')
                    obj.parameter_history.epsilon = [];
                end
                if ~isfield(obj.parameter_history, 'q_values')
                    obj.parameter_history.q_values = [];
                end
                
                % 初始化 performance_history 字段
                if ~isfield(obj.performance_history, 'reward_100')
                    obj.performance_history.reward_100 = [];
                end
                if ~isfield(obj.performance_history, 'td_error_100')
                    obj.performance_history.td_error_100 = [];
                end
                
                % 安全地记录数据
                try
                    obj.parameter_history.learning_rate(end+1) = obj.lr_scheduler.current_lr;
                    obj.parameter_history.epsilon(end+1) = obj.epsilon;
                    obj.parameter_history.q_values(end+1) = mean(obj.Q_table(:));
                    
                    obj.performance_history.reward_100(end+1) = obj.total_reward / max(1, obj.update_count);
                    obj.performance_history.td_error_100(end+1) = abs(td_error);
                catch ME
                    fprintf('[WARNING] QLearningAgent recordPerformance 出错: %s\n', ME.message);
                end
            end
        end        
        function policy = getPolicy(obj)
            % 获取当前策略
            try
                % 更新Q_table属性
                obj.updateQTableProperty();
                
                % 检查Q表是否为空
                if isempty(obj.Q_table) || size(obj.Q_table, 1) == 0
                    policy = ones(1, obj.action_dim) / obj.action_dim;
                    return;
                end
                
                % 基于平均Q值的策略
                avg_q_values = mean(obj.Q_table, 1);
                
                % ===== 修复：使用 exploration_strategy 判断 =====
                if strcmp(obj.exploration_strategy, 'softmax')
                    % Softmax策略
                    temperature = max(0.1, obj.temperature);
                    exp_values = exp(avg_q_values / temperature);
                    policy = exp_values / sum(exp_values);
                else
                    % Epsilon-贪婪策略
                    policy = ones(1, obj.action_dim) * obj.epsilon / obj.action_dim;
                    [~, best_action] = max(avg_q_values);
                    policy(best_action) = policy(best_action) + (1 - obj.epsilon);
                end
                
            catch ME
                warning('DoubleQLearningAgent.getPolicy 出错: %s', ME.message);
                policy = ones(1, obj.action_dim) / obj.action_dim;
            end
        end
        
        function strategy = getStrategy(obj)
            % 获取策略（与getPolicy相同）
            strategy = obj.getPolicy();
        end
        
        function stats = getStatistics(obj)
            % 获取详细统计信息
            try
                stats = struct();
                stats.name = obj.name;
                stats.agent_type = obj.agent_type;
                stats.update_count = obj.update_count;
                stats.total_reward = obj.total_reward;
                
                % Q值统计
                if ~isempty(obj.Q_table)
                    stats.avg_q_value = mean(obj.Q_table(:));
                    stats.max_q_value = max(obj.Q_table(:));
                    stats.min_q_value = min(obj.Q_table(:));
                    stats.q_value_std = std(obj.Q_table(:));
                else
                    stats.avg_q_value = 0;
                    stats.max_q_value = 0;
                    stats.min_q_value = 0;
                    stats.q_value_std = 0;
                end
                
                % Double Q特有统计
                if ~isempty(obj.Q_table_A) && ~isempty(obj.Q_table_B)
                    stats.q1_avg = mean(obj.Q_table_A(:));
                    stats.q2_avg = mean(obj.Q_table_B(:));
                    stats.q_difference = mean(abs(obj.Q_table_A(:) - obj.Q_table_B(:)));
                end
                
                % 学习参数
                stats.current_learning_rate = obj.learning_rate;
                stats.current_epsilon = obj.epsilon;
                
                % 探索统计
                if ~isempty(obj.visit_count)
                    stats.total_state_visits = sum(obj.visit_count(:));
                    stats.explored_states = sum(sum(obj.visit_count > 0));
                    stats.exploration_ratio = stats.explored_states / numel(obj.visit_count);
                else
                    stats.total_state_visits = 0;
                    stats.explored_states = 0;
                    stats.exploration_ratio = 0;
                end
                
            catch ME
                warning(ME.identifier, 'DoubleQLearningAgent.getStatistics 出错: %s', ME.message);
                stats = struct('name', obj.name, 'agent_type', obj.agent_type, 'update_count', 0);
            end
        end
        
        function reset(obj)
            % 重置智能体状态
            obj.update_count = 0;
            obj.total_reward = 0;
            obj.strategy_history = [];
            obj.performance_history = struct();
            obj.parameter_history = struct();
            obj.parameter_history.learning_rate = [];
            obj.parameter_history.epsilon = [];
            obj.parameter_history.q_values = [];
        end
    end
end