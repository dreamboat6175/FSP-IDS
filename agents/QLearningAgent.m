classdef QLearningAgent < RLAgent
    properties
        Q_table          % Q值表
        visit_count      % 状态-动作访问计数
        lr_scheduler     % 学习率调度器
        strategy_history     % 策略历史记录
        performance_history  % 性能历史记录
        parameter_history    % 参数历史记录
    end
    
    methods
        function obj = QLearningAgent(name, agent_type, config, state_dim, action_dim)
            % 构造函数
            % 修正：确保 agent_type 参数被正确接收并传递给基类
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 初始化Q表和访问计数
            if state_dim * action_dim > 1e6
                obj.Q_table = sparse(state_dim, action_dim);
                obj.visit_count = sparse(state_dim, action_dim);
            else
                obj.Q_table = zeros(state_dim, action_dim);
                obj.visit_count = zeros(state_dim, action_dim);
            end
            
            % 乐观初始化
            initial_value = 5.0;
            noise_level = 0.5;
            
            if issparse(obj.Q_table)
                [rows, cols] = size(obj.Q_table);
                init_indices = randi([1, rows*cols], [1, min(1000, rows*cols/10)]);
                obj.Q_table(init_indices) = initial_value + randn(size(init_indices)) * noise_level;
            else
                obj.Q_table = ones(state_dim, action_dim) * initial_value + ...
                              randn(state_dim, action_dim) * noise_level;
            end
            
            % 初始化学习率调度器
            obj.lr_scheduler = struct();
            if isfield(config, 'learning_rate')
                obj.lr_scheduler.initial_lr = config.learning_rate;
            else
                obj.lr_scheduler.initial_lr = 0.15;
                % 移除警告，改为调试信息
                if ~exist('QLearningAgent_warning_shown', 'var')
                    fprintf('[DEBUG] QLearningAgent: 配置中未找到learning_rate，使用默认值 0.15\n');
                    global QLearningAgent_warning_shown;
                    QLearningAgent_warning_shown = true;
                end
            end
            obj.lr_scheduler.min_lr = 0.001;
            obj.lr_scheduler.decay_steps = 1000;
            obj.lr_scheduler.current_lr = obj.lr_scheduler.initial_lr;
            obj.lr_scheduler.step_count = 0;
            obj.lr_scheduler.decay_rate = 0.99;
            
            % 现在使用基类的 exploration_strategy 属性
            
            % 确保基类属性有默认值
            if isempty(obj.epsilon)
                obj.epsilon = 0.9;
            end
            if isempty(obj.epsilon_min)
                obj.epsilon_min = 0.01;
            end
            if isempty(obj.epsilon_decay)
                obj.epsilon_decay = 0.995;
            end
            if isempty(obj.temperature)
                obj.temperature = 1.0;
            end
            if isempty(obj.temperature_decay)
                obj.temperature_decay = 0.995;
            end
            if isempty(obj.learning_rate_min)
                obj.learning_rate_min = 0.01;
            end
            if isempty(obj.learning_rate_decay)
                obj.learning_rate_decay = 0.9995;
            end
            obj.strategy_history = [];
            obj.performance_history = struct();
            obj.parameter_history = struct();
            obj.parameter_history.learning_rate = [];
            obj.parameter_history.epsilon = [];
            obj.parameter_history.q_values = [];
        end
        
        function state_idx = getStateIndex(obj, state)
            % getStateIndex方法 - 为了向后兼容
            state_idx = obj.encodeState(state);
        end
        
        function action = selectAction(obj, state_vec)
            % 智能体动作选择方法（支持防御者和攻击者）
            
            % 健壮性检查
            if isempty(state_vec)
                state_vec = ones(1, obj.state_dim);
            end
            state_vec = reshape(state_vec, 1, []);
            
            % 获取状态索引
            state_idx = obj.encodeState(mean(state_vec));
            
            % 获取Q值
            q_values = obj.Q_table(state_idx, :);
            
            % 确保Q值有效
            if any(isnan(q_values)) || any(isinf(q_values))
                q_values = randn(size(q_values)) * 0.1;
            end
            
            % 动态调整参数
            if obj.epsilon_decay < 1
                obj.epsilon = max(obj.epsilon_min, obj.epsilon * obj.epsilon_decay);
            end
            
            % 使用 exploration_strategy 而不是 use_softmax
            if strcmp(obj.exploration_strategy, 'softmax') && obj.temperature_decay < 1
                obj.temperature = max(0.1, obj.temperature * obj.temperature_decay);
            end
            
            % 区分防御者和攻击者的动作生成
            if contains(obj.agent_type, 'attacker') || contains(obj.name, 'attacker')
                % 攻击者：返回单个站点索引
                
                % 确定站点数量 - 优先使用传入的config
                if isfield(obj.config, 'n_stations')
                    n_stations = obj.config.n_stations;
                elseif obj.action_dim > 0
                    n_stations = obj.action_dim;
                else
                    error('无法确定站点数量：config.n_stations和action_dim都不可用');
                end
                
                % 使用 exploration_strategy 判断
                if strcmp(obj.exploration_strategy, 'softmax')
                    % Softmax选择
                    temperature = max(0.1, obj.temperature);
                    q_normalized = q_values(1:min(n_stations, length(q_values))) - max(q_values(1:min(n_stations, length(q_values))));
                    exp_values = exp(q_normalized / temperature);
                    probabilities = exp_values / sum(exp_values);
                    
                    % 基于概率选择动作
                    cumsum_probs = cumsum(probabilities);
                    rand_val = rand();
                    action = find(cumsum_probs >= rand_val, 1);
                    if isempty(action)
                        action = 1; % Fallback if no action is chosen (shouldn't happen with proper probabilities)
                    end
                else
                    % Epsilon-贪婪选择
                    if rand() < obj.epsilon
                        % 探索：随机选择站点
                        action = randi(n_stations);
                    else
                        % 利用：选择Q值最高的站点
                        valid_q_values = q_values(1:min(n_stations, length(q_values)));
                        [~, action] = max(valid_q_values);
                    end
                end
                
                % 确保攻击者动作在有效范围内
                action = max(1, min(n_stations, round(action)));
                
                % 调试信息 - 每100步或前5步打印
                if mod(obj.update_count, 100) == 0 || obj.update_count < 5
                    fprintf('[QLearningAgent] 攻击者 %s (更新次数 %d): 选择目标站点=%d, 站点数=%d\n', ...
                            obj.name, obj.update_count, action, n_stations);
                end
                
            else
                % 防御者：返回资源分配向量
                
                % 确定站点数量
                if isprop(obj, 'config') && isfield(obj.config, 'n_stations')
                    n_stations = obj.config.n_stations;
                else
                    n_stations = min(obj.action_dim, 10); % Default to 10 if not specified
                end
                
                % 生成资源分配向量
                action = zeros(1, n_stations);
                
                % 使用 exploration_strategy 判断
                if strcmp(obj.exploration_strategy, 'softmax')
                    % Softmax策略选择
                    temperature = max(0.1, obj.temperature);
                    q_normalized = q_values - max(q_values);
                    exp_values = exp(q_normalized / temperature);
                    probabilities = exp_values / sum(exp_values);
                    
                    % 转换为站点级资源分配
                    for i = 1:n_stations
                        % Distribute Q-values across the number of stations
                        % This is a simplified approach, a more sophisticated approach
                        % would involve discretizing the action space for resource allocation.
                        q_start = (i-1) * obj.action_dim / n_stations + 1;
                        q_end = i * obj.action_dim / n_stations;
                        q_start = max(1, round(q_start));
                        q_end = min(obj.action_dim, round(q_end));
                        
                        if q_start <= q_end
                            station_probs = probabilities(q_start:q_end);
                            action(i) = sum(station_probs);
                        end
                    end
                    
                    % 归一化
                    action = action / max(sum(action), 1e-6); % Avoid division by zero
                    
                else
                    % Epsilon-贪婪策略选择
                    if rand() < obj.epsilon
                        % 探索：随机分配资源
                        action = rand(1, n_stations);
                        action = action / sum(action);
                    else
                        % 利用：基于Q值分配资源
                        % This is a heuristic to convert Q-values to resource allocation.
                        % A more precise method would involve discretizing resource allocations
                        % into distinct actions and learning Q-values for those.
                        for i = 1:n_stations
                            q_start = (i-1) * obj.action_dim / n_stations + 1;
                            q_end = i * obj.action_dim / n_stations;
                            q_start = max(1, round(q_start));
                            q_end = min(obj.action_dim, round(q_end));
                            
                            if q_start <= q_end
                                action(i) = mean(q_values(q_start:q_end));
                            end
                        end
                        
                        % 将Q值转换为资源分配概率
                        action = action - min(action) + 0.1; % Shift to positive and add a small base
                        action = action / sum(action); % Normalize to sum to 1
                    end
                end
                
                % 调试信息 - 每100步或前5步打印
                if mod(obj.update_count, 100) == 0 || obj.update_count < 5
                    fprintf('[QLearningAgent] 防御者 %s (更新次数 %d): 资源分配=%s\n', ...
                            obj.name, obj.update_count, mat2str(action, 3));
                end
            end
        end
        
        function update(obj, state_vec, action, reward, next_state_vec, next_action)
            % Q-Learning算法更新 - 增强调试版本
            
            % === 调试：确认进入更新方法 ===
            fprintf('[QLearningAgent] %s: 进入更新方法 (更新次数 %d)\n', obj.name, obj.update_count);
            
            % 健壮性检查
            if isempty(state_vec) || isempty(next_state_vec)
                fprintf('[QLearningAgent] %s: 状态向量为空，跳过更新。\n', obj.name);
                fprintf('[DEBUG] state_vec: %s, next_state_vec: %s\n', mat2str(state_vec), mat2str(next_state_vec));
                return;
            end
            
            % 验证输入参数
            if isnan(reward) || isinf(reward)
                fprintf('[QLearningAgent] %s: 奖励值无效 (%.3f)，设为0\n', obj.name, reward);
                reward = 0;
            end
            
            % 获取状态索引
            state_idx = obj.encodeState(mean(state_vec));
            next_state_idx = obj.encodeState(mean(next_state_vec));
            
            fprintf('[DEBUG] 状态编码: state_idx=%d, next_state_idx=%d\n', state_idx, next_state_idx);
            
            % 处理动作索引（防御者和攻击者可能不同）
            if isvector(action) && length(action) > 1
                % 防御者：资源分配向量转换为索引
                action_idx = obj.encodeAction(action);
                fprintf('[DEBUG] 防御者动作编码: action_vec=%s -> action_idx=%d\n', mat2str(action, 2), action_idx);
            else
                % 攻击者：单一目标索引
                action_idx = round(action);
                fprintf('[DEBUG] 攻击者动作编码: action=%d -> action_idx=%d\n', action, action_idx);
            end
            
            % 边界检查
            action_idx = max(1, min(obj.action_dim, action_idx));
            
            % 验证Q表大小
            if size(obj.Q_table, 1) < state_idx || size(obj.Q_table, 2) < action_idx
                fprintf('[ERROR] Q表大小不足: Q_table大小=%s, 需要访问(%d,%d)\n', ...
                        mat2str(size(obj.Q_table)), state_idx, action_idx);
                return;
            end
            
            % 获取当前Q值
            current_q = obj.Q_table(state_idx, action_idx);
            
            % 获取下一状态的最大Q值（Q-Learning的核心）
            next_q_values = obj.Q_table(next_state_idx, :);
            max_next_q = max(next_q_values);
            
            % 更新访问计数
            obj.visit_count(state_idx, action_idx) = obj.visit_count(state_idx, action_idx) + 1;
            
            % 动态学习率
            visit_count = obj.visit_count(state_idx, action_idx);
            adaptive_lr = obj.lr_scheduler.current_lr / (1 + visit_count * 0.01);
            
            % Q值更新
            td_error = reward + obj.discount_factor * max_next_q - current_q;
            new_q_value = current_q + adaptive_lr * td_error;
            obj.Q_table(state_idx, action_idx) = new_q_value;
            
            % 调试：打印TD误差和新Q值
            fprintf('[QLearningAgent] %s (更新次数 %d): 状态=%d, 动作=%d, 奖励=%.2f, TD误差=%.4f, 旧Q=%.4f, 新Q=%.4f\n', ...
                    obj.name, obj.update_count, state_idx, action_idx, reward, td_error, current_q, new_q_value);
        
            % 更新学习率
            obj.updateLearningRate();
            
            % 增加更新计数
            obj.update_count = obj.update_count + 1;
            obj.total_reward = obj.total_reward + reward;
            
            % 记录性能历史
            obj.recordPerformance(reward, td_error);
            
            fprintf('[QLearningAgent] %s: 更新完成，总更新次数=%d\n', obj.name, obj.update_count);
        end
        
        function state_idx = encodeState(obj, state_scalar)
            % 状态编码方法
            state_idx = max(1, min(obj.state_dim, round(state_scalar * obj.state_dim)));
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
            % 记录性能数据
            if mod(obj.update_count, 100) == 0
                obj.parameter_history.learning_rate(end+1) = obj.lr_scheduler.current_lr;
                obj.parameter_history.epsilon(end+1) = obj.epsilon;
                obj.parameter_history.q_values(end+1) = mean(obj.Q_table(:));
                
                obj.performance_history.reward_100(end+1) = obj.total_reward / max(1, obj.update_count);
                obj.performance_history.td_error_100(end+1) = abs(td_error);
            end
        end
        
        function policy = getPolicy(obj)
            % 获取当前策略分布
            % 简化版本：返回每个动作的选择概率
            
            % 计算所有状态的平均Q值
            avg_q_values = mean(obj.Q_table, 1);
            
            % 使用 exploration_strategy 判断
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
        end
        
        function strategy = getStrategy(obj)
            % 获取策略（与getPolicy相同）
            strategy = obj.getPolicy();
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
