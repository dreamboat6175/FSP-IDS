classdef SARSAAgent < RLAgent
    properties
        Q_table
        visit_count
        lr_scheduler
        strategy_history
        performance_history
        parameter_history
    end
    
    methods
        function obj = SARSAAgent(name, agent_type, config, state_dim, action_dim)
            % 修正：确保 agent_type 参数被正确接收并传递给基类
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 改进的Q表初始化 - 使用乐观初始化
            initial_value = 1.0; % 提高初始值
            noise_level = 0.1;   % 增加噪声
            obj.Q_table = ones(state_dim, action_dim) * initial_value + ...
                          randn(state_dim, action_dim) * noise_level;
            obj.visit_count = zeros(state_dim, action_dim);
            
            % ===== 重要修复：移除 use_softmax 属性定义 =====
            % 现在使用基类的 exploration_strategy 属性
            % obj.use_softmax = false; % 删除这行，改用基类属性
            
            % 初始化学习率调度器
            obj.lr_scheduler = struct();
            if isfield(config, 'learning_rate')
                obj.lr_scheduler.initial_lr = config.learning_rate;
                obj.lr_scheduler.current_lr = config.learning_rate;
            else
                obj.lr_scheduler.initial_lr = 0.15;
                obj.lr_scheduler.current_lr = 0.15;
            end
            obj.lr_scheduler.min_lr = 0.05;           % 提高最小学习率
            obj.lr_scheduler.decay_steps = 2000;      % 增加衰减间隔  
            obj.lr_scheduler.step_count = 0;
            obj.lr_scheduler.decay_rate = 0.995;      % 减缓衰减率
            
            % 确保基类属性有默认值
            obj.strategy_history = [];
            obj.performance_history = struct();
            obj.parameter_history = struct();
            obj.parameter_history.learning_rate = [];
            obj.parameter_history.epsilon = [];
            obj.parameter_history.q_values = [];
        end
        
        function action_vec = selectAction(obj, state_vec)
            % SARSA智能体的动作选择
            
            % 健壮性检查
            if isempty(state_vec)
                warning('SARSAAgent.selectAction: state_vec is empty, auto-fixing...');
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
            % SARSA算法更新（使用下一个动作）
            
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
            
            % 处理下一个动作索引
            if nargin > 5 && ~isempty(next_action)
                if isvector(next_action) && length(next_action) > 1
                    next_action_idx = obj.encodeAction(next_action);
                else
                    next_action_idx = round(next_action);
                end
            else
                % 如果没有提供下一个动作，使用当前策略选择
                next_q_values = obj.Q_table(next_state_idx, :);
                % ===== 修复：使用 exploration_strategy 判断 =====
                if strcmp(obj.exploration_strategy, 'softmax')
                    temperature = max(0.1, obj.temperature);
                    exp_values = exp(next_q_values / temperature);
                    probabilities = exp_values / sum(exp_values);
                    next_action_idx = randsample(1:length(probabilities), 1, true, probabilities);
                else
                    if rand() < obj.epsilon
                        next_action_idx = randi(obj.action_dim);
                    else
                        [~, next_action_idx] = max(next_q_values);
                    end
                end
            end
            
            % 边界检查
            action_idx = max(1, min(obj.action_dim, action_idx));
            next_action_idx = max(1, min(obj.action_dim, next_action_idx));
            
            % 获取当前Q值
            current_q = obj.Q_table(state_idx, action_idx);
            
            % 获取下一状态下一动作的Q值（SARSA的核心特点）
            next_q = obj.Q_table(next_state_idx, next_action_idx);
            
            % 更新访问计数
            obj.visit_count(state_idx, action_idx) = obj.visit_count(state_idx, action_idx) + 1;
            
            % 动态学习率
            visit_count = obj.visit_count(state_idx, action_idx);
            adaptive_lr = obj.lr_scheduler.current_lr / (1 + visit_count * 0.001);
            
            % SARSA更新公式
            td_error = reward + obj.discount_factor * next_q - current_q;
            obj.Q_table(state_idx, action_idx) = current_q + adaptive_lr * td_error;
            
            % 更新学习率
            obj.updateLearningRate();
            
            % 增加更新计数
            obj.update_count = obj.update_count + 1;
            obj.total_reward = obj.total_reward + reward;
            
            % 记录性能历史
            obj.recordPerformance(reward, td_error);
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
            obj.lr_scheduler.step_count = obj.lr_scheduler.step_count + 1;
            
            % 只有在更新次数较多且性能稳定时才衰减
            if mod(obj.lr_scheduler.step_count, obj.lr_scheduler.decay_steps) == 0 && ...
               obj.update_count > 1000
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
            % 获取当前策略分布
            % 计算所有状态的平均Q值
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
