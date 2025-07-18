%% QLearningAgent.m - Q-Learning智能体实现（完整修复版）
% =========================================================================
% 描述: 实现标准Q-Learning算法的智能体
% 修复了所有缺失的方法和属性
% =========================================================================

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
            
            % 初始化新添加的属性
            % obj.use_softmax = false;     % 默认使用epsilon-greedy
            
            % 确保基类属性有默认值
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
            if obj.use_softmax && obj.temperature_decay < 1
                obj.temperature = max(0.1, obj.temperature * obj.temperature_decay);
            end
            
            % ===== 关键修复：区分防御者和攻击者的动作生成 =====
            if contains(obj.agent_type, 'attacker') || contains(obj.name, 'attacker')
                % ===== 攻击者：返回单个站点索引 =====
                
                % 确定站点数量
                if isprop(obj, 'config') && isfield(obj.config, 'n_stations')
                    n_stations = obj.config.n_stations;
                else
                    n_stations = min(obj.action_dim, 10);  % 假设动作维度不会超过10个站点
                end
                
                if obj.use_softmax
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
                        action = 1;
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
                
                % 调试信息
                if obj.update_count <= 2
                    fprintf('[QLearningAgent] 攻击者 %s: 选择目标站点=%d, 站点数=%d\n', ...
                            obj.name, action, n_stations);
                end
                
            else
                % ===== 防御者：返回资源分配向量 =====
                
                % 确定站点数量
                if isprop(obj, 'config') && isfield(obj.config, 'n_stations')
                    n_stations = obj.config.n_stations;
                else
                    n_stations = min(obj.action_dim, 10);
                end
                
                % 生成资源分配向量
                action = zeros(1, n_stations);
                
                if obj.use_softmax
                    % Softmax策略选择
                    temperature = max(0.1, obj.temperature);
                    q_normalized = q_values - max(q_values);
                    exp_values = exp(q_normalized / temperature);
                    probabilities = exp_values / sum(exp_values);
                    
                    % 转换为站点级资源分配
                    if length(probabilities) >= n_stations
                        action = probabilities(1:n_stations);
                    else
                        action = ones(1, n_stations) / n_stations;
                    end
                    
                    % 归一化到总资源
                    total_resources = 100;  % 可以从config获取
                    action = action / sum(action) * total_resources;
                    
                else
                    % Epsilon-贪婪策略
                    if rand() < obj.epsilon
                        % 探索：随机分配资源
                        action = rand(1, n_stations);
                        action = action / sum(action) * 100;
                    else
                        % 利用：基于Q值分配资源
                        if length(q_values) >= n_stations
                            % 选择Q值最高的动作对应的资源分配
                            [~, sorted_indices] = sort(q_values, 'descend');
                            
                            % 给Q值高的位置分配更多资源
                            base_allocation = 100 / n_stations;
                            for i = 1:n_stations
                                if i <= length(sorted_indices)
                                    bonus = 0.1 * (n_stations - i + 1) / n_stations;
                                    action(i) = base_allocation * (1 + bonus);
                                else
                                    action(i) = base_allocation;
                                end
                            end
                            
                            % 重新归一化
                            action = action / sum(action) * 100;
                        else
                            % 如果Q值不够，均匀分配
                            action = ones(1, n_stations) * (100 / n_stations);
                        end
                    end
                end
                
                % 确保防御者动作向量有效
                action = max(action, 0.1);  % 最小值
                action = real(action);      % 确保是实数
                
                % 调试信息
                if obj.update_count <= 2
                    fprintf('[QLearningAgent] 防御者 %s: 动作向量长度=%d, 站点数=%d, 总和=%.2f\n', ...
                            obj.name, length(action), n_stations, sum(action));
                end
            end
            if strcmp(obj.agent_type, 'defender') && length(action) > 1
                obj.strategy_history(end+1, :) = action;
            end
            
            % 记录参数历史
            obj.parameter_history.learning_rate(end+1) = obj.learning_rate;
            obj.parameter_history.epsilon(end+1) = obj.epsilon;
            obj.parameter_history.q_values(end+1) = mean(obj.Q_table(:));
        end

        
        function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
            % 更新Q表
            
            % 健壮性检查
            if isempty(action_vec)
                action_vec = ones(1, 5);
            end
            action_vec = reshape(action_vec, 1, []);
            if isempty(state_vec)
                state_vec = ones(1, obj.state_dim);
            end
            state_vec = reshape(state_vec, 1, []);
            if ~isempty(next_state_vec)
                next_state_vec = reshape(next_state_vec, 1, []);
            else
                next_state_vec = state_vec;  % 如果没有下一状态，使用当前状态
            end
            
            % 获取状态索引
            state_idx = obj.encodeState(mean(state_vec));
            next_state_idx = obj.encodeState(mean(next_state_vec));
            
            % 简化的动作索引映射
            n_stations = length(action_vec);
            if n_stations > 0 && obj.action_dim >= n_stations
                % 使用第一个站点的动作作为主要索引
                primary_action = max(1, min(obj.action_dim, round(action_vec(1))));
            else
                primary_action = 1;
            end
            
            % Q-Learning更新
            current_q = obj.Q_table(state_idx, primary_action);
            max_next_q = max(obj.Q_table(next_state_idx, :));
            
            td_error = reward + obj.discount_factor * max_next_q - current_q;
            obj.Q_table(state_idx, primary_action) = current_q + obj.learning_rate * td_error;
            obj.visit_count(state_idx, primary_action) = obj.visit_count(state_idx, primary_action) + 1;
            
            % 更新计数器
            obj.update_count = obj.update_count + 1;
            obj.recordPerformance(reward, td_error);
        end
        
        function stats = getStatistics(obj)
            % 获取智能体统计信息
            stats = struct();
            
            % 基本统计
            stats.name = obj.name;
            stats.agent_type = obj.agent_type;
            
            % 检查属性是否存在
            if isprop(obj, 'update_count') || isfield(obj, 'update_count')
                stats.update_count = obj.update_count;
            else
                stats.update_count = 0;
            end
            
            if isprop(obj, 'total_reward') || isfield(obj, 'total_reward')
                stats.total_reward = obj.total_reward;
            else
                stats.total_reward = 0;
            end
            
            % Q表统计
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
            
            % 学习参数
            if isfield(obj.lr_scheduler, 'current_lr')
                stats.current_learning_rate = obj.lr_scheduler.current_lr;
            elseif isprop(obj, 'learning_rate') || isfield(obj, 'learning_rate')
                stats.current_learning_rate = obj.learning_rate;
            else
                stats.current_learning_rate = 0.1;
            end
            
            if isprop(obj, 'epsilon') || isfield(obj, 'epsilon')
                stats.current_epsilon = obj.epsilon;
            else
                stats.current_epsilon = 0.1;
            end
            
            % 探索统计
            if ~isempty(obj.visit_count)
                total_visits = sum(obj.visit_count(:));
                stats.total_state_visits = total_visits;
                stats.explored_states = sum(sum(obj.visit_count > 0));
                stats.exploration_ratio = stats.explored_states / numel(obj.visit_count);
            else
                stats.total_state_visits = 0;
                stats.explored_states = 0;
                stats.exploration_ratio = 0;
            end
            
            % 性能统计
            if isprop(obj, 'episode_rewards') || isfield(obj, 'episode_rewards')
                if ~isempty(obj.episode_rewards)
                    stats.avg_episode_reward = mean(obj.episode_rewards);
                    stats.best_episode_reward = max(obj.episode_rewards);
                    stats.worst_episode_reward = min(obj.episode_rewards);
                    stats.total_episodes = length(obj.episode_rewards);
                else
                    stats.avg_episode_reward = 0;
                    stats.best_episode_reward = 0;
                    stats.worst_episode_reward = 0;
                    stats.total_episodes = 0;
                end
            else
                stats.avg_episode_reward = 0;
                stats.best_episode_reward = 0;
                stats.worst_episode_reward = 0;
                stats.total_episodes = 0;
            end
        end
        
        function policy = getPolicy(obj)
            % 获取当前策略
            
            if isempty(obj.Q_table) || size(obj.Q_table, 1) == 0
                % 如果Q表为空，返回均匀策略
                policy = ones(1, obj.action_dim) / obj.action_dim;
                return;
            end
            
            % 基于平均Q值的策略
            avg_q_values = mean(obj.Q_table, 1);
            
            if all(avg_q_values == 0) || all(isnan(avg_q_values))
                % 如果所有Q值都是0或NaN，返回均匀策略
                policy = ones(1, obj.action_dim) / obj.action_dim;
            else
                % 使用softmax转换为概率分布
                temperature = 1.0;
                if isprop(obj, 'temperature') || isfield(obj, 'temperature')
                    temperature = obj.temperature;
                end
                
                if temperature > 0
                    scaled_q = avg_q_values / temperature;
                    % 数值稳定的softmax
                    exp_q = exp(scaled_q - max(scaled_q));
                    policy = exp_q / sum(exp_q);
                else
                    % 贪婪策略
                    policy = zeros(1, obj.action_dim);
                    [~, best_action] = max(avg_q_values);
                    policy(best_action) = 1;
                end
            end
            
            % 确保policy是有效的概率分布
            if any(isnan(policy)) || any(isinf(policy)) || sum(policy) == 0
                policy = ones(1, obj.action_dim) / obj.action_dim;
            else
                policy = policy / sum(policy); % 归一化
            end
        end
        
        function strategy = getStrategy(obj)
            % 获取当前策略分布（与getPolicy相同）
            strategy = obj.getPolicy();
        end
        
        function resetEpisode(obj)
            % 重置episode相关的状态
            
            % 更新探索率
            if isprop(obj, 'epsilon') || isfield(obj, 'epsilon')
                if isprop(obj, 'epsilon_min') || isfield(obj, 'epsilon_min')
                    epsilon_min = obj.epsilon_min;
                else
                    epsilon_min = 0.01;
                end
                
                if isprop(obj, 'epsilon_decay') || isfield(obj, 'epsilon_decay')
                    epsilon_decay = obj.epsilon_decay;
                else
                    epsilon_decay = 0.995;
                end
                
                if obj.epsilon > epsilon_min
                    obj.epsilon = obj.epsilon * epsilon_decay;
                end
            end
            
            % 更新温度参数（如果存在）
            if isprop(obj, 'temperature') || isfield(obj, 'temperature')
                if isprop(obj, 'temperature_decay') || isfield(obj, 'temperature_decay')
                    temperature_decay = obj.temperature_decay;
                    if obj.temperature > 0.1
                        obj.temperature = obj.temperature * temperature_decay;
                    end
                end
            end
        end
        
        function save(obj, filename)
            % 保存模型
            agent_data = struct();
            agent_data.Q_table = obj.Q_table;
            agent_data.visit_count = obj.visit_count;
            agent_data.lr_scheduler = obj.lr_scheduler;
            agent_data.name = obj.name;
            agent_data.agent_type = obj.agent_type;
            save(filename, 'agent_data');
        end
        
        function load(obj, filename)
            % 加载模型
            if exist(filename, 'file')
                loaded = load(filename);
                obj.Q_table = loaded.agent_data.Q_table;
                obj.visit_count = loaded.agent_data.visit_count;
                obj.lr_scheduler = loaded.agent_data.lr_scheduler;
            end
        end
        function recordPerformance(obj, reward, td_error)
            % 记录性能历史
            if ~isfield(obj.performance_history, 'rewards')
                obj.performance_history.rewards = [];
                obj.performance_history.td_errors = [];
                obj.performance_history.radi = [];
                obj.performance_history.damage = [];
                obj.performance_history.success_rate = [];
                obj.performance_history.detection_rate = [];
            end
            
            obj.performance_history.rewards(end+1) = reward;
            obj.performance_history.td_errors(end+1) = abs(td_error);
            
            % 计算并记录其他性能指标
            obj.performance_history.radi(end+1) = obj.calculateRADI();
            obj.performance_history.damage(end+1) = obj.calculateDamage();
            obj.performance_history.success_rate(end+1) = obj.calculateSuccessRate();
            obj.performance_history.detection_rate(end+1) = obj.calculateDetectionRate();
        end
        
        function radi = calculateRADI(obj)
            % 计算RADI指标
            if isempty(obj.strategy_history)
                radi = 0.5;
            else
                current_strategy = obj.strategy_history(end, :);
                entropy = -sum(current_strategy .* log(current_strategy + eps));
                radi = entropy / log(length(current_strategy));
            end
        end
        
        function damage = calculateDamage(obj)
            % 计算损害程度
            if isempty(obj.performance_history.rewards)
                damage = 0.5;
            else
                recent_rewards = obj.performance_history.rewards(max(1, end-9):end);
                damage = 1 - mean(recent_rewards);
                damage = max(0, min(1, damage));
            end
        end
        
        function success_rate = calculateSuccessRate(obj)
            % 计算成功率
            if isempty(obj.performance_history.rewards)
                success_rate = 0.5;
            else
                recent_rewards = obj.performance_history.rewards(max(1, end-19):end);
                success_rate = mean(recent_rewards > 0);
            end
        end
        
        function detection_rate = calculateDetectionRate(obj)
            % 计算检测率
            if strcmp(obj.agent_type, 'defender')
                if isempty(obj.performance_history.rewards)
                    detection_rate = 0.8;
                else
                    recent_rewards = obj.performance_history.rewards(max(1, end-19):end);
                    detection_rate = mean(recent_rewards > 0.5);
                end
            else
                detection_rate = NaN; % 攻击者不适用
            end
        end
    end
    
    methods
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
            
            % 使用简单的哈希函数
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
    end
    
    methods (Access = private)
        
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