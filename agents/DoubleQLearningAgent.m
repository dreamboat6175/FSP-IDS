%% DoubleQLearningAgent.m - Double Q-Learning智能体实现
classdef DoubleQLearningAgent < RLAgent
    properties
        Q1_table
        Q2_table
        visit_count
        use_softmax
        update_count
    end
    
    methods
        function obj = DoubleQLearningAgent(name, agent_type, config, state_dim, action_dim)
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            obj.Q1_table = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.Q2_table = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.visit_count = zeros(state_dim, action_dim);
            obj.use_softmax = false;
        end
        
        function action_vec = selectAction(obj, state_vec)
    % Double Q-Learning智能体的动作选择
    
    % 健壮性检查
    if isempty(state_vec)
        warning('DoubleQLearningAgent.selectAction: state_vec is empty, auto-fixing...');
        state_vec = ones(1, obj.state_dim);
    end
    state_vec = reshape(state_vec, 1, []);
    
    % 获取状态索引
    state_idx = obj.encodeState(mean(state_vec));
    
    % 获取组合Q值
    Q_combined = (obj.Q1_table + obj.Q2_table) / 2;
    q_values = Q_combined(state_idx, :);
    
    % 确保Q值有效
    if any(isnan(q_values)) || any(isinf(q_values))
        q_values = ones(size(q_values)) * 1.0;
    end
    
    % === Double Q-Learning特有的动作选择 ===
    if obj.use_softmax
        % 使用两个Q表的信息
        q1_values = obj.Q1_table(state_idx, :);
        q2_values = obj.Q2_table(state_idx, :);
        
        temperature = max(0.1, obj.temperature);
        
        % 组合两个Q表的softmax分布
        exp1 = exp(q1_values / temperature);
        exp2 = exp(q2_values / temperature);
        prob1 = exp1 / sum(exp1);
        prob2 = exp2 / sum(exp2);
        
        % 平均两个分布
        action_vec = (prob1 + prob2) / 2;
        
    else
        % Epsilon-greedy with double Q twist
        if rand() < obj.epsilon
            % 探索：结合两个Q表的不确定性
            q_diff = abs(obj.Q1_table(state_idx, :) - obj.Q2_table(state_idx, :));
            exploration_weights = 1 + q_diff / (max(q_diff) + 1e-10);
            action_vec = exploration_weights + rand(1, obj.action_dim) * 0.3;
        else
            % 利用：基于组合Q值
            action_vec = softmax(q_values * 3);  % 更sharp的分布
            
            % 添加基于Q值差异的噪声
            q_uncertainty = abs(obj.Q1_table(state_idx, :) - obj.Q2_table(state_idx, :));
            uncertainty_noise = q_uncertainty / (max(q_uncertainty) + 1e-10) * 0.1;
            action_vec = action_vec + uncertainty_noise;
        end
    end
    
    % 确保非负并归一化
    action_vec = max(0, action_vec);
    if sum(action_vec) > 0
        action_vec = action_vec / sum(action_vec);
    else
        action_vec = ones(1, obj.action_dim) / obj.action_dim;
    end
    
    % 记录动作
    [~, dominant_action] = max(action_vec);
    % obj.recordAction(state_idx, dominant_action);
end
        
        function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
            % Robust shape check
            if isempty(action_vec)
                warning('DoubleQLearningAgent.update: action_vec is empty, auto-fixing...');
                action_vec = ones(1, 5);
            end
            action_vec = reshape(action_vec, 1, []);
            if isempty(state_vec)
                warning('DoubleQLearningAgent.update: state_vec is empty, auto-fixing...');
                state_vec = ones(1, obj.state_dim);
            end
            state_vec = reshape(state_vec, 1, []);
            if ~isempty(next_state_vec)
                next_state_vec = reshape(next_state_vec, 1, []);
            end
            if ~isempty(next_action_vec)
                next_action_vec = reshape(next_action_vec, 1, []);
            end
            
            % 获取状态索引
            state_idx = obj.encodeState(mean(state_vec));
            next_state_idx = obj.encodeState(mean(next_state_vec));
            
            % 将站点级动作转换为Q表索引
            n_stations = length(action_vec);
            n_resource_types = obj.action_dim / n_stations;
            
            % 使用第一个站点的动作作为主要索引
            primary_station = 1;
            resource_type = action_vec(primary_station);
            resource_type = max(1, min(n_resource_types, round(resource_type)));
            q_action_idx = (primary_station - 1) * n_resource_types + resource_type;
            q_action_idx = max(1, min(obj.action_dim, q_action_idx));
            
            % Double Q-Learning更新
            if rand() < 0.5
                [~, best_action] = max(obj.Q1_table(next_state_idx, :));
                if isempty(next_action_vec)
                    td_error = reward + obj.discount_factor * obj.Q2_table(next_state_idx, best_action) ...
                              - obj.Q1_table(state_idx, q_action_idx);
                else
                    % 处理下一个动作
                    next_resource_type = next_action_vec(primary_station);
                    next_resource_type = max(1, min(n_resource_types, round(next_resource_type)));
                    next_q_action_idx = (primary_station - 1) * n_resource_types + next_resource_type;
                    next_q_action_idx = max(1, min(obj.action_dim, next_q_action_idx));
                    td_error = reward + obj.discount_factor * obj.Q2_table(next_state_idx, next_q_action_idx) ...
                              - obj.Q1_table(state_idx, q_action_idx);
                end
                obj.Q1_table(state_idx, q_action_idx) = obj.Q1_table(state_idx, q_action_idx) + ...
                                                      obj.learning_rate * td_error;
            else
                [~, best_action] = max(obj.Q2_table(next_state_idx, :));
                if isempty(next_action_vec)
                    td_error = reward + obj.discount_factor * obj.Q1_table(next_state_idx, best_action) ...
                              - obj.Q2_table(state_idx, q_action_idx);
                else
                    % 处理下一个动作
                    next_resource_type = next_action_vec(primary_station);
                    next_resource_type = max(1, min(n_resource_types, round(next_resource_type)));
                    next_q_action_idx = (primary_station - 1) * n_resource_types + next_resource_type;
                    next_q_action_idx = max(1, min(obj.action_dim, next_q_action_idx));
                    td_error = reward + obj.discount_factor * obj.Q1_table(next_state_idx, next_q_action_idx) ...
                              - obj.Q2_table(state_idx, q_action_idx);
                end
                obj.Q2_table(state_idx, q_action_idx) = obj.Q2_table(state_idx, q_action_idx) + ...
                                                      obj.learning_rate * td_error;
            end
            obj.visit_count(state_idx, q_action_idx) = obj.visit_count(state_idx, q_action_idx) + 1;
            
            % obj.recordReward(reward);
            obj.update_count = obj.update_count + 1;
        end
        
% 在 agents/QLearningAgent.m 文件中，在现有的 update() 方法之后添加以下方法：
% 
% 现有代码结构应该是：
%     methods
%         function obj = QLearningAgent(...)  % 构造函数
%         function action = selectAction(...)  % 选择动作
%         function update(...)                % 更新方法
%         
%         % === 在这里添加以下四个新方法 ===
        
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
        
        function save(obj, filename)
            if nargin < 2
                filename = sprintf('models/doubleq_%s_%s.mat', ...
                                 obj.agent_type, datestr(now, 'yyyymmdd_HHMMSS'));
            end
            [filepath, ~, ~] = fileparts(filename);
            if ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            save_data.Q1_table = obj.Q1_table;
            save_data.Q2_table = obj.Q2_table;
            save_data.visit_count = obj.visit_count;
            save_data.name = obj.name;
            save(filename, 'save_data');
        end
        
        function load(obj, filename)
            if exist(filename, 'file')
                load_data = load(filename);
                save_data = load_data.save_data;
                obj.Q1_table = save_data.Q1_table;
                obj.Q2_table = save_data.Q2_table;
                obj.visit_count = save_data.visit_count;
                obj.name = save_data.name;
            else
                error('模型文件不存在: %s', filename);
            end
        end
         function prob = softmax(x)
    % 计算softmax概率分布
    exp_x = exp(x - max(x));  % 数值稳定性
    prob = exp_x / sum(exp_x);
end

    function state_idx = encodeState(obj, state)
        if isempty(state) || ~isnumeric(state)
            state_idx = 1;
            return;
        end
        state = double(state(:));
        state(isnan(state)) = 0;
        state(isinf(state)) = 0;
        state_idx = mod(sum(state .* (1:numel(state))), obj.state_dim) + 1;
        state_idx = max(1, min(obj.state_dim, round(state_idx)));
    end

    function strategy = getStrategy(obj)
        % Double Q-Learning的策略提取
        if contains(obj.name, 'defender')
            n_stations = 10;
            n_resources = 5;
            strategy = zeros(1, n_stations);
            % 使用两个Q表的平均
            Q_combined = (obj.Q1_table + obj.Q2_table) / 2;
            for station = 1:n_stations
                station_values = [];
                for resource = 1:n_resources
                    action_idx = (station - 1) * n_resources + resource;
                    if action_idx <= obj.action_dim
                        avg_q = mean(Q_combined(:, action_idx));
                        station_values(end+1) = avg_q;
                    end
                end
                if ~isempty(station_values)
                    strategy(station) = mean(station_values);
                end
            end
            if sum(strategy) > 0
                strategy = strategy / sum(strategy);
            else
                strategy = ones(1, n_stations) / n_stations;
            end
        else
            Q_combined = (obj.Q1_table + obj.Q2_table) / 2;
            avg_q = mean(Q_combined, 1);
            if any(avg_q)
                % Double Q-Learning使用更保守的温度
                strategy = exp(avg_q * 0.8) / sum(exp(avg_q * 0.8));
            else
                strategy = ones(1, obj.action_dim) / obj.action_dim;
            end
        end
        strategy = strategy(:)';
    end
    end
end