%% SARSAAgent.m - SARSA智能体实现 (修复版)
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
        % TODO: 验证此函数是否被使用
        % TODO: 验证此函数是否被使用
        function obj = SARSAAgent(name, agent_type, config, state_dim, action_dim)
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 改进的Q表初始化 - 使用乐观初始化
            initial_value = 5.0; % 提高初始值
            noise_level = 0.5;   % 增加噪声
            obj.Q_table = ones(state_dim, action_dim) * initial_value + ...
                          randn(state_dim, action_dim) * noise_level;
            obj.visit_count = zeros(state_dim, action_dim);
            
            % 默认使用epsilon-greedy策略，更稳定
            % obj.use_softmax = false; % 删除重复定义
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
            
            % 初始化更新计数器
            % 初始化新添加的属性
            % obj.use_softmax = false;     % 默认使用epsilon-greedy % 删除重复定义
            
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
                q_values = ones(size(q_values)) * 1.0;
            end
            
            % === SARSA特有的动作选择策略 ===
            if obj.use_softmax
                % Softmax选择
                temperature = max(0.1, obj.temperature);
                exp_values = exp(q_values / temperature);
                probabilities = exp_values / sum(exp_values);
                action_vec = probabilities;
                
                % SARSA特色：基于当前策略的噪声
                if rand() < 0.2
                    policy_noise = randn(1, obj.action_dim) * 0.05;
                    action_vec = action_vec + policy_noise;
                end
                
            else
                % Epsilon-greedy with SARSA twist
                if rand() < obj.epsilon
                    % SARSA探索：倾向于平衡分配
                    action_vec = ones(1, obj.action_dim) + rand(1, obj.action_dim) * 0.5;
                else
                    % 基于Q值的Boltzmann分布
                    beta = 2.0;  % 温度参数
                    weights = exp(beta * (q_values - max(q_values)));
                    action_vec = weights / sum(weights);
                    
                    % 添加策略噪声
                    noise = randn(1, obj.action_dim) * 0.08;
                    action_vec = action_vec + noise;
                end
            end
            
            % 确保非负并归一化
            action_vec = max(0, action_vec);
            if sum(action_vec) > 0
                action_vec = action_vec / sum(action_vec);
            else
                action_vec = ones(1, obj.action_dim) / obj.action_dim;
            end
            
            % 记录动作 - 修复：使用action_vec而不是未定义的action
            [~, dominant_action] = max(action_vec);
            if strcmp(obj.agent_type, 'defender') && length(action_vec) > 1
                if isempty(obj.strategy_history)
                    obj.strategy_history = action_vec;
                else
                    obj.strategy_history(end+1, :) = action_vec;
                end
            end
            
            % 记录参数历史
            obj.parameter_history.learning_rate(end+1) = obj.learning_rate;
            obj.parameter_history.epsilon(end+1) = obj.epsilon;
            obj.parameter_history.q_values(end+1) = mean(obj.Q_table(:));
        end

        function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
            % SARSA更新规则
            
            % 健壮性检查
            if isempty(action_vec)
                warning('SARSAAgent.update: action_vec is empty, auto-fixing...');
                action_vec = ones(1, 5);
            end
            action_vec = reshape(action_vec, 1, []);
            if isempty(state_vec)
                warning('SARSAAgent.update: state_vec is empty, auto-fixing...');
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
            
            % 计算Q表动作索引 - 使用第一个站点的动作作为主要索引
            primary_station = 1;
            resource_type = action_vec(primary_station);
            resource_type = max(1, min(n_resource_types, round(resource_type)));
            q_action_idx = (primary_station - 1) * n_resource_types + resource_type;
            
            % 确保Q表索引有效
            q_action_idx = max(1, min(obj.action_dim, q_action_idx));
            
            % 处理下一个动作
            if isempty(next_action_vec)
                % 如果没有提供下一个动作，使用当前策略选择
                next_q_values = obj.Q_table(next_state_idx, :);
                next_action_vec = obj.convertToStationActions(next_q_values, n_stations);
            end
            
            % 计算下一个动作的Q表索引
            next_resource_type = next_action_vec(primary_station);
            next_resource_type = max(1, min(n_resource_types, round(next_resource_type)));
            next_q_action_idx = (primary_station - 1) * n_resource_types + next_resource_type;
            next_q_action_idx = max(1, min(obj.action_dim, next_q_action_idx));
            
            % 计算TD误差
            current_q = obj.Q_table(state_idx, q_action_idx);
            next_q = obj.Q_table(next_state_idx, next_q_action_idx);
            td_error = reward + obj.discount_factor * next_q - current_q;
            
            % 更新Q值
            obj.Q_table(state_idx, q_action_idx) = current_q + obj.learning_rate * td_error;
            obj.visit_count(state_idx, q_action_idx) = obj.visit_count(state_idx, q_action_idx) + 1;
            
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
            stats.update_count = obj.update_count;
            
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
            else
                stats.current_learning_rate = obj.learning_rate;
            end
            
            stats.current_epsilon = obj.epsilon;
            
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
            if obj.epsilon > obj.epsilon_min
                obj.epsilon = obj.epsilon * obj.epsilon_decay;
            end
            
            % 更新温度参数（如果存在）
            if isprop(obj, 'temperature') || isfield(obj, 'temperature')
                if isprop(obj, 'temperature_decay') || isfield(obj, 'temperature_decay')
                    if obj.temperature > 0.1
                        obj.temperature = obj.temperature * obj.temperature_decay;
                    end
                end
            end
        end
        
        function action_vec = convertToStationActions(obj, q_values, n_stations)
            % 将Q值转换为站点级动作向量
            
            if isempty(q_values)
                action_vec = ones(1, n_stations);
                return;
            end
            
            n_resource_types = length(q_values) / n_stations;
            action_vec = zeros(1, n_stations);
            
            for station = 1:n_stations
                start_idx = (station - 1) * n_resource_types + 1;
                end_idx = station * n_resource_types;
                station_q_values = q_values(start_idx:end_idx);
                [~, best_resource] = max(station_q_values);
                action_vec(station) = best_resource;
            end
        end
        
        function recordPerformance(obj, reward, td_error)
            % 记录性能指标
            
            if ~isfield(obj.performance_history, 'rewards')
                obj.performance_history.rewards = [];
                obj.performance_history.td_errors = [];
            end
            
            obj.performance_history.rewards(end+1) = reward;
            obj.performance_history.td_errors(end+1) = td_error;
        end
        
        function save(obj, filename)
            % 保存智能体模型
            save_data = struct();
            save_data.Q_table = obj.Q_table;
            save_data.visit_count = obj.visit_count;
            save_data.name = obj.name;
            save_data.update_count = obj.update_count;
            save_data.lr_scheduler = obj.lr_scheduler;
            save(filename, 'save_data');
        end
        
        function load(obj, filename)
            % 加载智能体模型
            if exist(filename, 'file')
                load_data = load(filename);
                save_data = load_data.save_data;
                obj.Q_table = save_data.Q_table;
                obj.visit_count = save_data.visit_count;
                obj.name = save_data.name;
                obj.update_count = save_data.update_count;
                if isfield(save_data, 'lr_scheduler')
                    obj.lr_scheduler = save_data.lr_scheduler;
                end
            else
                error('模型文件不存在: %s', filename);
            end
        end
    end
end