%% SARSAAgent.m - SARSA智能体实现
classdef SARSAAgent < RLAgent
    properties
        Q_table
        visit_count
        use_softmax
        update_count
    end
    
    methods
        function obj = SARSAAgent(name, agent_type, config, state_dim, action_dim)
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            
            % 改进的Q表初始化 - 使用乐观初始化
            initial_value = 5.0; % 提高初始值
            noise_level = 0.5;   % 增加噪声
            obj.Q_table = ones(state_dim, action_dim) * initial_value + ...
                          randn(state_dim, action_dim) * noise_level;
            obj.visit_count = zeros(state_dim, action_dim);
            
            % 默认使用epsilon-greedy策略，更稳定
            obj.use_softmax = false;
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
    
    % 记录动作
    [~, dominant_action] = max(action_vec);
    % obj.recordAction(state_idx, dominant_action);
end


        function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
            % Robust shape check
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
            % action_vec包含每个站点的资源类型选择 (1-5)
            % 需要转换为Q表中的对应索引
            n_stations = length(action_vec);
            n_resource_types = obj.action_dim / n_stations;
            
            % 计算Q表动作索引 - 使用第一个站点的动作作为主要索引
            % 这是一个简化的方法，假设主要关注第一个站点的动作
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
            
            % obj.recordReward(reward);
            obj.update_count = obj.update_count + 1;
        end
        
        function policy = getPolicy(obj)
            policy = obj.Q_table;
        end
        
        function save(obj, filename)
            if nargin < 2
                filename = sprintf('models/sarsa_%s_%s.mat', ...
                                 obj.agent_type, datestr(now, 'yyyymmdd_HHMMSS'));
            end
            [filepath, ~, ~] = fileparts(filename);
            if ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            save_data.Q_table = obj.Q_table;
            save_data.visit_count = obj.visit_count;
            save_data.name = obj.name;
            save_data.agent_type = obj.agent_type;
            save_data.update_count = obj.update_count;
            save(filename, 'save_data');
        end
        
        function load(obj, filename)
            if exist(filename, 'file')
                load_data = load(filename);
                save_data = load_data.save_data;
                obj.Q_table = save_data.Q_table;
                obj.visit_count = save_data.visit_count;
                obj.name = save_data.name;
                obj.update_count = save_data.update_count;
            else
                error('模型文件不存在: %s', filename);
            end
        end
         function prob = softmax(x)
    % 计算softmax概率分布
    exp_x = exp(x - max(x));  % 数值稳定性
    prob = exp_x / sum(exp_x);
end

        function strategy = getStrategy(obj)
            % SARSA的策略提取
            if contains(obj.name, 'defender')
                n_stations = 10;
                n_resources = 5;
                strategy = zeros(1, n_stations);
                for station = 1:n_stations
                    station_values = [];
                    for resource = 1:n_resources
                        action_idx = (station - 1) * n_resources + resource;
                        if action_idx <= obj.action_dim
                            % SARSA使用当前策略的Q值
                            avg_q = mean(obj.Q_table(:, action_idx));
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
                avg_q = mean(obj.Q_table, 1);
                if any(avg_q)
                    strategy = exp(avg_q * 1.5) / sum(exp(avg_q * 1.5));  % 更高温度
                else
                    strategy = ones(1, obj.action_dim) / obj.action_dim;
                end
            end
            strategy = strategy(:)';
        end

        function action_vec = convertToStationActions(obj, q_values, n_stations)
            % 将Q值向量转换为每个站点的资源类型动作
            n_resource_types = obj.action_dim / n_stations;
            action_vec = zeros(1, n_stations);
            for s = 1:n_stations
                idx_start = (s-1)*n_resource_types + 1;
                idx_end = s*n_resource_types;
                [~, best_resource] = max(q_values(idx_start:idx_end));
                action_vec(s) = best_resource;
            end
        end

        function state_idx = encodeState(obj, state)
            % 将状态向量编码为索引（与QLearningAgent一致）
            if isempty(state) || ~isnumeric(state)
                state_idx = 1;
                return;
            end
            state = double(state(:));
            state(isnan(state)) = 0;
            state(isinf(state)) = 0;
            % 简单哈希
            state_idx = mod(sum(state .* (1:numel(state))), obj.state_dim) + 1;
            state_idx = max(1, min(obj.state_dim, round(state_idx))); % 保证索引有效
        end
    end
end
