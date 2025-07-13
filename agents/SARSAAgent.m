%% SARSAAgent.m - SARSA智能体实现
classdef SARSAAgent < RLAgent
    properties
        Q_table
        visit_count
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
            % 输入: state_vec (1 x state_dim)
            % 输出: action_vec (1 x n_stations)
            % Robust shape check
            if isempty(state_vec)
                warning('SARSAAgent.selectAction: state_vec is empty, auto-fixing...');
                state_vec = ones(1, obj.state_dim);
            end
            state_vec = reshape(state_vec, 1, []);
            
            % 获取状态索引
            state_idx = obj.getStateIndex(mean(state_vec));
            
            % 获取Q值
            q_values = obj.Q_table(state_idx, :);
            
            % 确保Q值有效
            if any(isnan(q_values)) || any(isinf(q_values))
                q_values = ones(size(q_values)) * 1.0;
            end
            
            % 使用辅助函数转换为站点级动作
            action_vec = obj.convertToStationActions(q_values, 5); % 固定为5个站点
            
            % 记录动作
            obj.recordAction(state_idx, action_vec(1));
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
            state_idx = obj.getStateIndex(mean(state_vec));
            next_state_idx = obj.getStateIndex(mean(next_state_vec));
            
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
            
            obj.recordReward(reward);
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
    end
end
