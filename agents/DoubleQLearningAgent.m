%% DoubleQLearningAgent.m - Double Q-Learning智能体实现
classdef DoubleQLearningAgent < RLAgent
    properties
        Q1_table
        Q2_table
        visit_count
    end
    
    methods
        function obj = DoubleQLearningAgent(name, agent_type, config, state_dim, action_dim)
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            obj.Q1_table = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.Q2_table = zeros(state_dim, action_dim) + randn(state_dim, action_dim) * 0.01;
            obj.visit_count = zeros(state_dim, action_dim);
        end
        
        function action_vec = selectAction(obj, state_vec)
            % 输入: state_vec (1 x state_dim)
            % 输出: action_vec (1 x n_stations)
            % Robust shape check
            if isempty(state_vec)
                warning('DoubleQLearningAgent.selectAction: state_vec is empty, auto-fixing...');
                state_vec = ones(1, obj.state_dim);
            end
            state_vec = reshape(state_vec, 1, []);
            
            % 获取状态索引
            state_idx = obj.getStateIndex(mean(state_vec));
            
            % 获取组合Q值
            Q_combined = (obj.Q1_table + obj.Q2_table) / 2;
            q_values = Q_combined(state_idx, :);
            
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
            state_idx = obj.getStateIndex(mean(state_vec));
            next_state_idx = obj.getStateIndex(mean(next_state_vec));
            
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
            
            obj.recordReward(reward);
            obj.update_count = obj.update_count + 1;
        end
        
        function policy = getPolicy(obj)
            policy = (obj.Q1_table + obj.Q2_table) / 2;
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
    end
end