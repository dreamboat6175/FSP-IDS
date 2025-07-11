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
            % Robust shape check
            if isempty(state_vec) || numel(state_vec) ~= 5
                warning('DoubleQLearningAgent.selectAction: state_vec is empty or not length 5, auto-fixing...');
                state_vec = ones(1, 5);
            end
            state_vec = reshape(state_vec, 1, 5);
            % 输入: state_vec (1 x n_stations)
            % 输出: action_vec (1 x n_stations)
            n = length(state_vec);
            action_vec = zeros(1, n);
            for j = 1:n
                state_idx = obj.getStateIndex(state_vec(j));
                Q_combined = (obj.Q1_table + obj.Q2_table) / 2;
                q_values = Q_combined(state_idx, :);
                action_vec(j) = obj.epsilonGreedyAction(state_idx, q_values);
                obj.recordAction(state_idx, action_vec(j));
            end
        end
        
        function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
            % Robust shape check
            if isempty(action_vec) || numel(action_vec) ~= 5
                warning('DoubleQLearningAgent.update: action_vec is empty or not length 5, auto-fixing...');
                action_vec = ones(1, 5);
            end
            action_vec = reshape(action_vec, 1, 5);
            if isempty(state_vec) || numel(state_vec) ~= 5
                warning('DoubleQLearningAgent.update: state_vec is empty or not length 5, auto-fixing...');
                state_vec = ones(1, 5);
            end
            state_vec = reshape(state_vec, 1, 5);
            if ~isempty(next_state_vec) && numel(next_state_vec) ~= 5
                warning('DoubleQLearningAgent.update: next_state_vec is not length 5, auto-fixing...');
                next_state_vec = ones(1, 5);
            end
            if ~isempty(next_state_vec)
                next_state_vec = reshape(next_state_vec, 1, 5);
            end
            if ~isempty(next_action_vec) && numel(next_action_vec) ~= 5
                warning('DoubleQLearningAgent.update: next_action_vec is not length 5, auto-fixing...');
                next_action_vec = ones(1, 5);
            end
            if ~isempty(next_action_vec)
                next_action_vec = reshape(next_action_vec, 1, 5);
            end
            n = length(state_vec);
            for j = 1:n
                state_idx = obj.getStateIndex(state_vec(j));
                next_state_idx = obj.getStateIndex(next_state_vec(j));
                a = action_vec(j);
                if rand() < 0.5
                    [~, best_action] = max(obj.Q1_table(next_state_idx, :));
                    if isempty(next_action_vec)
                        td_error = reward + obj.discount_factor * obj.Q2_table(next_state_idx, best_action) ...
                                  - obj.Q1_table(state_idx, a);
                    else
                        td_error = reward + obj.discount_factor * obj.Q2_table(next_state_idx, next_action_vec(j)) ...
                                  - obj.Q1_table(state_idx, a);
                    end
                    obj.Q1_table(state_idx, a) = obj.Q1_table(state_idx, a) + ...
                                                obj.learning_rate * td_error;
                else
                    [~, best_action] = max(obj.Q2_table(next_state_idx, :));
                    if isempty(next_action_vec)
                        td_error = reward + obj.discount_factor * obj.Q1_table(next_state_idx, best_action) ...
                                  - obj.Q2_table(state_idx, a);
                    else
                        td_error = reward + obj.discount_factor * obj.Q1_table(next_state_idx, next_action_vec(j)) ...
                                  - obj.Q2_table(state_idx, a);
                    end
                    obj.Q2_table(state_idx, a) = obj.Q2_table(state_idx, a) + ...
                                                obj.learning_rate * td_error;
                end
                obj.visit_count(state_idx, a) = obj.visit_count(state_idx, a) + 1;
            end
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