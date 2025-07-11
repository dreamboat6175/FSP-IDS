%% SARSAAgent.m - SARSA智能体实现
classdef SARSAAgent < RLAgent
    properties
        Q_table
        visit_count
    end
    
    methods
        function obj = SARSAAgent(name, agent_type, config, state_dim, action_dim)
            obj@RLAgent(name, agent_type, config, state_dim, action_dim);
            obj.Q_table = zeros(state_dim, action_dim);
            obj.visit_count = zeros(state_dim, action_dim);
            obj.Q_table = obj.Q_table + randn(state_dim, action_dim) * 0.01;
            obj.use_softmax = true;
        end

        function action_vec = selectAction(obj, state_vec)
            % Robust shape check
            if isempty(state_vec) || numel(state_vec) ~= 5
                warning('SARSAAgent.selectAction: state_vec is empty or not length 5, auto-fixing...');
                state_vec = ones(1, 5);
            end
            state_vec = reshape(state_vec, 1, 5);
            % 输入: state_vec (1 x n_stations)
            % 输出: action_vec (1 x n_stations)
            n = length(state_vec);
            action_vec = zeros(1, n);
            for j = 1:n
                state_idx = obj.getStateIndex(state_vec(j));
                q_values = obj.Q_table(state_idx, :);
                if obj.use_softmax
                    action_vec(j) = obj.boltzmannAction(state_idx, q_values);
                else
                    action_vec(j) = obj.epsilonGreedyAction(state_idx, q_values);
                end
                obj.recordAction(state_idx, action_vec(j));
            end
        end

        function update(obj, state_vec, action_vec, reward, next_state_vec, next_action_vec)
            % Robust shape check
            if isempty(action_vec) || numel(action_vec) ~= 5
                warning('SARSAAgent.update: action_vec is empty or not length 5, auto-fixing...');
                action_vec = ones(1, 5);
            end
            action_vec = reshape(action_vec, 1, 5);
            if isempty(state_vec) || numel(state_vec) ~= 5
                warning('SARSAAgent.update: state_vec is empty or not length 5, auto-fixing...');
                state_vec = ones(1, 5);
            end
            state_vec = reshape(state_vec, 1, 5);
            if ~isempty(next_state_vec) && numel(next_state_vec) ~= 5
                warning('SARSAAgent.update: next_state_vec is not length 5, auto-fixing...');
                next_state_vec = ones(1, 5);
            end
            if ~isempty(next_state_vec)
                next_state_vec = reshape(next_state_vec, 1, 5);
            end
            if ~isempty(next_action_vec) && numel(next_action_vec) ~= 5
                warning('SARSAAgent.update: next_action_vec is not length 5, auto-fixing...');
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
                if isempty(next_action_vec)
                    next_a = obj.selectAction(next_state_vec(j));
                else
                    next_a = next_action_vec(j);
                end
                td_error = reward + obj.discount_factor * obj.Q_table(next_state_idx, next_a) ...
                           - obj.Q_table(state_idx, a);
                obj.Q_table(state_idx, a) = obj.Q_table(state_idx, a) + ...
                                       obj.learning_rate * td_error;
                obj.visit_count(state_idx, a) = obj.visit_count(state_idx, a) + 1;
            end
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
