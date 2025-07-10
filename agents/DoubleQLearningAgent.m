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
        
        function action = selectAction(obj, state)
            Q_combined = (obj.Q1_table + obj.Q2_table) / 2;
            q_values = Q_combined(state, :);
            action = obj.epsilonGreedyAction(state, q_values);
            obj.recordAction(state, action);
        end
        
        function update(obj, state, action, reward, next_state, ~)
            if rand() < 0.5
                [~, best_action] = max(obj.Q1_table(next_state, :));
                td_error = reward + obj.discount_factor * obj.Q2_table(next_state, best_action) ...
                          - obj.Q1_table(state, action);
                obj.Q1_table(state, action) = obj.Q1_table(state, action) + ...
                                            obj.learning_rate * td_error;
            else
                [~, best_action] = max(obj.Q2_table(next_state, :));
                td_error = reward + obj.discount_factor * obj.Q1_table(next_state, best_action) ...
                          - obj.Q2_table(state, action);
                obj.Q2_table(state, action) = obj.Q2_table(state, action) + ...
                                            obj.learning_rate * td_error;
            end
            obj.visit_count(state, action) = obj.visit_count(state, action) + 1;
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
